r"""
deepfense_patches.py

Runtime fixes for deepfense bugs that only bite on Windows.

Both patches exist for the same underlying reason. On Linux, DataLoader
workers are created with fork, which inherits memory rather than
pickling it, so unpicklable objects in the dataset or loader never
matter. On Windows, workers are created with spawn, which pickles
everything it sends to the child process. deepfense builds two things
with closures, and both of them die under spawn.

That matters here specifically because codec augmentation is CPU-bound
(~368 ms/clip measured). Forced onto num_workers: 0, a full epoch of
~79k clips would spend many hours in ffmpeg with the GPU idle. Getting
workers back is the difference between a multi-day run and a
multi-week one.

PATCH 1 -- picklable transform pipeline
---------------------------------------

deepfense.utils.registry.build_transforms_pipeline builds the pipeline
out of two unpicklable things:

    registry.py:126-128   a lambda wrapping each function transform
    registry.py:132-136   a `pipeline` closure returned at the end

which fails with:

    AttributeError: Can't pickle local object
        'build_transforms_pipeline.<locals>.pipeline'

The fix replaces both with module-level classes.

PATCH 2 -- picklable collate_fn
-------------------------------

data_utils.py:109 passes the collate function as a lambda:

    collate_fn=lambda b: collate_fn(b, max_pad=config.get("max_len"))

which fails the same way, one layer further in:

    AttributeError: Can't pickle local object
        'build_dataloader.<locals>.<lambda>'

This one only surfaces AFTER patch 1 is working, because the loader is
not built until the dataset is. The fix is functools.partial around the
module-level collate_fn, which pickles fine because both the function
and the bound kwarg are importable/serialisable by name.

While rebuilding the loader, patch 2 also exposes three DataLoader
arguments deepfense hardcodes. Defaults are chosen for a spawn platform
running CPU-bound augmentation, so THEY DIFFER FROM STOCK DEEPFENSE:

    persistent_workers   default True when num_workers > 0
                         Stock torch default is False, which tears down
                         and respawns every worker at each epoch
                         boundary. On Windows each respawn re-imports
                         torch in four processes -- tens of seconds per
                         epoch, for nothing.

    prefetch_factor      default 4 when num_workers > 0 (torch: 2)
                         Deeper queue so ffmpeg jitter does not stall
                         the GPU.

    pin_memory           default True
                         Faster host-to-device copies. ~1 MB per batch
                         at batch 4 x 64600 samples, so the memory cost
                         is negligible.

All three are overridable per split in the YAML if you want stock
behaviour back:

    train:
      persistent_workers: False
      prefetch_factor: 2
      pin_memory: False

Neither patch changes any numerical result -- same transforms, same
collation, same batches, same order for a given seed. They only change
how the work is distributed across processes.

Apply BEFORE any dataloader is built:

    import deepfense_patches
    deepfense_patches.apply()

run_training.py does this for you.

Usage (verify the patches apply cleanly, without training):

    python training\deepfense_patches.py --verify
"""

import argparse
import functools
import pickle
import sys

# ---------------------------------------------------------------------------
# PATCH 1 -- PICKLABLE PIPELINE PIECES
# ---------------------------------------------------------------------------


class BoundTransform:
    """
    Picklable stand-in for `lambda x: func(x, **kwargs)`.

    Used for transforms registered as plain functions (pad,
    load_audio, rawboost, ...).
    """

    def __init__(self, func, kwargs):
        self.func = func
        self.kwargs = kwargs

    def __call__(self, x):
        return self.func(x, **self.kwargs)

    def __repr__(self):
        name = getattr(self.func, "__name__", repr(self.func))
        return f"BoundTransform({name}, {self.kwargs})"


class TransformPipeline:
    """
    Picklable stand-in for the `pipeline` closure.

    Applies each transform in order, exactly as the original did.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def __repr__(self):
        return f"TransformPipeline({self.transforms!r})"


def build_transforms_pipeline(config_list):
    """
    Drop-in replacement for the deepfense function of the same name.

    Same behaviour, same error messages, but returns picklable objects
    so Windows spawn-based DataLoader workers survive.
    """
    if not config_list:
        return None

    from deepfense.utils.registry import TRANSFORM_REGISTRY

    transforms = []

    for cfg in config_list:
        cfg_copy = dict(cfg)
        transform_type = cfg_copy.pop("type")
        transform_obj = TRANSFORM_REGISTRY.get(transform_type)

        if isinstance(transform_obj, type):
            try:
                transforms.append(transform_obj(**cfg_copy))
            except Exception as e:
                raise ValueError(
                    f"Failed to instantiate transform {transform_type}: {e}"
                )
        elif callable(transform_obj):
            transforms.append(BoundTransform(transform_obj, cfg_copy))
        else:
            raise ValueError(
                f"Transform {transform_type} is neither a class nor a "
                "callable function."
            )

    return TransformPipeline(transforms)


# ---------------------------------------------------------------------------
# PATCH 2 -- PICKLABLE DATALOADER
# ---------------------------------------------------------------------------


def build_dataloader(config):
    """
    Drop-in replacement for deepfense.data.data_utils.build_dataloader.

    Identical to the original except:
      - collate_fn is a functools.partial, not a lambda (picklable)
      - persistent_workers / prefetch_factor / pin_memory are read from
        the split config, with spawn-friendly defaults

    See the module docstring for why those defaults differ from stock.
    """
    from torch.utils.data import DataLoader, DistributedSampler

    from deepfense.data.data_utils import collate_fn
    from deepfense.utils.registry import build_dataset

    dataset_name = config["dataset_type"]
    ds = build_dataset(dataset_name, cfg=config)

    if len(ds) == 0:
        raise ValueError(
            f"Dataset '{dataset_name}' is empty. Please check your data"
            " configuration:\n"
            "  - parquet_files:"
            f" {config.get('parquet_files', 'not specified')}\n"
            f"  - root_dir: {config.get('root_dir', 'not specified')}\n"
            f"  - label_map: {config.get('label_map', 'not specified')}"
        )

    batch_size = config.get("batch_size", 8)
    shuffle = config.get("shuffle", False)
    num_workers = config.get("num_workers", 0)

    sampler = None
    if config.get("distributed", False):
        sampler = DistributedSampler(
            ds,
            num_replicas=config.get("world_size", 1),
            rank=config.get("rank", 0),
            shuffle=shuffle,
        )
        shuffle = False

    # THE FIX: partial, not lambda. collate_fn is module-level, so this
    # pickles by name.
    collate = functools.partial(
        collate_fn, max_pad=config.get("max_len", None)
    )

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "collate_fn": collate,
        "num_workers": num_workers,
        "pin_memory": config.get("pin_memory", True),
    }

    # torch rejects both of these when num_workers == 0.
    if num_workers > 0:
        kwargs["persistent_workers"] = config.get("persistent_workers", True)
        kwargs["prefetch_factor"] = config.get("prefetch_factor", 4)

    return DataLoader(ds, **kwargs)


# ---------------------------------------------------------------------------
# APPLY
# ---------------------------------------------------------------------------


def apply() -> list:
    """
    Patch both picklable replacements into deepfense.

    Modules that do `from X import name` keep their own reference to
    the original, so rebinding on the defining module is not enough --
    every importing module has to be patched too. Concretely:

        build_transforms_pipeline  defined in utils/registry.py
                                   imported by data/detection_dataset.py

        build_dataloader           defined in data/data_utils.py
                                   imported by cli/commands/train.py

    cli.commands.train is imported here rather than waited for, because
    run_training.py does not import it until after this call.

    Returns the list of module paths that were patched.
    """
    patched = []

    # -- patch 1 --------------------------------------------------------
    try:
        import deepfense.utils.registry as registry

        registry.build_transforms_pipeline = build_transforms_pipeline
        patched.append("deepfense.utils.registry.build_transforms_pipeline")
    except ImportError as e:
        raise RuntimeError(f"deepfense not importable: {e}")

    try:
        import deepfense.data.detection_dataset as detection_dataset

        detection_dataset.build_transforms_pipeline = build_transforms_pipeline
        patched.append(
            "deepfense.data.detection_dataset.build_transforms_pipeline"
        )
    except ImportError:
        print(
            "WARNING: could not patch deepfense.data.detection_dataset; "
            "if workers still fail to pickle, set num_workers: 0",
            file=sys.stderr,
        )

    # -- patch 2 --------------------------------------------------------
    try:
        import deepfense.data.data_utils as data_utils

        data_utils.build_dataloader = build_dataloader
        patched.append("deepfense.data.data_utils.build_dataloader")
    except ImportError:
        print(
            "WARNING: could not patch deepfense.data.data_utils",
            file=sys.stderr,
        )

    try:
        import deepfense.cli.commands.train as train_cmd

        train_cmd.build_dataloader = build_dataloader
        patched.append("deepfense.cli.commands.train.build_dataloader")
    except ImportError:
        print(
            "WARNING: could not patch deepfense.cli.commands.train; this "
            "is the one that actually matters -- it does `from "
            "deepfense.data.data_utils import build_dataloader` at "
            "import time, so without it the collate lambda comes back.",
            file=sys.stderr,
        )

    return patched


# ---------------------------------------------------------------------------
# VERIFY
# ---------------------------------------------------------------------------


def _check_pipeline() -> int:
    """Prove the transform pipeline pickles and still works."""
    print("PATCH 1 -- transform pipeline")

    cfg = [
        {
            "type": "pad",
            "max_len": 64600,
            "pad_type": "repeat",
            "random_pad": False,
        }
    ]

    try:
        pipeline = build_transforms_pipeline(cfg)
    except Exception as e:
        print(f"  ERROR: could not build pipeline: {e}", file=sys.stderr)
        return 1

    print(f"  built:   {pipeline!r}")

    try:
        blob = pickle.dumps(pipeline)
    except Exception as e:
        print(f"  FAIL:    not picklable -- {e}", file=sys.stderr)
        return 1

    restored = pickle.loads(blob)
    print(f"  pickled: {len(blob)} bytes, restored OK")

    try:
        import numpy as np

        out = restored(np.zeros(16000, dtype=np.float32))
        print(f"  applied: 16000 -> {len(out)} samples")

        if len(out) != 64600:
            print(
                f"  FAIL:    expected 64600 samples, got {len(out)}",
                file=sys.stderr,
            )
            return 1
    except ImportError:
        print("  (numpy missing, skipped functional check)")

    print("  PASS")
    return 0


def _check_collate() -> int:
    """
    Prove the collate partial pickles.

    This is the exact object that blew up as
    'build_dataloader.<locals>.<lambda>'. Building a real DataLoader
    would need a real dataset, so this checks the one piece that was
    unpicklable, in isolation.
    """
    print()
    print("PATCH 2 -- collate function")

    try:
        from deepfense.data.data_utils import collate_fn
    except ImportError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return 1

    collate = functools.partial(collate_fn, max_pad=64600)

    try:
        blob = pickle.dumps(collate)
    except Exception as e:
        print(f"  FAIL:    not picklable -- {e}", file=sys.stderr)
        return 1

    restored = pickle.loads(blob)
    print(f"  pickled: {len(blob)} bytes, restored OK")

    try:
        import torch

        batch = [
            {
                "x": torch.zeros(64600),
                "label": torch.tensor(1),
                "dataset_name": "ceb_pld",
                "ID": f"utt{i}",
            }
            for i in range(4)
        ]
        out = restored(batch)
        shape = tuple(out["x"].shape)
        print(f"  applied: 4 items -> x{shape}, keys {sorted(out)}")

        if shape != (4, 64600):
            print(
                f"  FAIL:    expected (4, 64600), got {shape}",
                file=sys.stderr,
            )
            return 1
    except ImportError:
        print("  (torch missing, skipped functional check)")

    print("  PASS")
    return 0


def _check_dataloader_signature() -> int:
    """Confirm train.py now points at the patched builder."""
    print()
    print("PATCH 2 -- rebind check")

    try:
        import deepfense.cli.commands.train as train_cmd
        import deepfense.data.data_utils as data_utils
    except ImportError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return 1

    ok = True

    for label, mod in (
        ("data_utils", data_utils),
        ("cli.commands.train", train_cmd),
    ):
        bound = getattr(mod, "build_dataloader", None)

        if bound is build_dataloader:
            print(f"  {label}: patched")
        else:
            print(f"  {label}: NOT PATCHED -> {bound!r}", file=sys.stderr)
            ok = False

    if not ok:
        return 1

    print("  PASS")
    return 0


def verify() -> int:
    print("=" * 70)
    print("DEEPFENSE PATCH VERIFICATION")
    print("=" * 70)

    try:
        patched = apply()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    for module in patched:
        print(f"  patched: {module}")

    print()

    rc = 0
    rc |= _check_pipeline()
    rc |= _check_collate()
    rc |= _check_dataloader_signature()

    print()

    if rc:
        print(
            "RESULT: FAILED -- set num_workers: 0 in your configs and "
            "expect a much slower run.",
            file=sys.stderr,
        )
        return 1

    print("RESULT: all patches good -- num_workers > 0 is safe.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Windows compatibility patches for deepfense. "
            "Import and call apply() before building dataloaders."
        )
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Apply the patches and confirm both the transform pipeline "
            "and the collate function pickle, which is what Windows "
            "DataLoader workers need."
        ),
    )

    args = parser.parse_args()

    if args.verify:
        sys.exit(verify())

    parser.print_help()


if __name__ == "__main__":
    main()
