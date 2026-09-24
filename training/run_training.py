r"""
run_training.py

Launches DeepFense training with this project's custom transforms
registered.

Why this wrapper exists:

    opus_codec and amrnb_codec register themselves through
    @register_transform in codec_opus_amr.py. That decorator only runs
    when the module is imported, and the `deepfense` CLI has no reason
    to import it -- so launching the CLI directly on a config that
    references those transforms fails with:

        KeyError: 'opus_codec' not found in Transform registry

    This script imports codec_opus_amr first, then hands control to
    DeepFense's own train command. Everything else -- config parsing,
    logging, checkpointing -- is unchanged.

Use the plain CLI for configs with no custom transforms:

    deepfense train --config training\configs\smoke.yaml

Use this wrapper for anything with codec augmentation:

    python training\run_training.py --config training\configs\train_main.yaml

Any extra arguments are passed straight through to `deepfense train`.

Usage:

    python training\run_training.py --config training\configs\train_main.yaml

    # verify registration without training
    python training\run_training.py --list-transforms
"""

import argparse
import sys
from pathlib import Path

# Make sibling modules importable regardless of the working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Importing this is the whole point -- it fires @register_transform
import codec_opus_amr  # noqa: F401  (imported for side effects)
import deepfense_patches


def list_transforms() -> int:
    """Print the Transform registry so you can confirm registration."""
    try:
        from deepfense.utils.registry import TRANSFORM_REGISTRY
    except ImportError as e:
        print(f"ERROR: deepfense not importable: {e}", file=sys.stderr)
        return 1

    names = sorted(TRANSFORM_REGISTRY.list())

    print("=" * 70)
    print("REGISTERED TRANSFORMS")
    print("=" * 70)

    for name in names:
        mark = ""
        if name in ("opus_codec", "amrnb_codec"):
            mark = "   <-- custom, from codec_opus_amr.py"
        print(f"  {name}{mark}")

    print()

    missing = [n for n in ("opus_codec", "amrnb_codec") if n not in names]

    if missing:
        print(f"RESULT: MISSING {missing}", file=sys.stderr)
        return 1

    print("RESULT: both custom transforms registered.")
    return 0


def run_train(passthrough: list) -> int:
    """Hand off to DeepFense's train command, in this process."""
    try:
        from deepfense.cli.commands.train import train
    except ImportError as e:
        print(
            f"ERROR: could not import deepfense train command: {e}\n"
            "Is deepfense installed in the active environment?",
            file=sys.stderr,
        )
        return 1

    # `train` is a click command; .main() parses argv-style arguments.
    # standalone_mode=False lets exceptions propagate instead of click
    # swallowing them into its own exit path.
    try:
        train.main(args=passthrough, standalone_mode=False)
    except SystemExit as e:
        return int(e.code or 0)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run DeepFense training with this project's custom "
            "codec transforms registered."
        ),
        add_help=True,
    )

    parser.add_argument(
        "--list-transforms",
        action="store_true",
        help=(
            "Print the Transform registry and exit. Use this to "
            "confirm opus_codec / amrnb_codec registered correctly."
        ),
    )

    known, passthrough = parser.parse_known_args()

    if known.list_transforms:
        sys.exit(list_transforms())

    if not passthrough:
        parser.print_help()
        print(
            "\nNothing to pass to deepfense. Did you forget --config?",
            file=sys.stderr,
        )
        sys.exit(2)

    if not codec_opus_amr.DEEPFENSE_AVAILABLE:
        print(
            "ERROR: codec_opus_amr could not import deepfense, so the "
            "custom transforms are NOT registered.\n"
            "Activate the environment where deepfense is installed.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Make the transform pipeline picklable so Windows DataLoader
    # workers survive spawn. Must happen before any loader is built.
    try:
        patched = deepfense_patches.apply()
        print(f"Applied deepfense patches: {', '.join(patched)}")
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(run_train(passthrough))


if __name__ == "__main__":
    main()
