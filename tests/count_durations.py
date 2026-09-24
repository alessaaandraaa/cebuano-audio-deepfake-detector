r"""
count_durations.py

Quick duration summary straight from the JSONL manifest -- no audio
files are opened.

Reports total clip count, how many clips fall under 1 second, and the
mean duration across the manifest.

Input:

    manifests/manifest.json

    One JSON object per line, each containing a "duration_sec" field.

For per-speaker duration stats read from the audio files themselves,
use tests/check_durations.py instead.

Usage:

    python tests\count_durations.py
"""

import json
from pathlib import Path

durations = []
with open("manifests/manifest.json") as f:
    for line in f:
        durations.append(json.loads(line)["duration_sec"])

under_1s = sum(1 for d in durations if d < 1.0)
print(f"Total clips: {len(durations)}")
print(f"Under 1s: {under_1s} ({100*under_1s/len(durations):.1f}%)")
print(f"Mean: {sum(durations)/len(durations):.2f}s")
