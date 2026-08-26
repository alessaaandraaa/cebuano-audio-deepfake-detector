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