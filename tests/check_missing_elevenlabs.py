"""
check_missing_elevenlabs.py

Scans the ElevenLabs output directory and cross-references it against
the original transcripts to find any missing/failed generations.
Accounts for the .2.wav suffix and ignores files that were intentionally
skipped because they are reference clips.

Output:
    manifests/missing_elevenlabs.txt
"""

import sys
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS_ROOT = PROJECT_ROOT / "transcripts"
ELEVENLABS_ROOT = PROJECT_ROOT / "data" / "processed" / "elevenlabs"
REFERENCE_ROOT = PROJECT_ROOT / "elevenlabs-reference"
OUTPUT_FILE = PROJECT_ROOT / "manifests" / "missing_elevenlabs.txt"

def main():
    if not ELEVENLABS_ROOT.exists():
        print(f"ERROR: ElevenLabs output folder not found at {ELEVENLABS_ROOT}")
        sys.exit(1)

    # Only check speakers that have an output folder created
    speaker_dirs = sorted(d for d in ELEVENLABS_ROOT.iterdir() if d.is_dir())
    
    if not speaker_dirs:
        print("No speaker folders found in the ElevenLabs directory.")
        sys.exit(0)

    print(f"Checking {len(speaker_dirs)} speakers for missing files...\n")
    
    missing_utterances = []
    total_expected = 0
    total_found = 0
    total_reference = 0

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        transcript_path = TRANSCRIPTS_ROOT / f"{speaker_id}.txt"
        
        if not transcript_path.exists():
            print(f"WARNING: No transcript found for {speaker_id}. Skipping.")
            continue
            
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                    
                wav_filename, text = line.split("\t", 1)
                
                # Check if it was intentionally skipped as a reference clip
                ref_path = REFERENCE_ROOT / speaker_id / wav_filename
                if ref_path.exists():
                    total_reference += 1
                    continue
                    
                total_expected += 1
                
                # Construct the expected ElevenLabs .2.wav filename
                base_name = Path(wav_filename).stem
                expected_generated_name = f"{base_name}.2.wav"
                generated_path = speaker_dir / expected_generated_name
                
                if generated_path.exists():
                    total_found += 1
                else:
                    missing_utterances.append((speaker_id, wav_filename, text))

    # --- Write Results ---
    if missing_utterances:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for spk, wav, txt in missing_utterances:
                f.write(f"{spk}\t{wav}\t{txt}\n")
                
    # --- Summary ---
    print("=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)
    print(f"Total Expected (excluding refs) : {total_expected}")
    print(f"Total Successfully Generated    : {total_found}")
    print(f"Total Reference Clips Skipped   : {total_reference}")
    print(f"Total Missing / Failed          : {len(missing_utterances)}")
    print("-" * 60)
    
    if missing_utterances:
        print(f"Missing files logged to: {OUTPUT_FILE.resolve()}")
        print("You can use this file with a targeted regeneration script.")
    else:
        print("All clear! No files are missing.")

if __name__ == "__main__":
    main()