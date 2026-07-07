#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

VOICE_DIR = Path(__file__).parent / "voices"


def main():
    result = subprocess.run(
        [sys.executable, "-m", "piper.download_voices"],
        capture_output=True,
        text=True,
        check=True,
    )
    voices = [name for name in result.stdout.splitlines() if name.startswith("en_")]
    if not voices:
        raise SystemExit("No English Piper voices found")

    print("English Piper voices:\n")
    print(*voices, sep="\n")
    choice = input("\nVoice name: ").strip()
    if choice not in voices:
        raise SystemExit(f"Unknown English voice: {choice}")

    VOICE_DIR.mkdir(exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "piper.download_voices",
            "--download-dir",
            str(VOICE_DIR),
            choice,
        ],
        check=True,
    )
    pending = VOICE_DIR / "selected.tmp"
    pending.write_text(choice + "\n")
    pending.replace(VOICE_DIR / "selected")
    print(f"\nGlossy now uses {choice}.")


if __name__ == "__main__":
    main()
