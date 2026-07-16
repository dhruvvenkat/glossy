#!/usr/bin/env python3
from faster_whisper import WhisperModel
from openai import OpenAI

import sys

from listener import listen
from settings import MODEL_DIR, load_environment, load_settings
from threads import ThreadStore


def main():
    try:
        load_environment()
        settings = load_settings()
        transcriber = WhisperModel(
            settings["transcription_model"],
            device="cpu",
            compute_type="int8",
            download_root=str(MODEL_DIR),
        )
        listen(OpenAI(), transcriber, settings, ThreadStore())
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"Glossy: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
