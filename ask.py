#!/usr/bin/env python3

import os
import select
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from evdev import InputDevice, ecodes, list_devices
from openai import OpenAI

ENV_FILE = Path("~/.config/glossy.env").expanduser()
MIN_HOLD_SECONDS = 0.35
SYSTEM_PROMPT = (
    "You are a quick-reference tutor for a reader of technical books. Answer the "
    "user's question directly and concisely enough to be useful when spoken aloud. "
    "Balance high-level intuition with enough concrete technical depth to educate "
    "them properly, and briefly define unfamiliar jargon. Do not sacrifice correctness "
    "for brevity."
)


def load_config(path=ENV_FILE):
    if path.exists():
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.removeprefix("export ").partition("=")
            if not separator:
                raise RuntimeError(f"Invalid line in {path}: {raw_line!r}")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)

    missing = [name for name in ("OPENAI_API_KEY", "GLOSSY_MODEL") if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing {', '.join(missing)} in {path}")
    return os.environ["GLOSSY_MODEL"]


def find_keyboards():
    keyboards = []
    permission_denied = False
    for path in list_devices():
        try:
            device = InputDevice(path)
            if ecodes.KEY_CAPSLOCK in device.capabilities().get(ecodes.EV_KEY, []):
                keyboards.append(device)
            else:
                device.close()
        except PermissionError:
            permission_denied = True

    if not keyboards:
        reason = "permission denied" if permission_denied else "no keyboard found"
        raise RuntimeError(
            f"Cannot listen for Caps Lock ({reason}). Add this user to the input group "
            "and log out and back in."
        )
    return keyboards


def start_recording(path):
    path.unlink(missing_ok=True)
    return subprocess.Popen(
        [
            "arecord",
            "--quiet",
            "--file-type=wav",
            "--format=S16_LE",
            "--rate=16000",
            "--channels=1",
            str(path),
        ],
        stderr=subprocess.PIPE,
        text=True,
    )


def stop_recording(recorder, path):
    failed_early = recorder.poll() is not None
    if not failed_early:
        recorder.send_signal(signal.SIGINT)
    try:
        _, error = recorder.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        recorder.kill()
        recorder.communicate()
        raise RuntimeError("arecord did not stop")
    if failed_early and recorder.returncode:
        raise RuntimeError(error.strip() or "arecord failed")
    if not path.exists() or path.stat().st_size <= 44:
        raise RuntimeError(error.strip() or "No audio was recorded")


def speak(text):
    subprocess.run(
        ["spd-say", "--wait", "--pipe-mode"],
        input=text.replace("\n", " ") + "\n",
        text=True,
        stdout=subprocess.DEVNULL,
        check=True,
    )


def answer_question(client, model, audio_path):
    with audio_path.open("rb") as audio:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio
        ).text.strip()
    if not transcript:
        raise RuntimeError("Whisper returned an empty transcript")

    answer = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=transcript,
    ).output_text.strip()
    if not answer:
        raise RuntimeError("OpenAI returned an empty answer")
    speak(answer)


def report_error(error):
    print(f"Glossy: {error}", file=sys.stderr, flush=True)
    try:
        speak("Glossy failed. Check the service log.")
    except Exception:
        pass


def listen(client, model):
    keyboards = find_keyboards()
    audio_path = Path(tempfile.gettempdir()) / f"glossy-{os.getpid()}.wav"
    recorder = None
    started_at = 0.0
    print("Glossy is listening for Caps Lock.", flush=True)

    try:
        while True:
            readable, _, _ = select.select(keyboards, [], [])
            for keyboard in readable:
                for event in keyboard.read():
                    if event.type != ecodes.EV_KEY or event.code != ecodes.KEY_CAPSLOCK:
                        continue
                    if event.value == 1 and recorder is None:
                        recorder = start_recording(audio_path)
                        started_at = time.monotonic()
                        print("Recording...", flush=True)
                    elif event.value == 0 and recorder is not None:
                        held_for = time.monotonic() - started_at
                        try:
                            stop_recording(recorder, audio_path)
                            if held_for < MIN_HOLD_SECONDS:
                                print("Ignored short press.", flush=True)
                            else:
                                print("Answering...", flush=True)
                                answer_question(client, model, audio_path)
                                print("Ready.", flush=True)
                        except Exception as error:
                            report_error(error)
                        finally:
                            recorder = None
                            audio_path.unlink(missing_ok=True)
    finally:
        if recorder is not None and recorder.poll() is None:
            recorder.terminate()
        audio_path.unlink(missing_ok=True)
        for keyboard in keyboards:
            keyboard.close()


def main():
    try:
        model = load_config()
        listen(OpenAI(), model)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"Glossy: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
