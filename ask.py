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
VOICE_DIR = Path(__file__).parent / "voices"
DEFAULT_VOICE = "en_US-lessac-medium"
MIN_HOLD_SECONDS = 1
SYSTEM_PROMPT = (Path(__file__).parent / "system-prompt.md").read_text().strip()


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
            if ecodes.KEY_RIGHTALT in device.capabilities().get(ecodes.EV_KEY, []):
                keyboards.append(device)
            else:
                device.close()
        except PermissionError:
            permission_denied = True

    if not keyboards:
        reason = "permission denied" if permission_denied else "no keyboard found"
        raise RuntimeError(
            f"Cannot listen for Right Alt ({reason}). Add this user to the input group "
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


def play_blip():
    subprocess.run(
        [
            "canberra-gtk-play",
            "--id=audio-volume-change",
            "--description=Glossy recording",
        ],
        check=False,
    )


def speak(text):
    selection = VOICE_DIR / "selected"
    voice = selection.read_text().strip() if selection.exists() else DEFAULT_VOICE
    voice_model = VOICE_DIR / f"{voice}.onnx"
    if not voice_model.exists():
        raise RuntimeError(f"Piper voice not found: {voice_model}")

    speech_path = Path(tempfile.gettempdir()) / f"glossy-speech-{os.getpid()}.wav"
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "piper",
                "--model",
                str(voice_model),
                "--output-file",
                str(speech_path),
            ],
            input=text.replace("\n", " ") + "\n",
            text=True,
            check=True,
        )
        subprocess.run(["aplay", "--quiet", str(speech_path)], check=True)
    finally:
        speech_path.unlink(missing_ok=True)


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
    pressed_at = None
    print("Glossy is listening for Right Alt.", flush=True)

    try:
        while True:
            timeout = None
            if pressed_at is not None and recorder is None:
                remaining = MIN_HOLD_SECONDS - (time.monotonic() - pressed_at)
                if remaining <= 0:
                    play_blip()
                    recorder = start_recording(audio_path)
                    print("Recording...", flush=True)
                else:
                    timeout = remaining

            readable, _, _ = select.select(keyboards, [], [], timeout)
            for keyboard in readable:
                for event in keyboard.read():
                    if event.type != ecodes.EV_KEY or event.code != ecodes.KEY_RIGHTALT:
                        continue
                    if event.value == 1 and pressed_at is None:
                        pressed_at = time.monotonic()
                    elif event.value == 0 and pressed_at is not None:
                        try:
                            if recorder is None:
                                print("Ignored short press.", flush=True)
                            else:
                                stop_recording(recorder, audio_path)
                                print("Answering...", flush=True)
                                answer_question(client, model, audio_path)
                                print("Ready.", flush=True)
                        except Exception as error:
                            report_error(error)
                        finally:
                            pressed_at = None
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
