#!/usr/bin/env python3

import json
import math
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
CONFIG_FILE = Path(__file__).parent / "config.json"
VOICE_DIR = Path(__file__).parent / "voices"
VISUALIZER_SCRIPT = Path(__file__).parent / "visualizer.py"
START_BLIP_SOUND = Path(__file__).parent / "blip.mp3"
STOP_BLIP_SOUND = Path(__file__).parent / "blip-reversed.mp3"
DEFAULT_VOICE = "en_US-lessac-medium"
SYSTEM_PROMPT = (Path(__file__).parent / "system-prompt.md").read_text().strip()


def load_settings(path=CONFIG_FILE):
    try:
        settings = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot load {path}: {error}") from error

    expected = {"model", "reasoning_effort", "hold_seconds", "button"}
    if not isinstance(settings, dict) or set(settings) != expected:
        raise RuntimeError(f"{path} must contain exactly: {', '.join(sorted(expected))}")
    if not isinstance(settings["model"], str) or not settings["model"].strip():
        raise RuntimeError("model must be a non-empty string")
    efforts = {None, "none", "low", "medium", "high", "xhigh"}
    if settings["reasoning_effort"] not in efforts:
        raise RuntimeError(
            "reasoning_effort must be null, none, low, medium, high, or xhigh"
        )
    hold_seconds = settings["hold_seconds"]
    if (
        isinstance(hold_seconds, bool)
        or not isinstance(hold_seconds, (int, float))
        or not math.isfinite(hold_seconds)
        or hold_seconds < 0
    ):
        raise RuntimeError("hold_seconds must be a non-negative number")
    if not isinstance(settings["button"], str) or not isinstance(
        getattr(ecodes, settings["button"], None), int
    ):
        raise RuntimeError("button must be a Linux evdev key name such as KEY_RIGHTALT")
    return settings


def load_environment(path=ENV_FILE):
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

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"Missing OPENAI_API_KEY in {path}")


def find_keyboards(button_code, button_name):
    keyboards = []
    permission_denied = False
    for path in list_devices():
        try:
            device = InputDevice(path)
            if button_code in device.capabilities().get(ecodes.EV_KEY, []):
                keyboards.append(device)
            else:
                device.close()
        except PermissionError:
            permission_denied = True

    if not keyboards:
        reason = "permission denied" if permission_denied else "no keyboard found"
        raise RuntimeError(
            f"Cannot listen for {button_name} ({reason}). Add this user to the "
            "input group and log out and back in."
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


def start_visualizer(audio_path):
    return subprocess.Popen([sys.executable, str(VISUALIZER_SCRIPT), str(audio_path)])


def stop_visualizer(visualizer):
    if visualizer.poll() is None:
        visualizer.terminate()
        try:
            visualizer.wait(timeout=1)
        except subprocess.TimeoutExpired:
            visualizer.kill()
            visualizer.wait()


def play_blip(sound):
    subprocess.run(["paplay", str(sound)], check=False)


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


def answer_question(client, settings, audio_path):
    with audio_path.open("rb") as audio:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio
        ).text.strip()
    if not transcript:
        raise RuntimeError("Whisper returned an empty transcript")

    request = dict(
        model=settings["model"],
        instructions=SYSTEM_PROMPT,
        input=transcript,
    )
    if settings["reasoning_effort"] is not None:
        request["reasoning"] = {"effort": settings["reasoning_effort"]}
    answer = client.responses.create(**request).output_text.strip()
    if not answer:
        raise RuntimeError("OpenAI returned an empty answer")
    speak(answer)


def report_error(error):
    print(f"Glossy: {error}", file=sys.stderr, flush=True)
    try:
        speak("Glossy failed. Check the service log.")
    except Exception:
        pass


def listen(client, settings):
    button_name = settings["button"]
    button_code = getattr(ecodes, button_name)
    keyboards = find_keyboards(button_code, button_name)
    audio_path = Path(tempfile.gettempdir()) / f"glossy-{os.getpid()}.wav"
    recorder = None
    visualizer = None
    pressed_at = None
    print(f"Glossy is listening for {button_name}.", flush=True)

    try:
        while True:
            timeout = None
            if pressed_at is not None and recorder is None:
                held_for = time.monotonic() - pressed_at
                remaining = settings["hold_seconds"] - held_for
                if remaining <= 0:
                    play_blip(START_BLIP_SOUND)
                    recorder = start_recording(audio_path)
                    visualizer = start_visualizer(audio_path)
                    print("Recording...", flush=True)
                else:
                    timeout = remaining

            readable, _, _ = select.select(keyboards, [], [], timeout)
            for keyboard in readable:
                for event in keyboard.read():
                    if event.type != ecodes.EV_KEY or event.code != button_code:
                        continue
                    if event.value == 1 and pressed_at is None:
                        pressed_at = time.monotonic()
                    elif event.value == 0 and pressed_at is not None:
                        try:
                            if recorder is None:
                                print("Ignored short press.", flush=True)
                            else:
                                stop_recording(recorder, audio_path)
                                stop_visualizer(visualizer)
                                visualizer = None
                                play_blip(STOP_BLIP_SOUND)
                                print("Answering...", flush=True)
                                answer_question(client, settings, audio_path)
                                print("Ready.", flush=True)
                        except Exception as error:
                            report_error(error)
                        finally:
                            if visualizer is not None:
                                stop_visualizer(visualizer)
                                visualizer = None
                            pressed_at = None
                            recorder = None
                            audio_path.unlink(missing_ok=True)
    finally:
        if visualizer is not None:
            stop_visualizer(visualizer)
        if recorder is not None and recorder.poll() is None:
            recorder.terminate()
        audio_path.unlink(missing_ok=True)
        for keyboard in keyboards:
            keyboard.close()


def main():
    try:
        load_environment()
        listen(OpenAI(), load_settings())
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"Glossy: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
