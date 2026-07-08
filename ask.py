#!/usr/bin/env python3

import errno
import json
import math
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from array import array
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
os.environ["HF_HOME"] = str(MODEL_DIR / ".cache")

from evdev import InputDevice, ecodes, list_devices
from faster_whisper import WhisperModel
from openai import OpenAI
import webrtcvad

ENV_FILE = Path("~/.config/glossy.env").expanduser()
CONFIG_FILE = Path(__file__).parent / "config.json"
VOICE_DIR = Path(__file__).parent / "voices"
VISUALIZER_SCRIPT = Path(__file__).parent / "visualizer.py"
START_BLIP_SOUND = Path(__file__).parent / "blip.mp3"
STOP_BLIP_SOUND = Path(__file__).parent / "blip-reversed.mp3"
DEFAULT_VOICE = "en_US-lessac-medium"
RECONNECT_SECONDS = 5
TRANSCRIPT_PREVIEW_SECONDS = 1
SYSTEM_PROMPT = (Path(__file__).parent / "system-prompt.md").read_text().strip()


def load_settings(path=CONFIG_FILE):
    try:
        settings = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot load {path}: {error}") from error

    expected = {
        "model",
        "reasoning_effort",
        "transcription_model",
        "transcription_beam_size",
        "hold_seconds",
        "button",
        "visualizer_sensitivity",
        "speech_rms_threshold",
        "minimum_speech_seconds",
        "vad_aggressiveness",
        "speech_snr_ratio",
    }
    if not isinstance(settings, dict) or set(settings) != expected:
        raise RuntimeError(f"{path} must contain exactly: {', '.join(sorted(expected))}")
    if not isinstance(settings["model"], str) or not settings["model"].strip():
        raise RuntimeError("model must be a non-empty string")
    efforts = {None, "none", "low", "medium", "high", "xhigh"}
    if settings["reasoning_effort"] not in efforts:
        raise RuntimeError(
            "reasoning_effort must be null, none, low, medium, high, or xhigh"
        )
    if (
        not isinstance(settings["transcription_model"], str)
        or not settings["transcription_model"].strip()
    ):
        raise RuntimeError("transcription_model must be a non-empty string")
    beam_size = settings["transcription_beam_size"]
    if isinstance(beam_size, bool) or not isinstance(beam_size, int) or beam_size < 1:
        raise RuntimeError("transcription_beam_size must be a positive integer")
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
    sensitivity = settings["visualizer_sensitivity"]
    if (
        isinstance(sensitivity, bool)
        or not isinstance(sensitivity, (int, float))
        or not math.isfinite(sensitivity)
        or sensitivity <= 0
    ):
        raise RuntimeError("visualizer_sensitivity must be a positive number")
    speech_threshold = settings["speech_rms_threshold"]
    if (
        isinstance(speech_threshold, bool)
        or not isinstance(speech_threshold, (int, float))
        or not math.isfinite(speech_threshold)
        or speech_threshold <= 0
    ):
        raise RuntimeError("speech_rms_threshold must be a positive number")
    minimum_speech = settings["minimum_speech_seconds"]
    if (
        isinstance(minimum_speech, bool)
        or not isinstance(minimum_speech, (int, float))
        or not math.isfinite(minimum_speech)
        or minimum_speech < 0
    ):
        raise RuntimeError("minimum_speech_seconds must be a non-negative number")
    aggressiveness = settings["vad_aggressiveness"]
    if isinstance(aggressiveness, bool) or aggressiveness not in range(4):
        raise RuntimeError("vad_aggressiveness must be an integer from 0 to 3")
    snr_ratio = settings["speech_snr_ratio"]
    if (
        isinstance(snr_ratio, bool)
        or not isinstance(snr_ratio, (int, float))
        or not math.isfinite(snr_ratio)
        or snr_ratio <= 1
    ):
        raise RuntimeError("speech_snr_ratio must be a number greater than 1")
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


def find_keyboards(button_code):
    keyboards = []
    for path in list_devices():
        try:
            device = InputDevice(path)
            if button_code in device.capabilities().get(ecodes.EV_KEY, []):
                keyboards.append(device)
            else:
                device.close()
        except OSError:
            pass
    return keyboards


def wait_for_keyboards(button_code, button_name):
    announced = False
    while True:
        keyboards = find_keyboards(button_code)
        if keyboards:
            return keyboards
        if not announced:
            print(
                f"Glossy: no accessible keyboard supports {button_name}; "
                f"checking every {RECONNECT_SECONDS} seconds.",
                file=sys.stderr,
                flush=True,
            )
            announced = True
        time.sleep(RECONNECT_SECONDS)


def keyboards_connected(keyboards):
    try:
        for keyboard in keyboards:
            keyboard.active_keys()
        return True
    except OSError:
        return False


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


def start_visualizer(audio_path, sensitivity):
    return subprocess.Popen(
        [sys.executable, str(VISUALIZER_SCRIPT), str(audio_path), str(sensitivity)]
    )


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


# path to differentiate background noise (ex. if the user accidentally holds down the button) from actual voice
# continuing to edit
# TODO: add a feature to end the recording after an extended period of time (45-secs to one minute) but make it configurable in json
def has_speech(path, rms_threshold, minimum_seconds, aggressiveness, snr_ratio):
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise RuntimeError("Expected 16-bit mono recording")
        sample_rate = audio.getframerate()
        if sample_rate not in {8000, 16000, 32000, 48000}:
            raise RuntimeError("WebRTC VAD requires an 8, 16, 32, or 48 kHz recording")
        frames_per_window = sample_rate * 30 // 1000
        required_windows = max(1, math.ceil(minimum_seconds / 0.03))
        vad = webrtcvad.Vad(aggressiveness)
        windows = []
        while len(data := audio.readframes(frames_per_window)) == frames_per_window * 2:
            samples = array("h")
            samples.frombytes(data)
            if sys.byteorder != "little":
                samples.byteswap()
            rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
            windows.append((rms, vad.is_speech(data, sample_rate)))

    calibration_windows = min(5, len(windows))
    noise_rms = (
        sum(rms for rms, _ in windows[:calibration_windows]) / calibration_windows
        if calibration_windows
        else 0
    )
    effective_threshold = max(rms_threshold, noise_rms * snr_ratio)
    longest_run = current_run = 0
    for rms, is_voice in windows[calibration_windows:]:
        current_run = current_run + 1 if is_voice and rms >= effective_threshold else 0
        longest_run = max(longest_run, current_run)

    print(
        f"Glossy: local VAD noise={noise_rms:.0f}, threshold={effective_threshold:.0f}, "
        f"longest_speech={longest_run * 30}ms.",
        flush=True,
    )
    return longest_run >= required_windows


def transcribe_audio(transcriber, settings, audio_path):
    segments, _ = transcriber.transcribe(
        str(audio_path),
        language="en",
        beam_size=settings["transcription_beam_size"],
    )
    return "".join(segment.text for segment in segments).strip()


def stream_transcript(transcriber, settings, audio_path, stopped):
    preview_path = Path(tempfile.gettempdir()) / f"glossy-preview-{os.getpid()}.wav"
    previous = ""
    try:
        # ponytail: re-transcribes the growing clip; use streaming ASR for long holds.
        while not stopped.wait(TRANSCRIPT_PREVIEW_SECONDS):
            try:
                data = audio_path.read_bytes()
                if len(data) <= 44:
                    continue
                with wave.open(str(preview_path), "wb") as preview:
                    preview.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    preview.writeframes(data[44:])
                transcript = transcribe_audio(transcriber, settings, preview_path)
            except OSError:
                continue
            except Exception as error:
                print(
                    f"Glossy: live transcript stopped: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                return
            if transcript and transcript != previous:
                print(f"Glossy heard: {transcript}", flush=True)
                previous = transcript
    finally:
        preview_path.unlink(missing_ok=True)


def start_transcript_stream(transcriber, settings, audio_path):
    stopped = threading.Event()
    thread = threading.Thread(
        target=stream_transcript,
        args=(transcriber, settings, audio_path, stopped),
        daemon=True,
    )
    thread.start()
    return stopped, thread


def stop_transcript_stream(stream):
    stopped, thread = stream
    stopped.set()
    thread.join()


def answer_question(client, transcriber, settings, audio_path):
    if not has_speech(
        audio_path,
        settings["speech_rms_threshold"],
        settings["minimum_speech_seconds"],
        settings["vad_aggressiveness"],
        settings["speech_snr_ratio"],
    ):
        print("Glossy: no speech detected; skipped OpenAI.", flush=True)
        return False

    transcript = transcribe_audio(transcriber, settings, audio_path)
    if not transcript:
        raise RuntimeError("Local Whisper returned an empty transcript")
    print(f"Glossy transcript: {transcript}", flush=True)

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
    return True


def report_error(error):
    print(f"Glossy: {error}", file=sys.stderr, flush=True)
    try:
        speak("Glossy failed. Check the service log.")
    except Exception:
        pass


def listen_connected(
    client, transcriber, settings, keyboards, button_code, button_name
):
    audio_path = Path(tempfile.gettempdir()) / f"glossy-{os.getpid()}.wav"
    recorder = None
    transcript_stream = None
    visualizer = None
    pressed_at = None
    print(f"Glossy is listening for {button_name}.", flush=True)

    try:
        while True:
            timeout = RECONNECT_SECONDS
            if pressed_at is not None and recorder is None:
                held_for = time.monotonic() - pressed_at
                remaining = settings["hold_seconds"] - held_for
                if remaining <= 0:
                    play_blip(START_BLIP_SOUND)
                    recorder = start_recording(audio_path)
                    transcript_stream = start_transcript_stream(
                        transcriber, settings, audio_path
                    )
                    visualizer = start_visualizer(
                        audio_path, settings["visualizer_sensitivity"]
                    )
                    print("Recording...", flush=True)
                else:
                    timeout = min(timeout, remaining)

            readable, _, _ = select.select(keyboards, [], [], timeout)
            if not readable and not keyboards_connected(keyboards):
                raise OSError(errno.ENODEV, "keyboard disconnected")
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
                                stop_transcript_stream(transcript_stream)
                                transcript_stream = None
                                stop_visualizer(visualizer)
                                visualizer = None
                                play_blip(STOP_BLIP_SOUND)
                                print("Answering...", flush=True)
                                answer_question(
                                    client, transcriber, settings, audio_path
                                )
                                print("Ready.", flush=True)
                        except Exception as error:
                            report_error(error)
                        finally:
                            if transcript_stream is not None:
                                stop_transcript_stream(transcript_stream)
                                transcript_stream = None
                            if visualizer is not None:
                                stop_visualizer(visualizer)
                                visualizer = None
                            pressed_at = None
                            recorder = None
                            audio_path.unlink(missing_ok=True)
    finally:
        if transcript_stream is not None:
            stop_transcript_stream(transcript_stream)
        if visualizer is not None:
            stop_visualizer(visualizer)
        if recorder is not None and recorder.poll() is None:
            recorder.terminate()
            try:
                recorder.wait(timeout=1)
            except subprocess.TimeoutExpired:
                recorder.kill()
                recorder.wait()
        audio_path.unlink(missing_ok=True)
        for keyboard in keyboards:
            keyboard.close()


def listen(client, transcriber, settings):
    button_name = settings["button"]
    button_code = getattr(ecodes, button_name)
    while True:
        keyboards = wait_for_keyboards(button_code, button_name)
        try:
            listen_connected(
                client, transcriber, settings, keyboards, button_code, button_name
            )
        except OSError as error:
            if error.errno not in {errno.ENODEV, errno.EBADF}:
                raise
            print(
                "Glossy: keyboard disconnected; waiting to reconnect.",
                file=sys.stderr,
                flush=True,
            )


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
        listen(OpenAI(), transcriber, settings)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"Glossy: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
