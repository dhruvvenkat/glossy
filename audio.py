from evdev import ecodes
import webrtcvad

import math
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import wave
from array import array
from pathlib import Path

VOICE_DIR = Path(__file__).parent / "voices"
START_BLIP_SOUND = Path(__file__).parent / "blip.mp3"
STOP_BLIP_SOUND = Path(__file__).parent / "blip-reversed.mp3"
DEFAULT_VOICE = "en_US-lessac-medium"
TRANSCRIPT_PREVIEW_SECONDS = 0.25


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


def play_blip(sound):
    subprocess.run(["paplay", str(sound)], check=False)


def speech_cancelled(keyboards):
    readable, _, _ = select.select(keyboards, [], [], 0.05)
    for keyboard in readable:
        for event in keyboard.read():
            if (
                event.type == ecodes.EV_KEY
                and event.code == ecodes.KEY_ESC
                and event.value == 1
            ):
                return True
    return False


def speak(text, keyboards=(), speaking_path=None):
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
        if not keyboards:
            if speaking_path is not None:
                speaking_path.touch()
            subprocess.run(["aplay", "--quiet", str(speech_path)], check=True)
            return True
        if speaking_path is not None:
            speaking_path.touch()
        player = subprocess.Popen(["aplay", "--quiet", str(speech_path)])
        while player.poll() is None:
            if speech_cancelled(keyboards):
                player.terminate()
                try:
                    player.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    player.kill()
                    player.wait()
                print("Glossy: speech stopped.", flush=True)
                return False
        if player.returncode:
            raise subprocess.CalledProcessError(player.returncode, player.args)
        return True
    finally:
        if speaking_path is not None:
            speaking_path.unlink(missing_ok=True)
        speech_path.unlink(missing_ok=True)


# ponytail: local calibration assumes noise comes first; add a calibration phase if needed.
def has_speech(path, rms_threshold, minimum_seconds, aggressiveness, snr_ratio):
    with wave.open(str(path), "rb") as recorded_audio:
        if recorded_audio.getnchannels() != 1 or recorded_audio.getsampwidth() != 2:
            raise RuntimeError("Expected 16-bit mono recording")
        sample_rate = recorded_audio.getframerate()
        if sample_rate not in {8000, 16000, 32000, 48000}:
            raise RuntimeError("WebRTC VAD requires an 8, 16, 32, or 48 kHz recording")
        frames_per_window = sample_rate * 30 // 1000
        required_windows = max(1, math.ceil(minimum_seconds / 0.03))
        vad = webrtcvad.Vad(aggressiveness)
        windows = []
        while (
            len(data := recorded_audio.readframes(frames_per_window))
            == frames_per_window * 2
        ):
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
        best_of=1,
        temperature=0.0,
        without_timestamps=True,
    )
    return "".join(segment.text for segment in segments).strip()


def stream_transcript(transcriber, settings, audio_path, stopped, transcript_path=None):
    preview_path = Path(tempfile.gettempdir()) / f"glossy-preview-{os.getpid()}.wav"
    previous = ""
    line_width = 0
    try:
        # ponytail: re-transcribes the growing clip; use streaming ASR if this lags.
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
                if transcript_path is not None:
                    transcript_path.write_text(transcript + "\n")
                line = f"\rGlossy heard: {transcript}"
                print(line + " " * max(0, line_width - len(line)), end="", flush=True)
                line_width = len(line)
                previous = transcript
    finally:
        if previous:
            print(flush=True)
        preview_path.unlink(missing_ok=True)


def start_transcript_stream(transcriber, settings, audio_path, transcript_path=None):
    stopped = threading.Event()
    thread = threading.Thread(
        target=stream_transcript,
        args=(transcriber, settings, audio_path, stopped, transcript_path),
        daemon=True,
    )
    thread.start()
    return stopped, thread


def stop_transcript_stream(stream):
    stopped, thread = stream
    stopped.set()
    thread.join()
