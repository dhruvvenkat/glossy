#!/usr/bin/env python3
from evdev import InputDevice, ecodes, list_devices
from faster_whisper import WhisperModel
from openai import OpenAI
import webrtcvad

import errno
import json
import math
import os
import re
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from array import array
from datetime import datetime
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
os.environ["HF_HOME"] = str(MODEL_DIR / ".cache")

ENV_FILE = Path("~/.config/glossy.env").expanduser()
CONFIG_FILE = Path(__file__).parent / "config.json"
VOICE_DIR = Path(__file__).parent / "voices"
VISUALIZER_SCRIPT = Path(__file__).parent / "visualizer.py"
START_BLIP_SOUND = Path(__file__).parent / "blip.mp3"
STOP_BLIP_SOUND = Path(__file__).parent / "blip-reversed.mp3"
DEFAULT_VOICE = "en_US-lessac-medium"
RECONNECT_SECONDS = 5
TRANSCRIPT_PREVIEW_SECONDS = 0.25
SYSTEM_PROMPT = (Path(__file__).parent / "system-prompt.md").read_text().strip()
THREADS_DIR = Path("~/.config/glossy/threads").expanduser()
THREAD_RECENT_TURNS = 6
THREAD_SUMMARY_EVERY = 5
THREAD_SUMMARY_MAX_CHARS = 2500
THREAD_NAME_MAX_CHARS = 80
THREAD_PROMPT = """You are answering within an ongoing reading thread. Use the supplied
thread memory and recent conversation as background. Do not invent details that are not
present, and prioritize the user's current question if it conflicts with older context."""


class ThreadStore:
    def __init__(self, root=THREADS_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.state_path = self.root / "state.json"
        state = self._read(self.state_path, {"active_id": None, "enabled": False})
        self.active_id = state.get("active_id")
        self.enabled = bool(state.get("enabled"))

    def _read(self, path, default=None):
        if not path.exists() and default is not None:
            return default
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Cannot load {path}: {error}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"{path} must contain a JSON object")
        return value

    def _write(self, path, value):
        descriptor, temporary = tempfile.mkstemp(
            dir=self.root, prefix=f".{path.name}.", text=True
        )
        temporary = Path(temporary)
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump(value, output, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _save_state(self):
        self._write(
            self.state_path,
            {"active_id": self.active_id, "enabled": self.enabled},
        )

    def _path(self, thread_id):
        return self.root / f"thread-{thread_id}.json"

    def load(self, thread_id):
        thread = self._read(self._path(thread_id))
        required = {"id", "name", "summary", "summarized_turns", "turns"}
        if not required <= thread.keys() or not isinstance(thread["turns"], list):
            raise RuntimeError(f"Invalid thread file: {self._path(thread_id)}")
        return thread

    def list(self):
        threads = [
            self._read(path)
            for path in self.root.glob("thread-*.json")
        ]
        return sorted(threads, key=lambda thread: thread["name"].casefold())

    def current(self):
        return self.load(self.active_id) if self.enabled and self.active_id else None

    def selected(self):
        return self.load(self.active_id) if self.active_id else None

    def create(self, name):
        name = " ".join(name.split())
        if not name:
            name = datetime.now().strftime("Thread %Y-%m-%d %H-%M-%S")
        if len(name) > THREAD_NAME_MAX_CHARS:
            raise ValueError(f"Thread names must be at most {THREAD_NAME_MAX_CHARS} characters.")
        if any(thread["name"].casefold() == name.casefold() for thread in self.list()):
            raise ValueError(f"Thread {name} already exists.")
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        thread = {
            "id": uuid.uuid4().hex,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "summary": "",
            "summarized_turns": 0,
            "turns": [],
        }
        self._write(self._path(thread["id"]), thread)
        self.active_id = thread["id"]
        self.enabled = True
        self._save_state()
        return thread

    def activate(self, name):
        match = next(
            (
                thread
                for thread in self.list()
                if thread["name"].casefold() == name.strip().casefold()
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Thread {name.strip()} does not exist.")
        self.active_id = match["id"]
        self.enabled = True
        self._save_state()
        return match

    def set_enabled(self, enabled):
        if enabled and not self.active_id:
            raise ValueError("No thread is selected. Say new thread followed by a name.")
        self.enabled = enabled
        self._save_state()

    def append_turn(self, question, answer):
        thread = self.current()
        if thread is None:
            return None
        thread["turns"].append(
            {
                "question": question,
                "answer": answer,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )
        thread["updated_at"] = thread["turns"][-1]["created_at"]
        self._write(self._path(thread["id"]), thread)
        return thread

    def save_summary(self, thread, summary):
        thread["summary"] = summary[:THREAD_SUMMARY_MAX_CHARS]
        thread["summarized_turns"] = len(thread["turns"])
        self._write(self._path(thread["id"]), thread)


def handle_thread_command(transcript, store):
    command = transcript.strip().rstrip(".!?").strip()
    new_thread = re.fullmatch(r"new thread(?:\s+(.+))?", command, re.IGNORECASE)
    switch_thread = re.fullmatch(
        r"switch to thread\s+(.+)", command, re.IGNORECASE
    )
    try:
        if new_thread:
            thread = store.create(new_thread.group(1) or "")
            return f"Started thread {thread['name']}."
        if switch_thread:
            thread = store.activate(switch_thread.group(1))
            return f"Switched to thread {thread['name']}."
        if command.casefold() == "list threads":
            threads = store.list()
            return (
                "Your threads are " + ", ".join(thread["name"] for thread in threads) + "."
                if threads
                else "You do not have any threads yet."
            )
        if command.casefold() == "current thread":
            thread = store.selected()
            if thread is None:
                return "No thread is selected."
            status = "on" if store.enabled else "off"
            return f"The current thread is {thread['name']}. Threads mode is {status}."
        if command.casefold() == "threads mode":
            store.set_enabled(True)
            return f"Threads mode is on for {store.selected()['name']}."
        if command.casefold() == "exit threads mode":
            store.set_enabled(False)
            return "Threads mode is off."
    except ValueError as error:
        return str(error)
    return None


def thread_input(thread, question):
    parts = [f"Reading thread: {thread['name']}"]
    if thread["summary"]:
        parts.append(f"Thread memory:\n{thread['summary']}")
    if thread["turns"]:
        recent = []
        for turn in thread["turns"][-THREAD_RECENT_TURNS:]:
            recent.extend(
                [f"Reader: {turn['question']}", f"Assistant: {turn['answer']}"]
            )
        parts.append("Recent conversation:\n" + "\n".join(recent))
    parts.append(f"Current question:\n{question}")
    return "\n\n".join(parts)


def update_thread_summary(client, settings, store, thread):
    start = thread["summarized_turns"]
    if len(thread["turns"]) - start < THREAD_SUMMARY_EVERY:
        return
    pending = "\n".join(
        f"Reader: {turn['question']}\nAssistant: {turn['answer']}"
        for turn in thread["turns"][start:]
    )
    request = {
        "model": settings["model"],
        "instructions": (
            "Maintain a concise, factual memory for an ongoing reading thread. "
            "Preserve important concepts, the reader's demonstrated understanding, "
            "and unresolved questions. Do not guess. Return plain text under 2000 characters."
        ),
        "input": f"Existing memory:\n{thread['summary'] or 'None'}\n\nNew conversation:\n{pending}",
    }
    if settings["reasoning_effort"] is not None:
        request["reasoning"] = {"effort": settings["reasoning_effort"]}
    summary = client.responses.create(**request).output_text.strip()
    if not summary:
        raise RuntimeError("OpenAI returned an empty thread summary")
    store.save_summary(thread, summary)


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


def start_visualizer(audio_path, sensitivity, question_path=None):
    command = [sys.executable, str(VISUALIZER_SCRIPT), str(audio_path), str(sensitivity)]
    if question_path is not None:
        command.append(str(question_path))
    return subprocess.Popen(command)


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


def answer_question(
    client,
    transcriber,
    settings,
    audio_path,
    keyboards=(),
    transcript_path=None,
    speaking_path=None,
    thread_store=None,
):
    live_transcript = ""
    if transcript_path is not None:
        try:
            live_transcript = transcript_path.read_text().strip()
        except OSError:
            pass
    if not live_transcript and not has_speech(
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
    print(f"Glossy question: {transcript!r}", flush=True)
    audio_path.with_suffix(".txt").write_text(transcript + "\n")

    if thread_store is not None:
        command_answer = handle_thread_command(transcript, thread_store)
        if command_answer is not None:
            if speaking_path is None:
                speak(command_answer, keyboards)
            else:
                speak(command_answer, keyboards, speaking_path)
            return True

    thread = thread_store.current() if thread_store is not None else None

    request = dict(
        model=settings["model"],
        instructions=(
            f"{SYSTEM_PROMPT}\n\n{THREAD_PROMPT}" if thread else SYSTEM_PROMPT
        ),
        input=thread_input(thread, transcript) if thread else transcript,
    )
    if settings["reasoning_effort"] is not None:
        request["reasoning"] = {"effort": settings["reasoning_effort"]}
    answer = client.responses.create(**request).output_text.strip()
    if not answer:
        raise RuntimeError("OpenAI returned an empty answer")
    if thread is not None:
        thread = thread_store.append_turn(transcript, answer)
        try:
            update_thread_summary(client, settings, thread_store, thread)
        except Exception as error:
            print(
                f"Glossy: thread summary update failed: {error}",
                file=sys.stderr,
                flush=True,
            )
    if speaking_path is None:
        speak(answer, keyboards)
    else:
        speak(answer, keyboards, speaking_path)
    return True


def report_error(error, keyboards=()):
    print(f"Glossy: {error}", file=sys.stderr, flush=True)
    try:
        speak("Glossy failed. Check the service log.", keyboards)
    except Exception:
        pass


def listen_connected(
    client,
    transcriber,
    settings,
    keyboards,
    button_code,
    button_name,
    thread_store=None,
):
    audio_path = Path(tempfile.gettempdir()) / f"glossy-{os.getpid()}.wav"
    question_path = audio_path.with_suffix(".txt")
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
                    question_path.unlink(missing_ok=True)
                    audio_path.with_suffix(".speaking").unlink(missing_ok=True)
                    recorder = start_recording(audio_path)
                    transcript_stream = start_transcript_stream(
                        transcriber, settings, audio_path, question_path
                    )
                    visualizer = start_visualizer(
                        audio_path, settings["visualizer_sensitivity"], question_path
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
                                play_blip(STOP_BLIP_SOUND)
                                print("Answering...", flush=True)
                                answer_question(
                                    client,
                                    transcriber,
                                    settings,
                                    audio_path,
                                    keyboards,
                                    question_path,
                                    audio_path.with_suffix(".speaking"),
                                    thread_store,
                                )
                                stop_visualizer(visualizer)
                                visualizer = None
                                print("Ready.", flush=True)
                        except Exception as error:
                            report_error(error, keyboards)
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
                            question_path.unlink(missing_ok=True)
                            audio_path.with_suffix(".speaking").unlink(missing_ok=True)
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
        question_path.unlink(missing_ok=True)
        audio_path.with_suffix(".speaking").unlink(missing_ok=True)
        for keyboard in keyboards:
            keyboard.close()


def listen(client, transcriber, settings, thread_store=None):
    button_name = settings["button"]
    button_code = getattr(ecodes, button_name)
    while True:
        keyboards = wait_for_keyboards(button_code, button_name)
        try:
            listen_connected(
                client,
                transcriber,
                settings,
                keyboards,
                button_code,
                button_name,
                thread_store,
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
        listen(OpenAI(), transcriber, settings, ThreadStore())
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"Glossy: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
