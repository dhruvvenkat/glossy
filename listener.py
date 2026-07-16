from evdev import InputDevice, ecodes, list_devices

import errno
import os
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from audio import (
    START_BLIP_SOUND,
    STOP_BLIP_SOUND,
    has_speech,
    play_blip,
    speak,
    start_recording,
    start_transcript_stream,
    stop_recording,
    stop_transcript_stream,
    transcribe_audio,
)
from openai_api import answer as ask_openai
from openai_api import summarize as summarize_thread
from threads import handle_thread_command

VISUALIZER_SCRIPT = Path(__file__).parent / "visualizer.py"
RECONNECT_SECONDS = 5


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
    answer = ask_openai(client, settings, transcript, thread)
    if thread is not None:
        thread = thread_store.append_turn(transcript, answer)
        try:
            summary = summarize_thread(client, settings, thread)
            if summary is not None:
                thread_store.save_summary(thread, summary)
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
