import errno
import json
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import audio as audio_io
import listener
import openai_api
import settings as glossy_settings
import threads

TEST_SETTINGS = {
    "model": "test-model",
    "reasoning_effort": "none",
    "transcription_model": "small.en",
    "transcription_beam_size": 1,
    "hold_seconds": 0.35,
    "button": "KEY_RIGHTALT",
    "visualizer_sensitivity": 4.0,
    "speech_rms_threshold": 300,
    "minimum_speech_seconds": 0.3,
    "vad_aggressiveness": 3,
    "speech_snr_ratio": 2.0,
}


class ConfigTest(unittest.TestCase):
    def test_loads_runtime_settings(self):
        settings = {
            "model": "gpt-5.5",
            "reasoning_effort": "none",
            "transcription_model": "small.en",
            "transcription_beam_size": 1,
            "hold_seconds": 0.5,
            "button": "KEY_HOME",
            "visualizer_sensitivity": 4.0,
            "speech_rms_threshold": 300,
            "minimum_speech_seconds": 0.3,
            "vad_aggressiveness": 3,
            "speech_snr_ratio": 2.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(settings))
            self.assertEqual(glossy_settings.load_settings(path), settings)

    def test_rejects_unknown_button(self):
        settings = {
            "model": "gpt-5.5",
            "reasoning_effort": "none",
            "transcription_model": "small.en",
            "transcription_beam_size": 1,
            "hold_seconds": 0.5,
            "button": "KEY_NOT_REAL",
            "visualizer_sensitivity": 4.0,
            "speech_rms_threshold": 300,
            "minimum_speech_seconds": 0.3,
            "vad_aggressiveness": 3,
            "speech_snr_ratio": 2.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(settings))
            with self.assertRaisesRegex(RuntimeError, "evdev key name"):
                glossy_settings.load_settings(path)


class AnswerQuestionTest(unittest.TestCase):
    @patch("listener.has_speech", return_value=True)
    @patch("listener.speak")
    @patch("builtins.print")
    def test_transcribes_answers_and_speaks(self, output, speak, _has_speech):
        client = Mock()
        transcriber = Mock()
        transcriber.transcribe.return_value = (
            [SimpleNamespace(text="What is a mutex?")],
            SimpleNamespace(),
        )
        client.responses.create.return_value = SimpleNamespace(
            output_text="A mutex permits one thread at a time."
        )

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "question.wav"
            audio.write_bytes(b"RIFF fake audio")
            listener.answer_question(client, transcriber, TEST_SETTINGS, audio)
            self.assertEqual(audio.with_suffix(".txt").read_text(), "What is a mutex?\n")

        transcriber.transcribe.assert_called_once_with(
            str(audio),
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            without_timestamps=True,
        )
        client.responses.create.assert_called_once_with(
            model="test-model",
            instructions=openai_api.SYSTEM_PROMPT,
            input="What is a mutex?",
            reasoning={"effort": "none"},
        )
        speak.assert_called_once_with("A mutex permits one thread at a time.", ())
        output.assert_called_once_with("Glossy question: 'What is a mutex?'", flush=True)

    @patch("builtins.print")
    def test_streams_live_transcript(self, output):
        transcriber = Mock()
        transcriber.transcribe.return_value = (
            [SimpleNamespace(text="What is a mutex?")],
            SimpleNamespace(),
        )
        stopped = Mock()
        stopped.wait.side_effect = [False, True]

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "question.wav"
            transcript = Path(directory) / "question.txt"
            with wave.open(str(audio), "wb") as recording:
                recording.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                recording.writeframes(array("h", [1000, -1000] * 1600).tobytes())
            audio_io.stream_transcript(
                transcriber, TEST_SETTINGS, audio, stopped, transcript
            )
            self.assertEqual(transcript.read_text(), "What is a mutex?\n")

        output.assert_has_calls(
            [
                call("\rGlossy heard: What is a mutex?", end="", flush=True),
                call(flush=True),
            ]
        )

    @patch("builtins.print")
    def test_silence_never_reaches_openai(self, _print):
        client = Mock()
        transcriber = Mock()
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "silence.wav"
            with wave.open(str(audio), "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                output.writeframes(array("h", [0] * 16000).tobytes())
            self.assertFalse(
                listener.answer_question(client, transcriber, TEST_SETTINGS, audio)
            )

        transcriber.transcribe.assert_not_called()
        client.responses.create.assert_not_called()

    @patch("listener.has_speech", return_value=False)
    @patch("listener.speak")
    def test_live_transcript_bypasses_false_vad_rejection(self, speak, has_speech):
        client = Mock()
        transcriber = Mock()
        transcriber.transcribe.return_value = (
            [SimpleNamespace(text="What is a mutex?")],
            SimpleNamespace(),
        )
        client.responses.create.return_value = SimpleNamespace(
            output_text="A mutex permits one thread at a time."
        )

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "question.wav"
            transcript = Path(directory) / "question.txt"
            audio.write_bytes(b"RIFF fake audio")
            transcript.write_text("What is a mutex?\n")
            self.assertTrue(
                listener.answer_question(
                    client,
                    transcriber,
                    TEST_SETTINGS,
                    audio,
                    transcript_path=transcript,
                )
            )

        has_speech.assert_not_called()
        client.responses.create.assert_called_once()
        speak.assert_called_once_with("A mutex permits one thread at a time.", ())

    @patch("audio.webrtcvad.Vad")
    @patch("builtins.print")
    def test_detects_sustained_local_audio(self, _print, vad):
        vad.return_value.is_speech.return_value = True
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "voice.wav"
            with wave.open(str(audio), "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                samples = [0] * 2400 + [1000, -1000] * 2400
                output.writeframes(array("h", samples).tobytes())
            self.assertTrue(audio_io.has_speech(audio, 300, 0.3, 3, 2.0))
        vad.assert_called_once_with(3)

    @patch("audio.webrtcvad.Vad")
    @patch("builtins.print")
    def test_rejects_continuous_noise(self, _print, vad):
        vad.return_value.is_speech.return_value = True
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "noise.wav"
            with wave.open(str(audio), "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                output.writeframes(array("h", [1000, -1000] * 8000).tobytes())
            self.assertFalse(audio_io.has_speech(audio, 300, 0.3, 3, 2.0))

    @patch("audio.subprocess.run")
    def test_speak_uses_piper_then_aplay(self, run):
        with tempfile.TemporaryDirectory() as directory:
            voice_dir = Path(directory)
            voice = voice_dir / "voice.onnx"
            voice.touch()
            (voice_dir / "selected").write_text("voice\n")
            with patch.object(audio_io, "VOICE_DIR", voice_dir):
                audio_io.speak("A clear answer.")

        piper, player = run.call_args_list
        self.assertEqual(piper.args[0][0:3], [audio_io.sys.executable, "-m", "piper"])
        self.assertEqual(piper.kwargs["input"], "A clear answer.\n")
        self.assertEqual(player.args[0][0:2], ["aplay", "--quiet"])

    @patch("builtins.print")
    @patch("audio.select.select")
    @patch("audio.subprocess.Popen")
    @patch("audio.subprocess.run")
    def test_escape_stops_speech(self, run, popen, select_, output):
        keyboard = Mock()
        keyboard.read.return_value = [
            SimpleNamespace(
                type=audio_io.ecodes.EV_KEY,
                code=audio_io.ecodes.KEY_ESC,
                value=1,
            )
        ]
        select_.return_value = ([keyboard], [], [])
        player = Mock()
        player.poll.return_value = None
        popen.return_value = player

        with tempfile.TemporaryDirectory() as directory:
            voice_dir = Path(directory)
            voice = voice_dir / "voice.onnx"
            voice.touch()
            (voice_dir / "selected").write_text("voice\n")
            with patch.object(audio_io, "VOICE_DIR", voice_dir):
                self.assertFalse(audio_io.speak("Stop talking.", [keyboard]))

        run.assert_called_once()
        popen.assert_called_once()
        player.terminate.assert_called_once()
        player.wait.assert_called_once_with(timeout=1)
        output.assert_called_once_with("Glossy: speech stopped.", flush=True)

    @patch("builtins.print")
    @patch("audio.select.select")
    @patch("audio.subprocess.Popen")
    @patch("audio.subprocess.run")
    def test_record_button_stops_speech_and_requests_recording(
        self, _run, popen, select_, _output
    ):
        keyboard = Mock()
        keyboard.read.return_value = [
            SimpleNamespace(
                type=audio_io.ecodes.EV_KEY,
                code=audio_io.ecodes.KEY_RIGHTALT,
                value=1,
            )
        ]
        select_.return_value = ([keyboard], [], [])
        player = Mock()
        player.poll.return_value = None
        popen.return_value = player

        with tempfile.TemporaryDirectory() as directory:
            voice_dir = Path(directory)
            (voice_dir / "voice.onnx").touch()
            (voice_dir / "selected").write_text("voice\n")
            with patch.object(audio_io, "VOICE_DIR", voice_dir):
                result = audio_io.speak(
                    "Stop talking.",
                    [keyboard],
                    recording_button=audio_io.ecodes.KEY_RIGHTALT,
                )

        self.assertEqual(result, audio_io.RECORDING_REQUESTED)
        player.terminate.assert_called_once()

    @patch("audio.subprocess.run")
    def test_blips_use_requested_sounds(self, run):
        audio_io.play_blip(audio_io.START_BLIP_SOUND)
        audio_io.play_blip(audio_io.STOP_BLIP_SOUND)
        self.assertEqual(
            run.call_args_list,
            [
                call(["paplay", str(audio_io.START_BLIP_SOUND)], check=False),
                call(["paplay", str(audio_io.STOP_BLIP_SOUND)], check=False),
            ],
        )

    @patch("listener.subprocess.Popen")
    def test_visualizer_uses_recording_file(self, popen):
        audio = Path("question.wav")
        question = Path("question.txt")
        listener.start_visualizer(audio, 4.0, question)
        popen.assert_called_once_with(
            [
                listener.sys.executable,
                str(listener.VISUALIZER_SCRIPT),
                str(audio),
                "4.0",
                str(question),
            ]
        )


class ThreadModeTest(unittest.TestCase):
    def test_persists_and_switches_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            store = threads.ThreadStore(directory)
            first = store.create("Operating Systems")
            store.create("Compilers")
            selected = store.activate("operating systems")

            reloaded = threads.ThreadStore(directory)
            self.assertEqual(selected["id"], first["id"])
            self.assertEqual(reloaded.current()["name"], "Operating Systems")
            self.assertEqual(
                [thread["name"] for thread in reloaded.list()],
                ["Compilers", "Operating Systems"],
            )
            with self.assertRaisesRegex(ValueError, "already exists"):
                reloaded.create("OPERATING SYSTEMS")

    def test_voice_commands_control_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            store = threads.ThreadStore(directory)
            self.assertEqual(
                threads.handle_thread_command("threads mode.", store),
                "No thread is selected. Say new thread followed by a name.",
            )
            self.assertEqual(
                threads.handle_thread_command("New thread Operating Systems.", store),
                "Started thread Operating Systems.",
            )
            self.assertEqual(
                threads.handle_thread_command("exit threads mode", store),
                "Threads mode is off.",
            )
            self.assertIsNone(store.current())
            self.assertEqual(
                threads.handle_thread_command("threads mode", store),
                "Threads mode is on for Operating Systems.",
            )
            store.create("Compilers")
            self.assertEqual(
                threads.handle_thread_command(
                    "switch to thread Operating Systems", store
                ),
                "Switched to thread Operating Systems.",
            )
            self.assertIs(
                threads.handle_thread_command("list threads", store),
                threads.THREAD_PICKER_REQUESTED,
            )

    @patch("listener.select.select")
    def test_picker_uses_arrows_and_enter(self, select_):
        keyboard = Mock()
        keyboard.read.side_effect = [
            [
                SimpleNamespace(
                    type=listener.ecodes.EV_KEY,
                    code=listener.ecodes.KEY_UP,
                    value=1,
                )
            ],
            [
                SimpleNamespace(
                    type=listener.ecodes.EV_KEY,
                    code=listener.ecodes.KEY_ENTER,
                    value=1,
                )
            ],
        ]
        select_.side_effect = [([keyboard], [], []), ([keyboard], [], [])]

        with tempfile.TemporaryDirectory() as directory:
            store = threads.ThreadStore(Path(directory) / "threads")
            store.create("Compilers")
            store.create("Operating Systems")
            transcript = Path(directory) / "question.txt"
            selected = listener.pick_thread(store, [keyboard], transcript)

            self.assertEqual(selected["name"], "Compilers")
            self.assertEqual(store.current()["name"], "Compilers")
            self.assertIn("› Compilers", transcript.read_text())

    def test_picker_text_scrolls_to_selection(self):
        items = [{"name": f"Thread {index}"} for index in range(8)]
        rendered = threads.thread_picker_text(items, 7)
        self.assertIn("↑ more", rendered)
        self.assertIn("› Thread 7", rendered)
        self.assertNotIn("Thread 0", rendered)

    @patch("listener.pick_thread")
    @patch("listener.has_speech", return_value=True)
    @patch("listener.speak")
    def test_list_threads_opens_picker_without_speech(
        self, speak, _has_speech, pick_thread
    ):
        client = Mock()
        transcriber = Mock()
        transcriber.transcribe.return_value = (
            [SimpleNamespace(text="List threads.")],
            SimpleNamespace(),
        )
        keyboard = Mock()

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "question.wav"
            transcript = audio.with_suffix(".txt")
            audio.write_bytes(b"RIFF fake audio")
            store = threads.ThreadStore(Path(directory) / "threads")
            store.create("Operating Systems")
            self.assertTrue(
                listener.answer_question(
                    client,
                    transcriber,
                    TEST_SETTINGS,
                    audio,
                    keyboards=[keyboard],
                    transcript_path=transcript,
                    thread_store=store,
                )
            )

        pick_thread.assert_called_once_with(store, [keyboard], transcript)
        speak.assert_not_called()
        client.responses.create.assert_not_called()

    @patch("listener.has_speech", return_value=True)
    @patch("listener.speak")
    def test_thread_command_bypasses_openai(self, speak, _has_speech):
        client = Mock()
        transcriber = Mock()
        transcriber.transcribe.return_value = (
            [SimpleNamespace(text="New thread Operating Systems.")],
            SimpleNamespace(),
        )

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "question.wav"
            audio.write_bytes(b"RIFF fake audio")
            store = threads.ThreadStore(Path(directory) / "threads")
            self.assertTrue(
                listener.answer_question(
                    client,
                    transcriber,
                    TEST_SETTINGS,
                    audio,
                    thread_store=store,
                )
            )
            self.assertEqual(store.current()["name"], "Operating Systems")

        client.responses.create.assert_not_called()
        speak.assert_called_once_with("Started thread Operating Systems.", ())

    @patch("listener.has_speech", return_value=True)
    @patch("listener.speak")
    def test_thread_context_is_sent_and_saved(self, speak, _has_speech):
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(
            output_text="It prevents races around shared state."
        )
        transcriber = Mock()
        transcriber.transcribe.return_value = (
            [SimpleNamespace(text="Why do we need a mutex?")],
            SimpleNamespace(),
        )

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "question.wav"
            audio.write_bytes(b"RIFF fake audio")
            store = threads.ThreadStore(Path(directory) / "threads")
            store.create("Operating Systems")
            store.append_turn("What is a mutex?", "A mutual-exclusion lock.")
            listener.answer_question(
                client,
                transcriber,
                TEST_SETTINGS,
                audio,
                thread_store=store,
            )

            request = client.responses.create.call_args.kwargs
            self.assertIn(openai_api.THREAD_PROMPT, request["instructions"])
            self.assertIn("Reader: What is a mutex?", request["input"])
            self.assertIn("Current question:\nWhy do we need a mutex?", request["input"])
            self.assertEqual(len(store.current()["turns"]), 2)

        speak.assert_called_once_with("It prevents races around shared state.", ())

    @patch("listener.has_speech", return_value=True)
    @patch("listener.speak", return_value=False)
    def test_escape_during_speech_discards_thread_turn(self, _speak, _has_speech):
        client = Mock()
        client.responses.create.return_value = SimpleNamespace(
            output_text="A discarded answer."
        )
        transcriber = Mock()
        transcriber.transcribe.return_value = (
            [SimpleNamespace(text="A discarded question?")],
            SimpleNamespace(),
        )

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "question.wav"
            audio.write_bytes(b"RIFF fake audio")
            store = threads.ThreadStore(Path(directory) / "threads")
            store.create("Operating Systems")
            self.assertFalse(
                listener.answer_question(
                    client,
                    transcriber,
                    TEST_SETTINGS,
                    audio,
                    thread_store=store,
                )
            )
            self.assertEqual(store.current()["turns"], [])

    @patch("listener.has_speech", return_value=True)
    @patch("listener.speak")
    def test_summarizes_every_five_new_turns(self, _speak, _has_speech):
        client = Mock()
        client.responses.create.side_effect = [
            SimpleNamespace(output_text="The fifth answer."),
            SimpleNamespace(output_text="The reader is studying concurrency."),
        ]
        transcriber = Mock()
        transcriber.transcribe.return_value = (
            [SimpleNamespace(text="Question five?")],
            SimpleNamespace(),
        )

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "question.wav"
            audio.write_bytes(b"RIFF fake audio")
            store = threads.ThreadStore(Path(directory) / "threads")
            store.create("Operating Systems")
            for number in range(4):
                store.append_turn(f"Question {number}?", f"Answer {number}.")
            listener.answer_question(
                client,
                transcriber,
                TEST_SETTINGS,
                audio,
                thread_store=store,
            )

            thread = store.current()
            self.assertEqual(thread["summary"], "The reader is studying concurrency.")
            self.assertEqual(thread["summarized_turns"], 5)

        self.assertEqual(client.responses.create.call_count, 2)

    @patch("builtins.print")
    @patch("listener.has_speech", return_value=True)
    @patch("listener.speak")
    def test_summary_failure_does_not_suppress_answer(
        self, speak, _has_speech, _print
    ):
        client = Mock()
        client.responses.create.side_effect = [
            SimpleNamespace(output_text="The fifth answer."),
            RuntimeError("offline"),
        ]
        transcriber = Mock()
        transcriber.transcribe.return_value = (
            [SimpleNamespace(text="Question five?")],
            SimpleNamespace(),
        )

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "question.wav"
            audio.write_bytes(b"RIFF fake audio")
            store = threads.ThreadStore(Path(directory) / "threads")
            store.create("Operating Systems")
            for number in range(4):
                store.append_turn(f"Question {number}?", f"Answer {number}.")
            listener.answer_question(
                client,
                transcriber,
                TEST_SETTINGS,
                audio,
                thread_store=store,
            )
            self.assertEqual(len(store.current()["turns"]), 5)
            self.assertEqual(store.current()["summary"], "")

        speak.assert_called_once_with("The fifth answer.", ())


class InputDelayTest(unittest.TestCase):
    @patch("builtins.print")
    @patch("listener.time.sleep")
    @patch("listener.find_keyboards")
    def test_waits_for_replacement_keyboard(self, find_keyboards, sleep, _print):
        keyboard = Mock()
        find_keyboards.side_effect = [[], [], [keyboard]]
        self.assertEqual(listener.wait_for_keyboards(100, "KEY_RIGHTALT"), [keyboard])
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(listener.RECONNECT_SECONDS)

    def test_detects_disconnected_keyboard(self):
        keyboard = Mock()
        keyboard.active_keys.side_effect = OSError(errno.ENODEV, "gone")
        self.assertFalse(listener.keyboards_connected([keyboard]))

    @patch("builtins.print")
    @patch("listener.listen_connected")
    @patch("listener.wait_for_keyboards")
    def test_reconnects_in_process(self, wait_for_keyboards, listen_connected, _print):
        old_keyboard = Mock()
        new_keyboard = Mock()
        wait_for_keyboards.side_effect = [[old_keyboard], [new_keyboard]]
        listen_connected.side_effect = [
            OSError(errno.ENODEV, "gone"),
            KeyboardInterrupt,
        ]

        with self.assertRaises(KeyboardInterrupt):
            listener.listen(Mock(), Mock(), TEST_SETTINGS)

        self.assertEqual(wait_for_keyboards.call_count, 2)

    @patch("builtins.print")
    @patch("listener.start_visualizer")
    @patch("listener.start_recording")
    @patch("listener.play_blip")
    def test_short_press_never_starts_recording(
        self, play_blip, start_recording, start_visualizer, _print
    ):
        keyboard = Mock()
        keyboard.read.side_effect = [
            [
                SimpleNamespace(
                    type=listener.ecodes.EV_KEY,
                    code=listener.ecodes.KEY_RIGHTALT,
                    value=1,
                )
            ],
            [
                SimpleNamespace(
                    type=listener.ecodes.EV_KEY,
                    code=listener.ecodes.KEY_RIGHTALT,
                    value=0,
                )
            ],
        ]
        selections = [([keyboard], [], []), ([keyboard], [], []), KeyboardInterrupt]

        with (
            patch("listener.find_keyboards", return_value=[keyboard]),
            patch("listener.keyboards_connected", return_value=True),
            patch("listener.select.select", side_effect=selections),
            patch("listener.time.monotonic", side_effect=[0.0, 0.1]),
            self.assertRaises(KeyboardInterrupt),
        ):
            listener.listen(Mock(), Mock(), TEST_SETTINGS)

        start_recording.assert_not_called()
        start_visualizer.assert_not_called()
        play_blip.assert_not_called()

    @patch("builtins.print")
    @patch("listener.answer_question")
    @patch("listener.stop_transcript_stream")
    @patch("listener.start_transcript_stream")
    @patch("listener.stop_recording")
    @patch("listener.stop_visualizer")
    @patch("listener.start_visualizer")
    @patch("listener.start_recording")
    @patch("listener.play_blip")
    def test_recording_starts_after_threshold(
        self,
        play_blip,
        start_recording,
        start_visualizer,
        stop_visualizer,
        stop_recording,
        start_transcript_stream,
        stop_transcript_stream,
        answer_question,
        _print,
    ):
        order = []
        visualizer = Mock()
        transcript_stream = Mock()
        play_blip.side_effect = lambda sound: order.append(sound.name)
        start_recording.side_effect = lambda _path: order.append("record") or Mock()
        start_transcript_stream.side_effect = (
            lambda *_args: order.append("transcript-start") or transcript_stream
        )
        start_visualizer.side_effect = (
            lambda _path, _sensitivity, _question_path: order.append("visualizer-start")
            or visualizer
        )
        stop_visualizer.side_effect = lambda _process: order.append("visualizer-stop")
        stop_recording.side_effect = lambda *_args: order.append("stop")
        stop_transcript_stream.side_effect = lambda *_args: order.append(
            "transcript-stop"
        )
        answer_question.side_effect = lambda *_args: order.append("answer")
        keyboard = Mock()
        keyboard.read.side_effect = [
            [
                SimpleNamespace(
                    type=listener.ecodes.EV_KEY,
                    code=listener.ecodes.KEY_RIGHTALT,
                    value=1,
                )
            ],
            [
                SimpleNamespace(
                    type=listener.ecodes.EV_KEY,
                    code=listener.ecodes.KEY_RIGHTALT,
                    value=0,
                )
            ],
        ]
        selections = [([keyboard], [], []), ([keyboard], [], []), KeyboardInterrupt]

        with (
            patch("listener.find_keyboards", return_value=[keyboard]),
            patch("listener.keyboards_connected", return_value=True),
            patch("listener.select.select", side_effect=selections),
            patch(
                "listener.time.monotonic",
                side_effect=[0.0, TEST_SETTINGS["hold_seconds"]],
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            listener.listen(Mock(), Mock(), TEST_SETTINGS)

        start_recording.assert_called_once()
        self.assertEqual(
            order,
            [
                "blip.mp3",
                "record",
                "transcript-start",
                "visualizer-start",
                "stop",
                "transcript-stop",
                "blip-reversed.mp3",
                "answer",
                "visualizer-stop",
            ],
        )
        stop_visualizer.assert_called_once_with(visualizer)
        stop_recording.assert_called_once()
        stop_transcript_stream.assert_called_once_with(transcript_stream)
        answer_question.assert_called_once()

    @patch("builtins.print")
    @patch("listener.answer_question", return_value=audio_io.RECORDING_REQUESTED)
    @patch("listener.stop_transcript_stream")
    @patch("listener.start_transcript_stream")
    @patch("listener.stop_recording")
    @patch("listener.stop_visualizer")
    @patch("listener.start_visualizer")
    @patch("listener.start_recording")
    @patch("listener.play_blip")
    def test_record_button_during_speech_starts_another_hold(
        self,
        _play_blip,
        start_recording,
        start_visualizer,
        _stop_visualizer,
        _stop_recording,
        start_transcript_stream,
        _stop_transcript_stream,
        answer_question,
        _print,
    ):
        recorders = [Mock(), Mock()]
        for recorder in recorders:
            recorder.poll.return_value = 0
        start_recording.side_effect = recorders
        start_visualizer.side_effect = [Mock(), Mock()]
        start_transcript_stream.side_effect = [Mock(), Mock()]
        keyboard = Mock()
        keyboard.read.side_effect = [
            [
                SimpleNamespace(
                    type=listener.ecodes.EV_KEY,
                    code=listener.ecodes.KEY_RIGHTALT,
                    value=1,
                )
            ],
            [
                SimpleNamespace(
                    type=listener.ecodes.EV_KEY,
                    code=listener.ecodes.KEY_RIGHTALT,
                    value=0,
                )
            ],
        ]
        selections = [([keyboard], [], []), ([keyboard], [], []), KeyboardInterrupt]
        threshold = TEST_SETTINGS["hold_seconds"]

        with (
            patch("listener.find_keyboards", return_value=[keyboard]),
            patch("listener.keyboards_connected", return_value=True),
            patch("listener.select.select", side_effect=selections),
            patch(
                "listener.time.monotonic",
                side_effect=[0.0, threshold, 1.0, 1.0 + threshold],
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            listener.listen(Mock(), Mock(), TEST_SETTINGS)

        self.assertEqual(start_recording.call_count, 2)
        answer_question.assert_called_once()


if __name__ == "__main__":
    unittest.main()
