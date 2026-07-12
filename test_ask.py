import errno
import json
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import ask

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
            self.assertEqual(ask.load_settings(path), settings)

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
                ask.load_settings(path)


class AnswerQuestionTest(unittest.TestCase):
    @patch("ask.has_speech", return_value=True)
    @patch("ask.speak")
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
            ask.answer_question(client, transcriber, TEST_SETTINGS, audio)
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
            instructions=ask.SYSTEM_PROMPT,
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
            ask.stream_transcript(transcriber, TEST_SETTINGS, audio, stopped, transcript)
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
                ask.answer_question(client, transcriber, TEST_SETTINGS, audio)
            )

        transcriber.transcribe.assert_not_called()
        client.responses.create.assert_not_called()

    @patch("ask.has_speech", return_value=False)
    @patch("ask.speak")
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
                ask.answer_question(
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

    @patch("ask.webrtcvad.Vad")
    @patch("builtins.print")
    def test_detects_sustained_local_audio(self, _print, vad):
        vad.return_value.is_speech.return_value = True
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "voice.wav"
            with wave.open(str(audio), "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                samples = [0] * 2400 + [1000, -1000] * 2400
                output.writeframes(array("h", samples).tobytes())
            self.assertTrue(ask.has_speech(audio, 300, 0.3, 3, 2.0))
        vad.assert_called_once_with(3)

    @patch("ask.webrtcvad.Vad")
    @patch("builtins.print")
    def test_rejects_continuous_noise(self, _print, vad):
        vad.return_value.is_speech.return_value = True
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "noise.wav"
            with wave.open(str(audio), "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                output.writeframes(array("h", [1000, -1000] * 8000).tobytes())
            self.assertFalse(ask.has_speech(audio, 300, 0.3, 3, 2.0))

    @patch("ask.subprocess.run")
    def test_speak_uses_piper_then_aplay(self, run):
        with tempfile.TemporaryDirectory() as directory:
            voice_dir = Path(directory)
            voice = voice_dir / "voice.onnx"
            voice.touch()
            (voice_dir / "selected").write_text("voice\n")
            with patch.object(ask, "VOICE_DIR", voice_dir):
                ask.speak("A clear answer.")

        piper, player = run.call_args_list
        self.assertEqual(piper.args[0][0:3], [ask.sys.executable, "-m", "piper"])
        self.assertEqual(piper.kwargs["input"], "A clear answer.\n")
        self.assertEqual(player.args[0][0:2], ["aplay", "--quiet"])

    @patch("builtins.print")
    @patch("ask.select.select")
    @patch("ask.subprocess.Popen")
    @patch("ask.subprocess.run")
    def test_escape_stops_speech(self, run, popen, select_, output):
        keyboard = Mock()
        keyboard.read.return_value = [
            SimpleNamespace(
                type=ask.ecodes.EV_KEY,
                code=ask.ecodes.KEY_ESC,
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
            with patch.object(ask, "VOICE_DIR", voice_dir):
                self.assertFalse(ask.speak("Stop talking.", [keyboard]))

        run.assert_called_once()
        popen.assert_called_once()
        player.terminate.assert_called_once()
        player.wait.assert_called_once_with(timeout=1)
        output.assert_called_once_with("Glossy: speech stopped.", flush=True)

    @patch("ask.subprocess.run")
    def test_blips_use_requested_sounds(self, run):
        ask.play_blip(ask.START_BLIP_SOUND)
        ask.play_blip(ask.STOP_BLIP_SOUND)
        self.assertEqual(
            run.call_args_list,
            [
                call(["paplay", str(ask.START_BLIP_SOUND)], check=False),
                call(["paplay", str(ask.STOP_BLIP_SOUND)], check=False),
            ],
        )

    @patch("ask.subprocess.Popen")
    def test_visualizer_uses_recording_file(self, popen):
        audio = Path("question.wav")
        question = Path("question.txt")
        ask.start_visualizer(audio, 4.0, question)
        popen.assert_called_once_with(
            [ask.sys.executable, str(ask.VISUALIZER_SCRIPT), str(audio), "4.0", str(question)]
        )


class InputDelayTest(unittest.TestCase):
    @patch("builtins.print")
    @patch("ask.time.sleep")
    @patch("ask.find_keyboards")
    def test_waits_for_replacement_keyboard(self, find_keyboards, sleep, _print):
        keyboard = Mock()
        find_keyboards.side_effect = [[], [], [keyboard]]
        self.assertEqual(ask.wait_for_keyboards(100, "KEY_RIGHTALT"), [keyboard])
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(ask.RECONNECT_SECONDS)

    def test_detects_disconnected_keyboard(self):
        keyboard = Mock()
        keyboard.active_keys.side_effect = OSError(errno.ENODEV, "gone")
        self.assertFalse(ask.keyboards_connected([keyboard]))

    @patch("builtins.print")
    @patch("ask.listen_connected")
    @patch("ask.wait_for_keyboards")
    def test_reconnects_in_process(self, wait_for_keyboards, listen_connected, _print):
        old_keyboard = Mock()
        new_keyboard = Mock()
        wait_for_keyboards.side_effect = [[old_keyboard], [new_keyboard]]
        listen_connected.side_effect = [
            OSError(errno.ENODEV, "gone"),
            KeyboardInterrupt,
        ]

        with self.assertRaises(KeyboardInterrupt):
            ask.listen(Mock(), Mock(), TEST_SETTINGS)

        self.assertEqual(wait_for_keyboards.call_count, 2)

    @patch("builtins.print")
    @patch("ask.start_visualizer")
    @patch("ask.start_recording")
    @patch("ask.play_blip")
    def test_short_press_never_starts_recording(
        self, play_blip, start_recording, start_visualizer, _print
    ):
        keyboard = Mock()
        keyboard.read.side_effect = [
            [SimpleNamespace(type=ask.ecodes.EV_KEY, code=ask.ecodes.KEY_RIGHTALT, value=1)],
            [SimpleNamespace(type=ask.ecodes.EV_KEY, code=ask.ecodes.KEY_RIGHTALT, value=0)],
        ]
        selections = [([keyboard], [], []), ([keyboard], [], []), KeyboardInterrupt]

        with (
            patch("ask.find_keyboards", return_value=[keyboard]),
            patch("ask.keyboards_connected", return_value=True),
            patch("ask.select.select", side_effect=selections),
            patch("ask.time.monotonic", side_effect=[0.0, 0.1]),
            self.assertRaises(KeyboardInterrupt),
        ):
            ask.listen(Mock(), Mock(), TEST_SETTINGS)

        start_recording.assert_not_called()
        start_visualizer.assert_not_called()
        play_blip.assert_not_called()

    @patch("builtins.print")
    @patch("ask.answer_question")
    @patch("ask.stop_transcript_stream")
    @patch("ask.start_transcript_stream")
    @patch("ask.stop_recording")
    @patch("ask.stop_visualizer")
    @patch("ask.start_visualizer")
    @patch("ask.start_recording")
    @patch("ask.play_blip")
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
            [SimpleNamespace(type=ask.ecodes.EV_KEY, code=ask.ecodes.KEY_RIGHTALT, value=1)],
            [SimpleNamespace(type=ask.ecodes.EV_KEY, code=ask.ecodes.KEY_RIGHTALT, value=0)],
        ]
        selections = [([keyboard], [], []), ([keyboard], [], []), KeyboardInterrupt]

        with (
            patch("ask.find_keyboards", return_value=[keyboard]),
            patch("ask.keyboards_connected", return_value=True),
            patch("ask.select.select", side_effect=selections),
            patch("ask.time.monotonic", side_effect=[0.0, TEST_SETTINGS["hold_seconds"]]),
            self.assertRaises(KeyboardInterrupt),
        ):
            ask.listen(Mock(), Mock(), TEST_SETTINGS)

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


if __name__ == "__main__":
    unittest.main()
