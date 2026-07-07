import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import ask

TEST_SETTINGS = {
    "model": "test-model",
    "reasoning_effort": "none",
    "hold_seconds": 0.35,
    "button": "KEY_RIGHTALT",
}


class ConfigTest(unittest.TestCase):
    def test_loads_runtime_settings(self):
        settings = {
            "model": "gpt-5.5",
            "reasoning_effort": "none",
            "hold_seconds": 0.5,
            "button": "KEY_HOME",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(settings))
            self.assertEqual(ask.load_settings(path), settings)

    def test_rejects_unknown_button(self):
        settings = {
            "model": "gpt-5.5",
            "reasoning_effort": "none",
            "hold_seconds": 0.5,
            "button": "KEY_NOT_REAL",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(settings))
            with self.assertRaisesRegex(RuntimeError, "evdev key name"):
                ask.load_settings(path)


class AnswerQuestionTest(unittest.TestCase):
    @patch("ask.speak")
    def test_transcribes_answers_and_speaks(self, speak):
        client = Mock()
        client.audio.transcriptions.create.return_value = SimpleNamespace(
            text="What is a mutex?"
        )
        client.responses.create.return_value = SimpleNamespace(
            output_text="A mutex permits one thread at a time."
        )

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "question.wav"
            audio.write_bytes(b"RIFF fake audio")
            ask.answer_question(client, TEST_SETTINGS, audio)

        transcription = client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(transcription["model"], "whisper-1")
        client.responses.create.assert_called_once_with(
            model="test-model",
            instructions=ask.SYSTEM_PROMPT,
            input="What is a mutex?",
            reasoning={"effort": "none"},
        )
        speak.assert_called_once_with("A mutex permits one thread at a time.")

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
        ask.start_visualizer(audio)
        popen.assert_called_once_with(
            [ask.sys.executable, str(ask.VISUALIZER_SCRIPT), str(audio)]
        )


class InputDelayTest(unittest.TestCase):
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
            patch("ask.select.select", side_effect=selections),
            patch("ask.time.monotonic", side_effect=[0.0, 0.1]),
            self.assertRaises(KeyboardInterrupt),
        ):
            ask.listen(Mock(), TEST_SETTINGS)

        start_recording.assert_not_called()
        start_visualizer.assert_not_called()
        play_blip.assert_not_called()

    @patch("builtins.print")
    @patch("ask.answer_question")
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
        answer_question,
        _print,
    ):
        order = []
        visualizer = Mock()
        play_blip.side_effect = lambda sound: order.append(sound.name)
        start_recording.side_effect = lambda _path: order.append("record") or Mock()
        start_visualizer.side_effect = (
            lambda _path: order.append("visualizer-start") or visualizer
        )
        stop_visualizer.side_effect = lambda _process: order.append("visualizer-stop")
        stop_recording.side_effect = lambda *_args: order.append("stop")
        answer_question.side_effect = lambda *_args: order.append("answer")
        keyboard = Mock()
        keyboard.read.side_effect = [
            [SimpleNamespace(type=ask.ecodes.EV_KEY, code=ask.ecodes.KEY_RIGHTALT, value=1)],
            [SimpleNamespace(type=ask.ecodes.EV_KEY, code=ask.ecodes.KEY_RIGHTALT, value=0)],
        ]
        selections = [([keyboard], [], []), ([keyboard], [], []), KeyboardInterrupt]

        with (
            patch("ask.find_keyboards", return_value=[keyboard]),
            patch("ask.select.select", side_effect=selections),
            patch("ask.time.monotonic", side_effect=[0.0, TEST_SETTINGS["hold_seconds"]]),
            self.assertRaises(KeyboardInterrupt),
        ):
            ask.listen(Mock(), TEST_SETTINGS)

        start_recording.assert_called_once()
        self.assertEqual(
            order,
            [
                "blip.mp3",
                "record",
                "visualizer-start",
                "stop",
                "visualizer-stop",
                "blip-reversed.mp3",
                "answer",
            ],
        )
        stop_visualizer.assert_called_once_with(visualizer)
        stop_recording.assert_called_once()
        answer_question.assert_called_once()


if __name__ == "__main__":
    unittest.main()
