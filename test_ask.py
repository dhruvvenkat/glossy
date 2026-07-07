import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ask


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
            ask.answer_question(client, "test-model", audio)

        transcription = client.audio.transcriptions.create.call_args.kwargs
        self.assertEqual(transcription["model"], "whisper-1")
        client.responses.create.assert_called_once_with(
            model="test-model",
            instructions=ask.SYSTEM_PROMPT,
            input="What is a mutex?",
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


class InputDelayTest(unittest.TestCase):
    @patch("builtins.print")
    @patch("ask.start_recording")
    @patch("ask.play_blip")
    def test_short_press_never_starts_recording(self, play_blip, start_recording, _print):
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
            ask.listen(Mock(), "test-model")

        start_recording.assert_not_called()
        play_blip.assert_not_called()

    @patch("builtins.print")
    @patch("ask.answer_question")
    @patch("ask.stop_recording")
    @patch("ask.start_recording")
    @patch("ask.play_blip")
    def test_recording_starts_after_threshold(
        self, play_blip, start_recording, stop_recording, answer_question, _print
    ):
        order = []
        play_blip.side_effect = lambda: order.append("blip")
        start_recording.side_effect = lambda _path: order.append("record") or Mock()
        keyboard = Mock()
        keyboard.read.side_effect = [
            [SimpleNamespace(type=ask.ecodes.EV_KEY, code=ask.ecodes.KEY_RIGHTALT, value=1)],
            [SimpleNamespace(type=ask.ecodes.EV_KEY, code=ask.ecodes.KEY_RIGHTALT, value=0)],
        ]
        selections = [([keyboard], [], []), ([keyboard], [], []), KeyboardInterrupt]

        with (
            patch("ask.find_keyboards", return_value=[keyboard]),
            patch("ask.select.select", side_effect=selections),
            patch("ask.time.monotonic", side_effect=[0.0, ask.MIN_HOLD_SECONDS]),
            self.assertRaises(KeyboardInterrupt),
        ):
            ask.listen(Mock(), "test-model")

        start_recording.assert_called_once()
        self.assertEqual(order, ["blip", "record"])
        stop_recording.assert_called_once()
        answer_question.assert_called_once()


if __name__ == "__main__":
    unittest.main()
