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
            voice = Path(directory) / "voice.onnx"
            voice.touch()
            with patch.object(ask, "VOICE_MODEL", voice):
                ask.speak("A clear answer.")

        piper, player = run.call_args_list
        self.assertEqual(piper.args[0][0:3], [ask.sys.executable, "-m", "piper"])
        self.assertEqual(piper.kwargs["input"], "A clear answer.\n")
        self.assertEqual(player.args[0][0:2], ["aplay", "--quiet"])


if __name__ == "__main__":
    unittest.main()
