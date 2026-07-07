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


if __name__ == "__main__":
    unittest.main()
