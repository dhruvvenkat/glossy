import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import choose_voice


class ChooseVoiceTest(unittest.TestCase):
    @patch("builtins.print")
    @patch("builtins.input", return_value="en_US-amy-medium")
    @patch("choose_voice.subprocess.run")
    def test_downloads_and_selects_english_voice(self, run, _input, _print):
        run.side_effect = [
            subprocess.CompletedProcess(
                [], 0, "de_DE-thorsten-medium\nen_US-amy-medium\n", ""
            ),
            subprocess.CompletedProcess([], 0),
        ]

        with tempfile.TemporaryDirectory() as directory:
            voice_dir = Path(directory)
            with patch.object(choose_voice, "VOICE_DIR", voice_dir):
                choose_voice.main()
            self.assertEqual((voice_dir / "selected").read_text(), "en_US-amy-medium\n")

        download = run.call_args_list[1].args[0]
        self.assertIn("--download-dir", download)
        self.assertEqual(download[-1], "en_US-amy-medium")


if __name__ == "__main__":
    unittest.main()
