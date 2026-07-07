import tempfile
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from visualizer import audio_level, primary_geometry


class AudioLevelTest(unittest.TestCase):
    @patch("visualizer.subprocess.run")
    def test_uses_primary_monitor(self, run):
        run.return_value = SimpleNamespace(
            stdout="Monitors: 2\n 0: +*DP-6 1920/600x1080/330+0+0 DP-6\n"
        )
        self.assertEqual(primary_geometry(3000, 1920), (0, 0, 1920, 1080))

    def test_tracks_recent_pcm_amplitude(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "recording.wav"
            audio.write_bytes(b"\0" * 44 + array("h", [3000, -3000] * 100).tobytes())
            self.assertAlmostEqual(audio_level(audio), 0.5)
            self.assertEqual(audio_level(audio, sensitivity=2), 1.0)

    def test_missing_audio_is_silent(self):
        self.assertEqual(audio_level(Path("missing.wav")), 0.0)


if __name__ == "__main__":
    unittest.main()
