import tempfile
import unittest
from array import array
from pathlib import Path

from visualizer import audio_level


class AudioLevelTest(unittest.TestCase):
    def test_tracks_recent_pcm_amplitude(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "recording.wav"
            audio.write_bytes(b"\0" * 44 + array("h", [3000, -3000] * 100).tobytes())
            self.assertAlmostEqual(audio_level(audio), 0.5)

    def test_missing_audio_is_silent(self):
        self.assertEqual(audio_level(Path("missing.wav")), 0.0)


if __name__ == "__main__":
    unittest.main()
