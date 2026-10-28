import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from config import MusicGenConfig
from music_generator import MusicGenerator


class MusicGeneratorTests(unittest.TestCase):
    def test_config_rejects_invalid_duration(self):
        config = MusicGenConfig()
        for value in (0, 31, -1):
            with self.subTest(duration=value), self.assertRaises(ValueError):
                config.validate_duration(value)

    def test_waveform_conversion_supports_batch_and_stereo(self):
        stereo = torch.tensor([[[0.5, -0.5], [-0.5, 0.5]]])
        waveform = MusicGenerator._to_mono_waveform(stereo)
        self.assertEqual(waveform.shape, (2,))
        np.testing.assert_allclose(waveform, np.array([0.0, 0.0], dtype=np.float32))

    def test_wav_saved_with_pcm_header_and_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "musicgen_output.wav"
            waveform = np.array([0.0, 0.5, -0.5], dtype=np.float32)
            MusicGenerator.save_wav(destination, waveform, 32_000)
            self.assertTrue(destination.is_file())
            rate, data = wavfile.read(destination)
            self.assertEqual(rate, 32_000)
            self.assertEqual(data.dtype, np.int16)
            self.assertEqual(data.shape, waveform.shape)
            with wave.open(str(destination), "rb") as audio:
                self.assertEqual(audio.getframerate(), 32_000)
                self.assertEqual(audio.getnframes(), 3)

    def test_empty_prompt_is_rejected_before_model_load(self):
        generator = MusicGenerator()
        with self.assertRaisesRegex(ValueError, "prompt"):
            generator.generate("   ")
        self.assertFalse(generator.is_loaded)


if __name__ == "__main__":
    unittest.main()
