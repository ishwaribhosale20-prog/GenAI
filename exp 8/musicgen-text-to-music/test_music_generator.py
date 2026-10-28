"""Fast unit tests that do not download or load model weights."""

import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from music_generator import (
    MusicGenerator,
    save_waveform_as_wav,
    validate_generation_request,
)


class GenerationRequestTests(unittest.TestCase):
    def test_accepts_valid_request(self):
        validate_generation_request("calm piano and flute", 5)

    def test_rejects_empty_prompt(self):
        with self.assertRaisesRegex(ValueError, "prompt"):
            validate_generation_request("   ", 5)

    def test_rejects_non_integer_duration(self):
        with self.assertRaisesRegex(ValueError, "whole number"):
            validate_generation_request("soft piano", 2.5)

    def test_rejects_duration_outside_supported_range(self):
        with self.assertRaisesRegex(ValueError, "between"):
            validate_generation_request("soft piano", 21)

    def test_rejects_non_finite_duration(self):
        for duration in (float("nan"), float("inf")):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(ValueError, "between"):
                    validate_generation_request("soft piano", duration)


class WavOutputTests(unittest.TestCase):
    def test_writes_playable_mono_pcm_wav(self):
        waveform = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "musicgen_output.wav"
            saved_path = save_waveform_as_wav(waveform, 32_000, output_path)

            self.assertTrue(saved_path.is_file())
            with wave.open(str(saved_path), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 32_000)
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getnframes(), 5)

    def test_writes_playable_stereo_pcm_wav(self):
        waveform = np.zeros((4, 2), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "stereo.wav"
            save_waveform_as_wav(waveform, 32_000, output_path)

            with wave.open(str(output_path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 2)
                self.assertEqual(wav_file.getnframes(), 4)


class StubModelGenerationTests(unittest.TestCase):
    def test_generation_writes_audio_with_model_sample_rate_and_duration(self):
        class EncodedInputs(dict):
            def to(self, device):
                return self

        class FakeProcessor:
            def __init__(self):
                self.prompt = None

            def __call__(self, **kwargs):
                self.prompt = kwargs["text"][0]
                return EncodedInputs(input_ids=torch.tensor([[1, 2, 3]]))

        class FakeModel:
            config = SimpleNamespace(
                audio_encoder=SimpleNamespace(sampling_rate=32_000)
            )

            def __init__(self):
                self.generation_options = None

            def generate(self, **kwargs):
                self.generation_options = kwargs
                return torch.zeros((1, 1, 32_000))

        processor = FakeProcessor()
        model = FakeModel()
        generator = MusicGenerator()
        generator._processor = processor
        generator._model = model
        generator._torch = torch
        generator._device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "generated.wav"
            result = generator.generate("  calm piano  ", 1, output_path)

            self.assertEqual(processor.prompt, "calm piano")
            self.assertEqual(model.generation_options["max_new_tokens"], 50)
            self.assertEqual(model.generation_options["guidance_scale"], 3.0)
            self.assertEqual(result.sample_rate, 32_000)
            self.assertEqual(result.duration_seconds, 1.0)
            with wave.open(str(result.path), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 32_000)
                self.assertEqual(wav_file.getnframes(), 32_000)


if __name__ == "__main__":
    unittest.main()
