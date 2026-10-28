import tempfile
import unittest
from pathlib import Path

from config import GenerationSettings
from video_generator import VideoGenerationError, save_video, validate_prompt


class PromptValidationTests(unittest.TestCase):
    def test_prompt_is_trimmed(self):
        self.assertEqual(validate_prompt("  a quiet lake  "), "a quiet lake")

    def test_empty_prompt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Enter a prompt"):
            validate_prompt("  ")

    def test_long_prompt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "1000 characters"):
            validate_prompt("x" * 1001)


class SettingsValidationTests(unittest.TestCase):
    def test_default_settings_are_valid(self):
        self.assertIsInstance(GenerationSettings().validate(), GenerationSettings)

    def test_out_of_range_values_are_rejected(self):
        invalid_settings = (
            (GenerationSettings(num_frames=33), "between 8 and 32"),
            (GenerationSettings(fps=0), "between 1 and 24"),
            (GenerationSettings(num_inference_steps=51), "between 10 and 50"),
            (GenerationSettings(guidance_scale=15.1), "between 1 and 15"),
            (GenerationSettings(seed=-2), "4294967295"),
        )
        for settings, message in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(ValueError, message):
                    settings.validate()

    def test_seed_minus_one_is_supported(self):
        self.assertEqual(GenerationSettings(seed=-1).validate().seed, -1)


class VideoExportTests(unittest.TestCase):
    def test_empty_frame_list_is_rejected(self):
        with self.assertRaisesRegex(VideoGenerationError, "no frames"):
            save_video([], fps=8)

    def test_nonempty_frames_encode_as_mp4(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed")

        frames = [
            np.full((32, 48, 3), (1.0, 0.0, 0.0), dtype=np.float32),
            np.full((32, 48, 3), (0.0, 1.0, 0.0), dtype=np.float32),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = save_video(frames, fps=4, output_dir=Path(directory), prompt="test scene")
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(path.suffix, ".mp4")
            import imageio.v2 as imageio

            reader = imageio.get_reader(path, format="FFMPEG")
            try:
                self.assertEqual(reader.count_frames(), 2)
                first_decoded_frame = reader.get_data(0)
                self.assertEqual(first_decoded_frame.shape, (32, 48, 3))
                self.assertGreater(int(first_decoded_frame[16, 24, 0]), 150)
                self.assertLess(int(first_decoded_frame[16, 24, 1]), 100)
                self.assertAlmostEqual(reader.get_meta_data()["fps"], 4, places=1)
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()
