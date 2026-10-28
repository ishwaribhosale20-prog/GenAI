import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import torch
from PIL import Image
from torch.optim import Adam

from dataset import load_dataset, make_dataloader
from models import Discriminator, Generator, mix_styles
from utils.checkpoint import load_checkpoint, save_checkpoint


class StyleGANProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_models(self):
        generator = Generator(latent_dim=8, style_dim=8, mapping_depth=2, channel_base=2)
        discriminator = Discriminator(channel_base=2)
        return generator, discriminator

    def test_mapping_generation_shapes_and_value_range(self):
        generator, discriminator = self.make_models()
        z = torch.randn(2, generator.latent_dim)
        w = generator.mapping(z)
        styles = generator.styles_from_z(z)
        images = generator(z, noise_seed=4)
        logits = discriminator(images)
        self.assertEqual(tuple(w.shape), (2, 8))
        self.assertEqual(tuple(styles.shape), (2, 9, 8))
        self.assertEqual(tuple(images.shape), (2, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertGreaterEqual(float(images.detach().min()), -1.0)
        self.assertLessEqual(float(images.detach().max()), 1.0)

    def test_style_mixing_uses_both_inputs_at_cutoff(self):
        styles_a = torch.zeros(1, 9, 4)
        styles_b = torch.ones(1, 9, 4)
        mixed = mix_styles(styles_a, styles_b, 4)
        self.assertTrue(torch.equal(mixed[:, :4], styles_a[:, :4]))
        self.assertTrue(torch.equal(mixed[:, 4:], styles_b[:, 4:]))

    def test_noise_seed_is_reproducible_and_changes_output(self):
        generator, _ = self.make_models()
        z = torch.randn(1, generator.latent_dim)
        first = generator(z, noise_seed=100)
        same = generator(z, noise_seed=100)
        different = generator(z, noise_seed=101)
        self.assertTrue(torch.equal(first, same))
        self.assertFalse(torch.equal(first, different))
        self.assertTrue(torch.equal(first, generator(z, noise_seed=100, noise_strength=1)))

    def test_zip_dataset_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "faces.zip"
            image_path = Path(directory) / "face.png"
            Image.new("RGB", (80, 100), (255, 128, 0)).save(image_path)
            with ZipFile(archive_path, "w") as archive:
                archive.write(image_path, "images/face.png")
            dataset = load_dataset(archive_path, resolution=64)
            sample = dataset[0]
            self.assertEqual(tuple(sample.shape), (3, 64, 64))
            self.assertGreaterEqual(float(sample.min()), -1.0)
            self.assertLessEqual(float(sample.max()), 1.0)
            dataloader = make_dataloader(
                archive_path, batch_size=1, resolution=64, num_workers=0
            )
            batch = next(iter(dataloader))
            self.assertEqual(tuple(batch.shape), (1, 3, 64, 64))

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            generator, discriminator = self.make_models()
            g_optimizer, d_optimizer = Adam(generator.parameters()), Adam(discriminator.parameters())
            checkpoint = Path(directory) / "latest.pt"
            save_checkpoint(
                checkpoint,
                generator,
                discriminator,
                g_optimizer,
                d_optimizer,
                epoch=3,
                history={"generator": [1.0], "discriminator": [2.0]},
                config={"resolution": 64},
            )
            original = [parameter.detach().clone() for parameter in generator.parameters()]
            with torch.no_grad():
                next(generator.parameters()).add_(1)
            loaded = load_checkpoint(
                checkpoint,
                generator,
                discriminator,
                g_optimizer,
                d_optimizer,
            )
            self.assertEqual(loaded["epoch"], 3)
            self.assertTrue(
                all(torch.equal(before, after) for before, after in zip(original, generator.parameters()))
            )


if __name__ == "__main__":
    unittest.main()
