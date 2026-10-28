"""Small smoke tests for the educational demonstrations."""

import unittest

import numpy as np
import tensorflow as tf

from modules.tensor_arithmetic import add_compatible_tensors
from modules.tensor_creation import create_examples


class TensorLabTests(unittest.TestCase):
    def test_examples_cover_expected_ranks(self) -> None:
        examples = create_examples()
        self.assertEqual([int(tf.rank(value).numpy()) for value in examples.values()], [0, 1, 2, 3, 4])

    def test_compatible_addition_uses_tensorflow(self) -> None:
        result = add_compatible_tensors(
            tf.constant([[1.0, 2.0], [3.0, 4.0]]),
            tf.constant([10.0, 20.0]),
        )
        np.testing.assert_allclose(result.numpy(), [[11.0, 22.0], [13.0, 24.0]])

    def test_incompatible_addition_has_readable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "broadcast-compatible"):
            add_compatible_tensors(tf.ones((2, 2)), tf.ones((3,)))


if __name__ == "__main__":
    unittest.main()
