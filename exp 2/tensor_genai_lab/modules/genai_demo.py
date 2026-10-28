"""A toy feature-vector transformation illustrating tensors in AI."""

import tensorflow as tf


def run_demo() -> None:
    """Transform a small numerical feature vector and explain its role."""
    print("\nMODULE 8 - CONNECTION TO GENERATIVE AI")
    print("-" * 58)
    input_features = tf.constant([[0.2, 0.8, 0.5]], dtype=tf.float32)
    weights = tf.constant(
        [[0.5, 0.1], [0.3, 0.7], [0.2, 0.4]],
        dtype=tf.float32,
    )
    bias = tf.constant([0.1, -0.1], dtype=tf.float32)
    transformed_features = tf.matmul(input_features, weights) + bias

    print(f"Toy input features (batch, feature):\n{input_features.numpy()}")
    print(f"Input shape: {input_features.shape}")
    print(f"Toy weight matrix:\n{weights.numpy()}")
    print(f"Linear transformation (input @ weights + bias):\n{transformed_features.numpy()}")
    print(f"Output shape: {transformed_features.shape}")
    print(
        "In neural networks, tensors carry batches of input features, parameters, and "
        "intermediate activations. Transformers use tensors for token representations and "
        "attention computations; large language models use these operations to process "
        "numerical token embeddings and produce predictions. This tiny example is only a "
        "fixed arithmetic transformation: it does not train or implement a generative model."
    )
