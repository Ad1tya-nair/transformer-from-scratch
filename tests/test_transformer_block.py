import torch

from src.model.transformer_block import TransformerBlock


def main():

    batch_size = 32
    sequence_length = 128
    embedding_dim = 256
    num_heads = 4
    feed_forward_dim = 1024

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    print("Input shape:")
    print(x.shape)

    block = TransformerBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim
    )

    output = block(x)

    print("\nOutput shape:")
    print(output.shape)

    expected_shape = (
        batch_size,
        sequence_length,
        embedding_dim
    )

    assert output.shape == expected_shape

    print("\nTransformer block test passed!")


if __name__ == "__main__":
    main()