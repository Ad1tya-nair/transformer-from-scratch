import torch

from src.model.transformer import Transformer


def main():

    batch_size = 32
    sequence_length = 128
    embedding_dim = 256
    num_heads = 4
    feed_forward_dim = 1024
    num_layers = 4

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    print("Input shape:")
    print(x.shape)

    model = Transformer(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        num_layers=num_layers
    )

    output = model(x)

    print("\nOutput shape:")
    print(output.shape)

    expected_shape = (
        batch_size,
        sequence_length,
        embedding_dim
    )

    assert output.shape == expected_shape

    print("\nTransformer stack test passed!")


if __name__ == "__main__":
    main()