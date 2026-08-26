import torch

from src.model.feed_forward import FeedForward


def main():

    batch_size = 32
    sequence_length = 128
    embedding_dim = 256
    hidden_dim = 1024

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    print("Input shape:")
    print(x.shape)

    feed_forward = FeedForward(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )

    output = feed_forward(x)

    print("\nOutput shape:")
    print(output.shape)

    expected_shape = (
        batch_size,
        sequence_length,
        embedding_dim
    )

    assert output.shape == expected_shape

    print("\nFeed-forward test passed!")


if __name__ == "__main__":
    main()