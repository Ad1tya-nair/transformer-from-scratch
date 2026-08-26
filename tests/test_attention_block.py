import torch

from src.model.attention_block import AttentionBlock


def main():

    batch_size = 32
    sequence_length = 128
    embedding_dim = 256
    num_heads = 4

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    print("Input shape:")
    print(x.shape)

    block = AttentionBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads
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

    print("\nAttention block test passed!")


if __name__ == "__main__":
    main()