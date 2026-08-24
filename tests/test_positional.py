import torch

from src.model.positional import PositionalEmbedding


def main():

    # Model configuration
    batch_size = 32
    sequence_length = 128
    embedding_dim = 256

    # Simulate the output of the token embedding layer
    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    print("Input shape:")
    print(x.shape)

    # Create positional embedding layer
    positional_embedding = PositionalEmbedding(
        max_seq_len=sequence_length,
        embedding_dim=embedding_dim
    )

    # Apply positional embeddings
    output = positional_embedding(x)

    print("\nOutput shape:")
    print(output.shape)

    # Check shape
    assert output.shape == (
        batch_size,
        sequence_length,
        embedding_dim
    )

    # Check that positional information actually changed the tensor
    assert not torch.equal(x, output)

    print("\nPositional embedding test passed!")


if __name__ == "__main__":
    main()