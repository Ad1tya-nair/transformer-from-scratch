import torch

from src.model.lm_head import LMHead


def main():

    batch_size = 32
    sequence_length = 128
    embedding_dim = 256
    vocab_size = 8000

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    print("Input shape:")
    print(x.shape)

    lm_head = LMHead(
        embedding_dim=embedding_dim,
        vocab_size=vocab_size
    )

    logits = lm_head(x)

    print("\nOutput shape:")
    print(logits.shape)

    expected_shape = (
        batch_size,
        sequence_length,
        vocab_size
    )

    assert logits.shape == expected_shape

    print("\nLM head test passed!")


if __name__ == "__main__":
    main()