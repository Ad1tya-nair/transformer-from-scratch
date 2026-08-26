import torch

from src.model.language_model import LanguageModel


def main():

    batch_size = 32
    context_length = 128
    vocab_size = 8000

    embedding_dim = 256
    num_heads = 4
    feed_forward_dim = 1024
    num_layers = 4

    x = torch.randint(
        0,
        vocab_size,
        (batch_size, context_length)
    )

    print("Input shape:")
    print(x.shape)

    model = LanguageModel(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        num_layers=num_layers
    )

    logits = model(x)

    print("\nOutput shape:")
    print(logits.shape)

    expected_shape = (
        batch_size,
        context_length,
        vocab_size
    )

    assert logits.shape == expected_shape

    print("\nLanguage model forward pass passed!")


if __name__ == "__main__":
    main()