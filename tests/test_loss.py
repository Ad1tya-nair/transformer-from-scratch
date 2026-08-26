import torch

from src.loss import LanguageModelLoss


def main():

    batch_size = 32
    sequence_length = 128
    vocab_size = 8000

    logits = torch.randn(
        batch_size,
        sequence_length,
        vocab_size
    )

    targets = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length)
    )

    loss_fn = LanguageModelLoss()

    loss = loss_fn(
        logits,
        targets
    )

    print("Logits shape:")
    print(logits.shape)

    print("\nTargets shape:")
    print(targets.shape)

    print("\nLoss:")
    print(loss.item())

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    print("\nLoss test passed!")


if __name__ == "__main__":
    main()