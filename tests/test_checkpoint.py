import torch

from src.model.language_model import LanguageModel
from src.optimizer import create_optimizer
from src.checkpoint import load_checkpoint


def main():

    vocab_size = 8000
    context_length = 128
    embedding_dim = 256
    num_heads = 4
    feed_forward_dim = 1024
    num_layers = 4

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = LanguageModel(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        num_layers=num_layers
    ).to(device)

    optimizer = create_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=0.01
    )

    checkpoint_path = (
        "checkpoints/checkpoint_epoch_1.pt"
    )

    checkpoint = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        device
    )

    print("Checkpoint loaded successfully.")

    print(
        f"Saved epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Training loss: "
        f"{checkpoint['train_loss']:.4f}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint['validation_loss']:.4f}"
    )

    assert checkpoint["epoch"] == 1

    print("\nCheckpoint test passed!")


if __name__ == "__main__":
    main()