import torch

from src.model.language_model import LanguageModel
from src.loss import LanguageModelLoss
from src.optimizer import create_optimizer


def main():

    # Small test configuration
    batch_size = 2
    sequence_length = 128
    vocab_size = 8000

    embedding_dim = 256
    num_heads = 4
    feed_forward_dim = 1024
    num_layers = 4

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:")
    print(device)

    # Create model
    model = LanguageModel(
        vocab_size=vocab_size,
        context_length=sequence_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        num_layers=num_layers
    ).to(device)

    # Loss and optimizer
    loss_fn = LanguageModelLoss()

    optimizer = create_optimizer(model)

    # One fixed batch
    x = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        device=device
    )

    targets = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        device=device
    )

    print("\nTraining on the same batch repeatedly...\n")

    losses = []

    for step in range(10):

        optimizer.zero_grad()

        # Forward pass
        logits = model(x)

        # Loss
        loss = loss_fn(
            logits,
            targets
        )

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        losses.append(loss.item())

        print(
            f"Step {step + 1}: "
            f"loss = {loss.item():.4f}"
        )

    print("\nInitial loss:")
    print(losses[0])

    print("\nFinal loss:")
    print(losses[-1])

    assert torch.isfinite(torch.tensor(losses)).all()

    print("\nTraining loop smoke test passed!")


if __name__ == "__main__":
    main()