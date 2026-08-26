import torch

from src.model.language_model import LanguageModel
from src.loss import LanguageModelLoss
from src.optimizer import create_optimizer


def main():

    batch_size = 2
    sequence_length = 128
    vocab_size = 8000

    embedding_dim = 256
    num_heads = 4
    feed_forward_dim = 1024
    num_layers = 4

    model = LanguageModel(
        vocab_size=vocab_size,
        context_length=sequence_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        num_layers=num_layers
    )

    loss_fn = LanguageModelLoss()

    optimizer = create_optimizer(model)

    x = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length)
    )

    targets = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length)
    )

    # Save a copy of one parameter before training.
    parameter_before = (
        model.token_embedding.embedding.weight
        .detach()
        .clone()
    )

    # Forward pass
    logits = model(x)

    # Calculate loss
    loss = loss_fn(
        logits,
        targets
    )

    print("Loss before update:")
    print(loss.item())

    # Backpropagation
    optimizer.zero_grad()

    loss.backward()

    # Parameter update
    optimizer.step()

    parameter_after = (
        model.token_embedding.embedding.weight
        .detach()
        .clone()
    )

    changed = not torch.equal(
        parameter_before,
        parameter_after
    )

    print("\nParameter changed:")
    print(changed)

    assert changed

    print("\nTraining step test passed!")


if __name__ == "__main__":
    main()