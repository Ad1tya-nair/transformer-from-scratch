import os

import torch
from torch.utils.data import DataLoader

from src.dataset import LanguageModelDataset
from src.model.language_model import LanguageModel
from src.loss import LanguageModelLoss
from src.optimizer import create_optimizer
from src.checkpoint import save_checkpoint


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 32

CONTEXT_LENGTH = 128

VOCAB_SIZE = 8000

EMBEDDING_DIM = 256

NUM_HEADS = 4

FEED_FORWARD_DIM = 1024

NUM_LAYERS = 4

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 0.01

NUM_EPOCHS = 10

# --------------------------------------------------
# EXPERIMENT CONFIGURATION
# --------------------------------------------------

CHECKPOINT_DIR = "checkpoints_fresh"

RESUME = False


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0

    for step, (x, targets) in enumerate(
        dataloader,
        start=1
    ):

        x = x.to(device)
        targets = targets.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(x)

        # Calculate loss
        loss = loss_fn(
            logits,
            targets
        )

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:

            print(
                f"Step {step}/{len(dataloader)} "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / len(dataloader)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def evaluate(
    model,
    dataloader,
    loss_fn,
    device
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for x, targets in dataloader:

            x = x.to(device)
            targets = targets.to(device)

            logits = model(x)

            loss = loss_fn(
                logits,
                targets
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # DEVICE
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:")
    print(device)

    # --------------------------------------------------
    # DATASETS
    # --------------------------------------------------

    train_dataset = LanguageModelDataset(
        split="train",
        context_length=CONTEXT_LENGTH
    )

    validation_dataset = LanguageModelDataset(
        split="validation",
        context_length=CONTEXT_LENGTH
    )

    print(
        f"\nTraining examples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_dataset):,}"
    )

    # --------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(
        f"Training batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader):,}"
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = LanguageModel(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        feed_forward_dim=FEED_FORWARD_DIM,
        num_layers=NUM_LAYERS
    ).to(device)

    # --------------------------------------------------
    # LOSS
    # --------------------------------------------------

    loss_fn = LanguageModelLoss()

    # --------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------

    optimizer = create_optimizer(
        model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------
    # CHECKPOINT DIRECTORY
    # --------------------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------
    # FRESH TRAINING
    # --------------------------------------------------

    start_epoch = 1

    if RESUME:

        raise RuntimeError(
            "RESUME is True, but no resume "
            "checkpoint has been configured."
        )

    print(
        "\nStarting training from scratch..."
    )

    # --------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------

    for epoch in range(
        start_epoch,
        NUM_EPOCHS + 1
    ):

        print(
            f"\n========== "
            f"Epoch {epoch}/{NUM_EPOCHS} "
            f"==========\n"
        )

        print(
            "Starting training...\n"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device
        )

        print(
            f"\nTraining loss: "
            f"{train_loss:.4f}"
        )

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        print(
            "\nStarting validation...\n"
        )

        validation_loss = evaluate(
            model,
            validation_loader,
            loss_fn,
            device
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )

        # --------------------------------------------------
        # CHECKPOINT
        # --------------------------------------------------
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            checkpoint_dir=CHECKPOINT_DIR
        )


if __name__ == "__main__":
    main()