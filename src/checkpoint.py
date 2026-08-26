from pathlib import Path

import torch


# ============================================================
# DEFAULT CHECKPOINT DIRECTORY
# ============================================================

CHECKPOINT_DIR = (
    Path(__file__).resolve().parent.parent
    / "checkpoints"
)


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    validation_loss,
    checkpoint_dir=CHECKPOINT_DIR
):

    checkpoint_dir = Path(
        checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        checkpoint_dir /
        f"checkpoint_epoch_{epoch}.pt"
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "validation_loss":
            validation_loss
    }

    torch.save(
        checkpoint,
        checkpoint_path
    )

    print(
        f"Checkpoint saved: "
        f"{checkpoint_path}"
    )


# ============================================================
# GET LATEST CHECKPOINT
# ============================================================

def get_latest_checkpoint(
    checkpoint_dir=CHECKPOINT_DIR
):

    checkpoint_dir = Path(
        checkpoint_dir
    )

    if not checkpoint_dir.exists():

        return None

    checkpoints = list(
        checkpoint_dir.glob(
            "checkpoint_epoch_*.pt"
        )
    )

    if not checkpoints:

        return None

    def get_epoch(path):

        return int(
            path.stem.split("_")[-1]
        )

    return max(
        checkpoints,
        key=get_epoch
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu"
):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    return checkpoint