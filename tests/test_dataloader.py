import torch

from torch.utils.data import DataLoader

from src.dataset import LanguageModelDataset


BATCH_SIZE = 32


def main():

    # --------------------------------------------------
    # CREATE DATASET
    # --------------------------------------------------

    dataset = LanguageModelDataset(
        split="train"
    )

    print(
        f"Dataset size: {len(dataset):,}"
    )

    # --------------------------------------------------
    # CREATE DATALOADER
    # --------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    # --------------------------------------------------
    # GET ONE BATCH
    # --------------------------------------------------

    x, y = next(
        iter(dataloader)
    )

    print("\nBatch information:")

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Target shape:",
        y.shape
    )

    print(
        "Input dtype:",
        x.dtype
    )

    print(
        "Target dtype:",
        y.dtype
    )

    # --------------------------------------------------
    # VERIFY SHIFT
    # --------------------------------------------------

    print(
        "\nShift verification:"
    )

    print(
        "First example:",
        bool(
            (
                x[0, 1:] ==
                y[0, :-1]
            ).all()
        )
    )


if __name__ == "__main__":
    main()