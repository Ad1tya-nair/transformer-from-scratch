from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CONTEXT_LENGTH = 128

DATA_DIR = Path(
    r"C:\Users\Aditya N\Downloads\llm\tokenized"
)


# --------------------------------------------------
# LANGUAGE MODEL DATASET
# --------------------------------------------------

class LanguageModelDataset(Dataset):

    def __init__(
        self,
        split,
        context_length=CONTEXT_LENGTH
    ):

        self.context_length = context_length

        file_path = (
            DATA_DIR /
            f"{split}.bin"
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: {file_path}"
            )

        # Memory-map the binary file.
        #
        # The file is not completely loaded
        # into RAM.
        self.data = np.memmap(
            file_path,
            dtype=np.uint16,
            mode="r"
        )

        # Number of complete sequences
        self.num_sequences = (
            len(self.data)
            // context_length
        ) - 1

    def __len__(self):

        return self.num_sequences

    def __getitem__(self, index):

        start = (
            index *
            self.context_length
        )

        end = (
            start +
            self.context_length +
            1
        )

        chunk = self.data[
            start:end
        ]

        # Input
        x = chunk[:-1]

        # Target is shifted by one
        y = chunk[1:]

        # Convert to PyTorch tensors
        x = torch.tensor(
            x,
            dtype=torch.long
        )

        y = torch.tensor(
            y,
            dtype=torch.long
        )

        return x, y