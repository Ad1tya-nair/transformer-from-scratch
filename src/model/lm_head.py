import torch.nn as nn


class LMHead(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int
    ):
        super().__init__()

        self.projection = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):
        return self.projection(x)