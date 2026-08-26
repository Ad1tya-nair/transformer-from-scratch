import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.network(x)