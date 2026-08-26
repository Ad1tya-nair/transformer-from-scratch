import torch.nn as nn

from src.model.transformer_block import TransformerBlock


class Transformer(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        num_layers: int
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                feed_forward_dim=feed_forward_dim
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):

        for block in self.blocks:
            x = block(x)

        return x