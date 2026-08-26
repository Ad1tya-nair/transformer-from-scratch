import torch.nn as nn

from src.model.attention import MultiHeadAttention
from src.model.feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=feed_forward_dim
        )

    def forward(self, x):

        # Pre-LN attention sub-block
        attention_output = self.attention(
            self.norm1(x)
        )

        x = x + attention_output

        # Pre-LN feed-forward sub-block
        feed_forward_output = self.feed_forward(
            self.norm2(x)
        )

        x = x + feed_forward_output

        return x