import torch
import torch.nn as nn

from src.model.attention import MultiHeadAttention


class AttentionBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int
    ):
        super().__init__()

        self.norm = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

    def forward(self, x):

        # Pre-LayerNorm
        normalized_x = self.norm(x)

        # Multi-Head Self-Attention
        attention_output = self.attention(normalized_x)

        # Residual connection
        output = x + attention_output

        return output