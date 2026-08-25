import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0, (
            "embedding_dim must be divisible by num_heads"
        )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Q, K, V projections
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # Mix information from all heads
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        batch_size, sequence_length, _ = x.shape

        # ----------------------------------------
        # 1. Create Q, K, V
        # ----------------------------------------

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # [batch, sequence, embedding_dim]

        # ----------------------------------------
        # 2. Split into attention heads
        # ----------------------------------------

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        # Current:
        # [batch, sequence, heads, head_dim]

        # Rearrange:
        # [batch, heads, sequence, head_dim]

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ----------------------------------------
        # 3. Attention scores
        # ----------------------------------------

        scores = Q @ K.transpose(-2, -1)

        # [batch, heads, sequence, sequence]

        # ----------------------------------------
        # 4. Scale
        # ----------------------------------------

        scores = scores / math.sqrt(self.head_dim)

        # ----------------------------------------
        # 5. Causal mask
        # ----------------------------------------

        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        # ----------------------------------------
        # 6. Softmax
        # ----------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # ----------------------------------------
        # 7. Weighted Values
        # ----------------------------------------

        output = attention_weights @ V

        # [batch, heads, sequence, head_dim]

        # ----------------------------------------
        # 8. Combine heads
        # ----------------------------------------

        output = output.transpose(1, 2)

        # [batch, sequence, heads, head_dim]

        output = output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        # [batch, sequence, embedding_dim]

        # ----------------------------------------
        # 9. Output projection
        # ----------------------------------------

        output = self.output_projection(output)

        return output