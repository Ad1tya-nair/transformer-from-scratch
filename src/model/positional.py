import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.

    Each position from 0 to max_seq_len - 1
    gets its own embedding vector.
    """

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()

        self.position_embedding = nn.Embedding(
            max_seq_len,
            embedding_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            Token embeddings with shape
            [batch_size, sequence_length, embedding_dim]

        Returns:
            Token embeddings + positional embeddings
            [batch_size, sequence_length, embedding_dim]
        """

        batch_size, sequence_length, embedding_dim = x.shape

        # Positions: [0, 1, 2, ..., sequence_length - 1]
        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        # Shape: [sequence_length, embedding_dim]
        position_vectors = self.position_embedding(positions)

        # Broadcasting:
        # [batch, sequence, embedding_dim]
        # +
        # [sequence, embedding_dim]
        #
        # → [batch, sequence, embedding_dim]
        return x + position_vectors