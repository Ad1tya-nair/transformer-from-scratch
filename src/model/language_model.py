import torch.nn as nn

from src.model.embeddings import TokenEmbedding
from src.model.positional import PositionalEmbedding
from src.model.transformer import Transformer
from src.model.lm_head import LMHead


class LanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        num_layers: int
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )

        self.positional_embedding = PositionalEmbedding(
    max_seq_len=context_length,
    embedding_dim=embedding_dim
)

        self.transformer = Transformer(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            feed_forward_dim=feed_forward_dim,
            num_layers=num_layers
        )

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.lm_head = LMHead(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size
        )

    def forward(self, x):

        x = self.token_embedding(x)

        x = self.positional_embedding(x)

        x = self.transformer(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits