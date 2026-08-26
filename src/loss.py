import torch
import torch.nn as nn


class LanguageModelLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:

        batch_size, sequence_length, vocab_size = logits.shape

        logits = logits.reshape(
            batch_size * sequence_length,
            vocab_size
        )

        targets = targets.reshape(
            batch_size * sequence_length
        )

        loss = self.loss_fn(
            logits,
            targets
        )

        return loss
    