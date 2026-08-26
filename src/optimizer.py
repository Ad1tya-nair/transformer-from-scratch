import torch


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01
):
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )