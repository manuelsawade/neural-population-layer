from typing import Callable
import torch
from torch import Tensor, nn


def sharpness_metric(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    noise: float,
    base_loss: float,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    append_to: dict[str, list[float]] = {}
) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad == False: continue

        if not name.split('.')[2] in ["weight", "bias"]: continue

        original = param.data.clone()
        
        param_noise = noise * torch.randn_like(param)
        param.data += param_noise

        perturbed_loss = loss_fn(model(x), y).item()

        if name not in append_to:
            append_to[name] = []

        append_to[name].append(perturbed_loss - base_loss)

        param.data = original