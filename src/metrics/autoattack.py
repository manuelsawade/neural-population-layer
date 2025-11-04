from __future__ import annotations
from typing import Dict

import torch
from torch import nn

def autoattack_metric(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    eps: float,
    norm: str = "Linf",
    log_path: str | None = None,
) -> Dict[str, float]:
    ...
    # model.eval()
    # device = torch.device(device)

    # adversary = AutoAttack(model, norm=norm, eps=eps, version='standard', device=device, log_path=f"{log_path}/autoattack.txt")

    # #total, correct_clean = 0, 0
    # #correct_worst = 0

    # #with torch.no_grad():
    # pred = model(x).argmax(1)
    # correct_clean = pred.eq(y).sum().item()

    # x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
    # #with torch.no_grad():
    # pred_adv = model(x_adv).argmax(1)
    # correct_worst = pred_adv.eq(y).sum().item()
    # total = x.size(0)
    # #correct_worst += correct_adv


    # clean_acc = 100.0 * correct_clean / max(total, 1)
    # robust_acc = 100.0 * correct_worst / max(total, 1)
    # return clean_acc, robust_acc
    #     "clean_acc_%": clean_acc,
    #     "robust_acc_%": robust_acc,
    # }