import torch
import torch.nn.functional as F
from typing import Optional

@torch.no_grad()
def topk_candidate_classes(logits: torch.Tensor, true_labels: torch.Tensor, topk: int):
    B, C = logits.shape
    k_needed = min(C, topk + 1)
    top_vals, top_idx = logits.topk(k_needed, dim=1)
    candidates = []
    for i in range(B):
        row = top_idx[i]
        filtered = row[row != true_labels[i]]
        if filtered.numel() >= topk:
            candidates.append(filtered[:topk])
        else:
            mask = torch.ones(C, dtype=torch.bool, device=logits.device)
            mask[true_labels[i]] = False
            remaining = torch.nonzero(mask, as_tuple=False).squeeze(1)
            comb = torch.cat([filtered, remaining])
            candidates.append(comb[:topk])
    return torch.stack(candidates, dim=0)


def noise_sensitivity_metric(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    attack: str = "fgsm",    
    topk: int = 10,             
    max_score: float = 100.0,     
    clamp_grad: bool = False,
    append_to: list[float] = []    
):
    x_in = x.clone().detach().requires_grad_(True)

    model.eval()
    with torch.enable_grad():
        B = x.shape[0]

        pred = model(x_in)
        ce_loss = F.cross_entropy(pred, y, reduction="sum")  

        if x_in.grad is not None:
            x_in.grad.detach_()
            x_in.grad.zero_()
        model.zero_grad()
        ce_loss.backward(retain_graph=True)
        grad_input = x_in.grad.detach().clone()

        if clamp_grad:
            grad_input = grad_input.clamp(-1e6, 1e6)

        if attack.lower() == "fgsm":
            v = torch.sign(grad_input)
        elif attack.lower() == "fgm":
            v = grad_input             

        candidates = topk_candidate_classes(pred.detach(), y, topk=topk) 
        nss_list = torch.full((B,), float(max_score), device=x.device)

        logits_det = pred.detach()
        logit_y = logits_det[torch.arange(B), y]  

        for k in range(candidates.shape[1]):
            t_idx = candidates[:, k]  # (B,)

            if x_in.grad is not None:
                x_in.grad.detach_()
                x_in.grad.zero_()
            model.zero_grad()

            logits_for_back = model(x_in) 

            s_vals = logits_for_back[torch.arange(B), t_idx] - logits_for_back[torch.arange(B), y]
            s_sum = s_vals.sum()
            s_sum.backward(retain_graph=True)
            grad_diff = x_in.grad.detach().clone()  
                                                 

            rate = (grad_diff * v).view(B, -1).sum(dim=1) 

            gap = (logit_y - logits_det[torch.arange(B), t_idx]).detach()  

            eps = 1e-12

            positive_mask = rate > 1e-12
            cand_nss = torch.full((B,), float(max_score), device=x.device)
            cand_nss[positive_mask] = (gap[positive_mask] / (rate[positive_mask] + eps)).clamp(min=0.0, max=float(max_score))
            nss_list = torch.min(nss_list, cand_nss)

    if not attack in append_to:
        append_to[attack] = []

    append_to[attack].extend(nss_list.to_sparse().values().tolist())  # shape (B,)
