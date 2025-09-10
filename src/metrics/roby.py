from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch


def _validate_inputs(features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        raise ValueError(f"features must be 2D [N, D], got shape {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D [N], got shape {labels.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same number of samples")
    return features, labels


def _minmax_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x_min = torch.min(x)
    x_max = torch.max(x)
    denom = x_max - x_min
    if torch.isclose(denom, torch.tensor(0.0, device=x.device), atol=eps):
        return torch.zeros_like(x)
    return (x - x_min) / (denom + eps)


def _pairwise_upper_triangle(x: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    K = x.shape[0]

    dists = []
    for i in range(K - 1):
        # Broadcast difference to all j>i
        diffs = x[i].unsqueeze(0) - x[i + 1 :]
        if p == float("inf"):
            d = diffs.abs().amax(dim=1)
        else:
            d = diffs.abs().pow(p).sum(dim=1).pow(1.0 / p)
        dists.append(d)
    return torch.cat(dists, dim=0) if dists else x.new_zeros((0,))


def _class_centers(features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    classes = torch.unique(labels)
    K = classes.numel()
    D = features.shape[1]
    centers = features.new_zeros((K, D))
    for k_idx, k in enumerate(classes):
        mask = labels == k
        xk = features[mask]
        if xk.numel() == 0:
            raise ValueError(f"No samples for class label {k.item()}")
        centers[k_idx] = xk.mean(dim=0)
    return centers, classes


def _per_class_mean_intra_dists(
    features: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor, classes: torch.Tensor, p: float
) -> torch.Tensor:
    K = classes.numel()
    dks = features.new_zeros((K,))
    for k_idx, k in enumerate(classes):
        mask = labels == k
        xk = features[mask]
        diffs = (xk - centers[k_idx].unsqueeze(0)).abs()
        if p == float("inf"):
            d = diffs.amax(dim=1)
        else:
            d = diffs.pow(p).sum(dim=1).pow(1.0 / p)
        dks[k_idx] = d.mean()
    return dks


def roby_metric(
    x: torch.Tensor,
    pred: torch.Tensor,
    p: float = 2.0,
    metric: list[str] | str = ["fsa", "fsd", "roby"],
    append_to: dict[str, list[float]] = {}
) -> None:
    flatten = torch.nn.Flatten()

    x_in = flatten(x.clone())
    pred_in = pred.argmax(dim=1)

    x_in, pred_in = _validate_inputs(x_in, pred_in)

    device = x_in.device
    centers, classes = _class_centers(x_in, pred_in)

    fsa_per_class = _per_class_mean_intra_dists(x_in, pred_in, centers, classes, p=p)
    fsa_per_class_norm = _minmax_norm(fsa_per_class)
    FSA = 1.0 - fsa_per_class_norm.mean()

    pairwise_center_dists = _pairwise_upper_triangle(centers, p=p)
    pairwise_center_dists_norm = _minmax_norm(pairwise_center_dists) if pairwise_center_dists.numel() > 0 else pairwise_center_dists
    FSD = pairwise_center_dists_norm.mean() if pairwise_center_dists.numel() > 0 else torch.tensor(0.0, device=device)

    K = centers.shape[0]
    if K >= 2:
        roby_pairs = []
        idx_i = []
        idx_j = []

        for i in range(K - 1):
            for j in range(i + 1, K):

                diff = (centers[i] - centers[j]).abs()
                if p == float("inf"):
                    d_ij = diff.amax()
                else:
                    d_ij = diff.pow(p).sum().pow(1.0 / p)
                val = fsa_per_class[i] + fsa_per_class[j] - d_ij
                roby_pairs.append(val)
                idx_i.append(i)
                idx_j.append(j)

        ROBY_pairwise = torch.stack(roby_pairs)
        ROBY_pairwise_norm = _minmax_norm(ROBY_pairwise)
        ROBY = ROBY_pairwise_norm.mean()
    else:
        ROBY_pairwise = x_in.new_zeros((0,))
        ROBY_pairwise_norm = ROBY_pairwise
        ROBY = torch.tensor(0.0, device=device)

    if isinstance(metric, str):
        metric: list[str] = [metric]

    for m in metric:
        m_p= f"{m}_{p}"
        if m_p not in append_to:
            append_to[m_p] = []

        if m == "fsa": 
            append_to[m_p].append(FSA.detach().item())

        if m == "fsd":
            append_to[m_p].append(FSD.detach().item())

        if m == "roby":
            append_to[m_p].append(ROBY.detach().item())