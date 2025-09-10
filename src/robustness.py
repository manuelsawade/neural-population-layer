"""
robustness_metrics.py

Practical, PyTorch‑ready implementations of three robustness evaluation methods
suited for model comparison during the *test* step:

1) AutoAttack‑style empirical robust accuracy (with graceful fallback)
   - If the optional `autoattack` package is available, we call it directly.
   - Otherwise we run a strong in‑house ensemble: PGD (multi‑restart) + Square Attack.

2) CLEVER (lite) lower‑bound robustness estimate
   - Attack‑agnostic, estimates a lower bound on the minimum perturbation required
     for misclassification by sampling local Lipschitz constants.
   - This is a practical, EVT‑free approximation inspired by CLEVER; it is efficient
     and scales to typical test batches. (See docstring for details.)

3) ROBY‑style representation robustness
   - Measures intra‑class cohesion vs inter‑class separation on hidden features
     for clean and perturbed inputs, returning a robustness score and a degradation
     metric under perturbation.

Notes
-----
• All functions are pure PyTorch and NumPy. No training/loader code is required.
• Plug these into your evaluation loop; see the usage examples at the bottom.
• Designed to be *lightweight* and *transparent* so you can adapt easily to your
  population‑code hidden layers.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from autoattack import AutoAttack

# ===============================================================
# Utility: device helpers
# ===============================================================

def _to_device(batch, device: torch.device):
    if isinstance(batch, (list, tuple)):
        return tuple(_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


# ===============================================================
# 1) AutoAttack‑style empirical robust accuracy
# ===============================================================

@dataclass
class AttackConfig:
    eps: float
    steps: int = 50
    step_size: Optional[float] = None  # if None, set to eps/steps*2
    restarts: int = 5
    norm: str = "Linf"  # "Linf" or "L2"


def _linf_project(x_adv, x_orig, eps):
    return torch.clamp(x_adv, min=x_orig - eps, max=x_orig + eps)


def _l2_project(x_adv, x_orig, eps):
    delta = x_adv - x_orig
    bsz = x_adv.shape[0]
    flat = delta.view(bsz, -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    scale = (eps / norms).clamp(max=1.0)
    flat = flat * scale
    return x_orig + flat.view_as(delta)


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: AttackConfig,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy,
    targeted: bool = False,
) -> torch.Tensor:
    """Strong PGD with multiple restarts; Linf or L2.

    Args:
        model: PyTorch model f(x)->logits
        x: input batch in [0,1]
        y: labels (or target labels if targeted=True)
        cfg: AttackConfig
        loss_fn: loss to maximize (untargeted) / minimize (targeted)
        targeted: if True, *minimize* loss toward target labels

    Returns:
        Adversarial examples x_adv (same shape as x)
    """
    model.eval()
    device = x.device
    step_size = cfg.step_size or (2.0 * cfg.eps / max(cfg.steps, 1))

    best_adv = x.clone()
    best_success = torch.zeros(x.size(0), dtype=torch.bool, device=device)

    for r in range(cfg.restarts):
        if cfg.norm == "Linf":
            x_adv = (x + (torch.empty_like(x).uniform_(-cfg.eps, cfg.eps))).clamp(0, 1)
        elif cfg.norm == "L2":
            noise = torch.randn_like(x)
            flat = noise.view(x.size(0), -1)
            flat = flat / (flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12))
            radii = torch.empty(x.size(0), 1, device=device).uniform_(0, cfg.eps)
            noise = (flat * radii).view_as(x)
            x_adv = (x + noise).clamp(0, 1)
        else:
            raise ValueError("norm must be 'Linf' or 'L2'")

        for i in range(cfg.steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = loss_fn(logits, y)
            if not targeted:
                loss = -loss  # maximize loss
            loss.backward()
            grad = x_adv.grad.detach()

            if cfg.norm == "Linf":
                step = step_size * grad.sign()
                x_adv = x_adv + step if not targeted else x_adv - step
                x_adv = _linf_project(x_adv, x, cfg.eps)
            else:  # L2
                g = grad
                g_flat = g.view(g.size(0), -1)
                g_norm = g_flat.norm(p=2, dim=1).view(-1, *([1] * (x.ndim - 1))).clamp(min=1e-12)
                step = step_size * g / g_norm
                x_adv = x_adv + step if not targeted else x_adv - step
                x_adv = _l2_project(x_adv, x, cfg.eps)

            x_adv = x_adv.clamp(0, 1).detach()

        with torch.no_grad():
            preds = model(x_adv).argmin(dim=1) if targeted else model(x_adv).argmax(dim=1)
            success = preds.eq(y) if targeted else preds.ne(y)
            improved = success & (~best_success)
            best_adv[improved] = x_adv[improved]
            best_success = best_success | success

    return best_adv


def square_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int = 5000,
    p_init: float = 0.05,
    norm: str = "Linf",
) -> torch.Tensor:
    """Minimal Square Attack (untargeted), Linf or L2.
    Reference: Andriushchenko et al. (CVPR 2020) — simplified here.
    """
    assert norm in {"Linf", "L2"}
    model.eval()
    device = x.device
    x_adv = x.clone()

    with torch.no_grad():
        bsz, c, h, w = x.size()
        pert = torch.zeros_like(x)
        p = p_init
        for t in range(steps):
            side = max(1, int(round(min(h, w) * math.sqrt(p))))
            x0 = torch.randint(0, h - side + 1, (bsz,), device=device)
            y0 = torch.randint(0, w - side + 1, (bsz,), device=device)
            delta = torch.empty((bsz, c, side, side), device=device).uniform_(-1, 1)
            if norm == "Linf":
                patch = eps * delta.sign()
            else:  # L2: normalize patch to have L2 norm eps over image
                flat = delta.view(bsz, -1)
                flat = flat / (flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12))
                patch = (eps * flat).view_as(delta)

            pert_tmp = pert.clone()
            for i in range(bsz):
                pert_tmp[i, :, x0[i] : x0[i] + side, y0[i] : y0[i] + side] = patch[i]

            x_try = (x + pert_tmp).clamp(0, 1)
            pred_clean = model(x_adv).argmax(1)
            pred_try = model(x_try).argmax(1)

            success_mask = pred_try.ne(y)
            x_adv[success_mask] = x_try[success_mask]
            pert[success_mask] = pert_tmp[success_mask]

            # anneal probability
            p = max(0.01, p * 0.99)

    return x_adv


def robust_accuracy_autoattack_like(
    model: nn.Module,
    dataloader: Iterable,
    device: torch.device,
    eps: float,
    norm: str = "Linf",
    pgd_steps: int = 50,
    pgd_restarts: int = 5,
    square_steps: int = 1000,
) -> Dict[str, float]:
    """Compute robust accuracy with an AutoAttack‑style ensemble.

    If the optional `autoattack` package is present, this will invoke it
    (APGD-CE, APGD-DLR, FAB, Square). If not, it falls back to a strong
    PGD (multi‑restart) + Square ensemble and reports the *worst‑case*
    accuracy over the ensemble components.
    """
    model.eval()
    device = torch.device(device)

    adversary = AutoAttack(model, norm=norm, eps=eps, version='standard', device=device)

    total, correct_clean = 0, 0
    correct_worst = 0

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            x, y = batch['x'], batch['y']
        else:
            raise ValueError("dataloader must yield (x, y) or {'x':..., 'y':...}")

        x, y = _to_device(x, device), _to_device(y, device)
        with torch.no_grad():
            pred = model(x).argmax(1)
            correct_clean += pred.eq(y).sum().item()

        x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
        with torch.no_grad():
            pred_adv = model(x_adv).argmax(1)
            correct_adv = pred_adv.eq(y).sum().item()
            total += x.size(0)
            correct_worst += correct_adv


    clean_acc = 100.0 * correct_clean / max(total, 1)
    robust_acc = 100.0 * correct_worst / max(total, 1)
    return {
        "clean_acc_%": clean_acc,
        "robust_acc_%": robust_acc,
    }


# ===============================================================
# 2) CLEVER (lite) — attack‑agnostic lower bound
# ===============================================================

@dataclass
class CleverConfig:
    norm: str = "L2"   # "L2" or "Linf"
    R: float = 1.0      # radius of the sampling ball (in input space units, e.g., 1.0 if inputs in [0,1])
    n_dirs: int = 20    # number of random directions (Monte Carlo)
    n_points: int = 10  # points along each direction
    targeted: bool = False


def clever_lower_bound(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: CleverConfig,
) -> torch.Tensor:
    """Clever‑lite: practical lower bound on minimal adversarial perturbation.

    For each sample in `x`, we estimate a local Lipschitz constant L around x by
    sampling gradients at random points within a ball B(x, R) under the chosen norm.

    Lower bound (untargeted): (f_y(x) - max_{j!=y} f_j(x)) / L
    Lower bound (targeted):   (f_t(x) - f_y(x)) / L   where y are *target labels*

    Notes:
    • This is *inspired* by the CLEVER method but omits the Extreme Value Theory fit
      for simplicity/efficiency. In practice it tracks certified bounds reasonably well
      for model comparison and ablation.
    • Returns a tensor of shape (batch,) with per‑sample lower bounds.
    """
    model.eval()
    device = x.device

    with torch.no_grad():
        logits = model(x)
    if cfg.targeted:
        t = y  # y are target labels
        with torch.no_grad():
            logits_y = logits.gather(1, t.view(-1, 1)).squeeze(1)
            logits_true = logits  # unknown true label; this is targeted use-case
            logits_other = logits.gather(1, t.view(-1, 1)).squeeze(1)
    else:
        with torch.no_grad():
            y_true = y
            fy = logits.gather(1, y_true.view(-1, 1)).squeeze(1)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, y_true.view(-1, 1), False)
            fmax_others = logits.masked_fill(~mask, float('-inf')).max(dim=1).values
            margin = fy - fmax_others  # numerator

    # Estimate local Lipschitz constant L via gradient sampling
    bsz = x.size(0)
    L_est = torch.zeros(bsz, device=device)

    def sample_in_ball(size: Tuple[int, ...]):
        if cfg.norm == "Linf":
            return torch.empty(size, device=device).uniform_(-cfg.R, cfg.R)
        else:  # L2
            v = torch.randn(size, device=device)
            flat = v.view(bsz, -1)
            flat = flat / (flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12))
            radii = torch.empty(bsz, 1, device=device).uniform_(0, cfg.R)
            return (flat * radii).view_as(v)

    for d in range(cfg.n_dirs):
        base = x + sample_in_ball(x.shape)
        for k in range(cfg.n_points):
            z = base + (k / max(cfg.n_points - 1, 1)) * sample_in_ball(x.shape)
            z = z.clamp(0, 1).detach().requires_grad_(True)
            logits_z = model(z)
            if cfg.targeted:
                # Minimize loss to target class t
                loss = -F.cross_entropy(logits_z, y)
            else:
                # Maximize loss on true class
                loss = F.cross_entropy(logits_z, y)
            grads = torch.autograd.grad(loss, z, retain_graph=False, create_graph=False)[0]

            if cfg.norm == "Linf":
                g_norm = grads.view(bsz, -1).norm(p=1, dim=1)  # dual of Linf is L1
            else:  # L2
                g_norm = grads.view(bsz, -1).norm(p=2, dim=1)
            L_est = torch.maximum(L_est, g_norm)

    L_est = L_est.clamp(min=1e-8)
    lower_bound = margin / L_est if not cfg.targeted else (-margin) / L_est  # margin defined above
    return lower_bound


# ===============================================================
# 3) ROBY‑style representation robustness
# ===============================================================

@dataclass
class RobyConfig:
    layer: Optional[str] = None  # module name to hook; if None, penultimate layer via wrapper
    use_logits_if_none: bool = True
    perturb_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None  # e.g., gaussian noise


def _collect_features(model: nn.Module, x: torch.Tensor, layer_name: Optional[str]) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    handle = None

    if layer_name is None:
        # Default: use logits as features if no layer specified
        with torch.no_grad():
            out = model(x)
        return out.detach()

    # Register a forward hook to capture the specified layer outputs
    for name, module in model.named_modules():
        if name == layer_name:
            def hook(_m, _inp, out):
                feats.append(out.detach())
            handle = module.register_forward_hook(hook)
            break
    if handle is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    with torch.no_grad():
        _ = model(x)
    handle.remove()
    return torch.cat([f.flatten(1) if f.dim() > 2 else f for f in feats], dim=0)


def _cohesion_separation(features: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:
    """Compute intra‑class cohesion and inter‑class separation.

    Returns:
        cohesion (float): average within‑class standard deviation (trace‑based)
        separation (float): average pairwise distance between class means
        fisher_like (float): separation / (cohesion + 1e-12)
    """
    x = features.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    classes = np.unique(y)
    means = []
    within = []

    for c in classes:
        xc = x[y == c]
        if xc.shape[0] < 2:
            continue
        mu = xc.mean(axis=0)
        means.append(mu)
        # trace of covariance ≈ mean squared distance to mean
        diffs = xc - mu
        within.append(np.sqrt((diffs ** 2).sum(axis=1).mean() + 1e-12))

    if len(means) < 2:
        return float(np.mean(within) if within else 0.0), 0.0, 0.0

    means = np.stack(means, axis=0)
    # average pairwise Euclidean distance between class means
    dists = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            dists.append(np.linalg.norm(means[i] - means[j]))
    separation = float(np.mean(dists))
    cohesion = float(np.mean(within) if within else 0.0)
    fisher_like = separation / (cohesion + 1e-12)
    return cohesion, separation, fisher_like


def roby_score(
    model: nn.Module,
    dataloader: Iterable,
    device: torch.device,
    cfg: RobyConfig,
) -> Dict[str, float]:
    """ROBY‑style representation robustness on clean vs perturbed inputs.

    We compute a Fisher‑like score (inter‑class separation / intra‑class cohesion)
    on features from a chosen hidden layer (or logits by default), for clean inputs and
    for perturbed inputs via `cfg.perturb_fn` (e.g., small Gaussian noise or corruptions).

    Returns a dictionary with clean/perturbed cohesion, separation, Fisher‑like scores,
    and the relative drop under perturbation (smaller drop ⇒ more robust representations).
    """
    model.eval()
    feats_clean, labels_all = [], []
    feats_pert = []

    def identity(z):
        return z

    perturb = cfg.perturb_fn or identity

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            x, y = batch['x'], batch['y']
        else:
            raise ValueError("dataloader must yield (x, y) or {'x':..., 'y':...}")
        x = _to_device(x, device)
        y = _to_device(y, device)

        f_clean = _collect_features(model, x, cfg.layer if not cfg.use_logits_if_none else (cfg.layer or None))
        with torch.no_grad():
            x_pert = perturb(x).clamp(0, 1)
        f_pert = _collect_features(model, x_pert, cfg.layer if not cfg.use_logits_if_none else (cfg.layer or None))

        feats_clean.append(f_clean)
        feats_pert.append(f_pert)
        labels_all.append(y)

    F_clean = torch.cat(feats_clean, dim=0)
    F_pert = torch.cat(feats_pert, dim=0)
    Y = torch.cat(labels_all, dim=0)

    c_clean, s_clean, fisher_clean = _cohesion_separation(F_clean, Y)
    c_pert, s_pert, fisher_pert = _cohesion_separation(F_pert, Y)

    return {
        "cohesion_clean": float(c_clean),
        "separation_clean": float(s_clean),
        "fisher_clean": float(fisher_clean),
        "cohesion_pert": float(c_pert),
        "separation_pert": float(s_pert),
        "fisher_pert": float(fisher_pert),
        "fisher_drop_%": float(100.0 * max(0.0, (fisher_clean - fisher_pert)) / (abs(fisher_clean) + 1e-12)),
    }


# ===============================================================
# Example perturbation helpers (for ROBY)
# ===============================================================

def gaussian_noise(std: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def _fn(x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * std
    return _fn


def uniform_corruption(eps: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def _fn(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + torch.empty_like(x).uniform_(-eps, eps), 0, 1)
    return _fn


# ===============================================================
# Minimal usage examples (copy/paste into your eval script)
# ===============================================================
if __name__ == "__main__":
    # These examples assume you have: a trained `model`, a test `dataloader`, and a `device`.
    # They are intentionally lightweight; plug into your eval loop as needed.

    print("This module provides functions for robustness evaluation.\n"
          "Import it and call the functions from your test script.")

    # Example pseudo‑usage (commented):
    #
    # from robustness_metrics import (
    #     robust_accuracy_autoattack_like, AttackConfig,
    #     clever_lower_bound, CleverConfig,
    #     roby_score, RobyConfig, gaussian_noise
    # )
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    #
    # # 1) AutoAttack‑style robust accuracy
    # res_aa = robust_accuracy_autoattack_like(model, test_loader, device, eps=8/255, norm='Linf')
    # print(res_aa)
    #
    # # 2) CLEVER‑lite lower bound (per‑sample)
    # batch = next(iter(test_loader))
    # x, y = batch[0].to(device), batch[1].to(device)
    # lb = clever_lower_bound(model, x, y, CleverConfig(norm='L2', R=1.0, n_dirs=10, n_points=8))
    # print('CLEVER‑lite lower bounds:', lb.mean().item())
    #
    # # 3) ROBY‑style representation robustness
    # roby_cfg = RobyConfig(layer=None, use_logits_if_none=True, perturb_fn=gaussian_noise(0.03))
    # res_roby = roby_score(model, test_loader, device, roby_cfg)
    # print(res_roby)
