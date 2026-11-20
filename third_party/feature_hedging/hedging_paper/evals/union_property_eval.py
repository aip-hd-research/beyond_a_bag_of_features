import random
from itertools import combinations

import torch
from sae_lens import TrainingSAE

from hedging_paper.toy_models.toy_model import ToyModel


@torch.no_grad()
def evaluate_union_property(
    sae: TrainingSAE,
    toy_model: ToyModel,
    num_features: int,
    active_threshold: float = 0.0,
    max_pairs: int | None = None,
    seed: int = 0,
) -> dict:
    """
    For every pair (i, j) of ground-truth features, construct pure inputs
    x_i, x_j, x_{i+j}, encode them, threshold to binary active sets S_i, S_j,
    S_{i+j}, and measure how well S_{i+j} ≈ S_i ∪ S_j.

    When max_pairs is set and N*(N-1)/2 exceeds it, a random subset of pairs is
    drawn (without replacement, seeded for reproducibility).

    Metrics:
      jaccard   = |S_ij ∩ (S_i ∪ S_j)| / |S_ij ∪ (S_i ∪ S_j)|
      recall    = |S_ij ∩ (S_i ∪ S_j)| / |S_i ∪ S_j|
      precision = |S_ij ∩ (S_i ∪ S_j)| / |S_ij|
      extra     = |S_ij \\ (S_i ∪ S_j)|
      lost      = |(S_i ∪ S_j) \\ S_ij|
    """
    device = next(sae.parameters()).device
    sae.eval()
    toy_model.eval()

    total_pairs = num_features * (num_features - 1) // 2
    if max_pairs is not None and total_pairs > max_pairs:
        rng = random.Random(seed)
        # Sample without enumeration: draw unique (i, j) pairs directly.
        sampled: set[tuple[int, int]] = set()
        while len(sampled) < max_pairs:
            i = rng.randrange(num_features)
            j = rng.randrange(num_features)
            if i != j:
                sampled.add((min(i, j), max(i, j)))
        pairs = list(sampled)
    else:
        pairs = list(combinations(range(num_features), 2))

    pair_metrics = []
    for i, j in pairs:
        e_i = torch.zeros(num_features, device=device)
        e_j = torch.zeros(num_features, device=device)
        e_i[i] = 1.0
        e_j[j] = 1.0
        e_ij = e_i + e_j

        acts_i  = toy_model(e_i.unsqueeze(0))
        acts_j  = toy_model(e_j.unsqueeze(0))
        acts_ij = toy_model(e_ij.unsqueeze(0))

        S_i  = (sae.encode(acts_i)  > active_threshold).squeeze()
        S_j  = (sae.encode(acts_j)  > active_threshold).squeeze()
        S_ij = (sae.encode(acts_ij) > active_threshold).squeeze()

        union = S_i | S_j

        intersection  = (S_ij & union).float().sum()
        jaccard_denom = (S_ij | union).float().sum().clamp(min=1)
        union_size    = union.float().sum().clamp(min=1)
        S_ij_size     = S_ij.float().sum().clamp(min=1)

        pair_metrics.append({
            "jaccard":   (intersection / jaccard_denom).item(),
            "recall":    (intersection / union_size).item(),
            "precision": (intersection / S_ij_size).item(),
            "extra":     (S_ij & ~union).float().sum().item(),
            "lost":      (union & ~S_ij).float().sum().item(),
        })

    n = len(pair_metrics)
    agg: dict = {}
    for k in pair_metrics[0]:
        vals = [m[k] for m in pair_metrics]
        mean_val = sum(vals) / n
        agg[k] = mean_val
        var_val = sum((v - mean_val) ** 2 for v in vals) / max(n - 1, 1)
        agg[f"{k}_std"] = var_val ** 0.5
    agg["num_pairs"] = n
    return agg
