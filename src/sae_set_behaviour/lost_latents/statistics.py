import math
from typing import Dict, Tuple
import numpy as np
import torch

@torch.no_grad()
def aggregate_over_k(values_by_k: Dict[int, torch.Tensor]) -> Tuple[float, float, float]:
    """
    Aggregates a dict k -> samples over trials into one layer-level summary.
    Returns:
      mean_over_k:
        arithmetic mean of the per-k means
      propagated_std_of_mean:
        Gaussian error propagation for the mean over k, assuming
        independence of the per-k mean estimates:
            std(mean_k over k) = sqrt(sum_k sigma_k^2) / K
        where sigma_k is the std of the per-k sample distribution
      std_across_k_means:
        standard deviation across the per-k means themselves;
        this quantifies how constant the metric is over k"""
    ks = sorted(values_by_k.keys())
    if len(ks) == 0:
        return 0.0, 0.0, 0.0

    means = []
    stds = []
    for k in ks:
        vec = values_by_k[k]
        means.append(float(vec.mean().item()))
        stds.append(float(vec.std(unbiased=True).item()))

    means_t = torch.tensor(means, dtype=torch.float32)
    stds_t = torch.tensor(stds, dtype=torch.float32)

    K = float(len(ks))
    mean_over_k = float(means_t.mean().item())
    propagated_std_of_mean = float(torch.sqrt((stds_t ** 2).sum()).item() / K)
    std_across_k_means = float(means_t.std(unbiased=True).item())

    return mean_over_k, propagated_std_of_mean, std_across_k_means

@torch.no_grad()
def aggregate_metric_over_k(metric_by_k: Dict[int, float]) -> Tuple[float, float]:
    """Aggregate a per-k metric (already averaged over trials) into: mean over k, std across k"""
    ks = sorted(metric_by_k.keys())
    if len(ks) == 0:
        return 0.0, 0.0
    vals = torch.tensor([float(metric_by_k[k]) for k in ks], dtype=torch.float32)
    mean_over_k = float(vals.mean().item())
    std_across_k = float(vals.std(unbiased=True).item()) if len(vals) > 1 else 0.0
    return mean_over_k, std_across_k


def estimate_bin_width_from_pilot(values: torch.Tensor, target_bins: int = 75, min_width: float = 1e-6) -> float:
    """
    Estimate a histogram bin width from a small pilot sample of values, aiming for
    roughly `target_bins` bins across the pilot's "typical" (1st-99th percentile)
    range. Robust to outliers in the pilot itself, since it uses percentiles rather
    than min/max.

    Returns 0.0 if the pilot is too small/degenerate to estimate a scale from
    (caller should fall back to estimating from the full data in that case).
    """
    values = values.detach().float().cpu()
    if values.numel() < 2:
        return 0.0

    lo, hi = torch.quantile(values, torch.tensor([0.01, 0.99])).tolist()
    spread = hi - lo
    if spread <= 0:
        spread = float((values.max() - values.min()).item())
    if spread <= 0:
        return 0.0

    return max(spread / float(target_bins), min_width)


def build_histogram_bin_edges(values: torch.Tensor, bin_width: float, max_bins: int = 2000) -> np.ndarray:
    """
    Build bin edges of (approximately) fixed width `bin_width` spanning the full
    observed range of `values`. Unlike a fixed bin *count*, this lets the number of
    bins grow to cover outliers instead of clipping/widening the resolution for
    everyone -- capped at `max_bins` as a safety net, at which point the width is
    widened just enough to still cover the full range.
    """
    values = values.detach().float().cpu()
    lo = float(values.min().item())
    hi = float(values.max().item())
    if hi <= lo:
        hi = lo + max(bin_width, 1e-6)

    n_bins = max(1, math.ceil((hi - lo) / bin_width))
    if n_bins > max_bins:
        bin_width = (hi - lo) / max_bins
        n_bins = max_bins

    edges = lo + bin_width * np.arange(n_bins + 1, dtype=np.float64)
    edges[-1] = max(edges[-1], hi)
    return edges