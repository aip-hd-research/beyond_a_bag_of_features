from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import random
import csv
import torch
from transformer_lens import HookedTransformer

@torch.no_grad()
def sample_coeffs(*, trials: int, k_max: int, mode: str, dist_width: float, device: str, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    if mode == "rand_signed":
        return (torch.rand((trials, k_max), device=device, dtype=dtype, generator=generator) - 0.5) * dist_width
    if mode == "rand_pos":
        return torch.rand((trials, k_max), device=device, dtype=dtype, generator=generator) * dist_width
    if mode == "rand_neg":
        return -torch.rand((trials, k_max), device=device, dtype=dtype, generator=generator) * dist_width
    raise ValueError(f"Unknown mixture mode: {mode}")


@torch.no_grad()
def build_embedding_sum_chain(
    *,
    model: HookedTransformer,
    chain_texts: List[str],
    device: str,
    max_length: int = 128,
    eps: float = 1e-8,
    pad_token_id: int = 0) -> List[torch.Tensor]:
    """
    For each text t in chain_texts:
      E_t = token embeddings from hook_embed, shape [seq, d_model]
      v(t) = normalize(sum_{i where token!=PAD} E_t[i])

    Returns:
      List[Tensor[d_model]] aligned with chain_texts order.
    """
    toks = model.to_tokens(chain_texts, truncate=True).to(device)
    toks = toks[:, :max_length] 

    _, cache = model.run_with_cache(toks, names_filter=lambda name: name == "hook_embed")
    E = cache["hook_embed"] 
    mask = (toks != pad_token_id).to(E.dtype)
    V = (E * mask.unsqueeze(-1)).sum(dim=1)

    return [V[i] for i in range(V.shape[0])]

def load_layer_mean_norms(csv_path: Path) -> Dict[int, float]:
    """
    Expects a CSV with at least columns: layer, mean_l2_norm
    """
    out: Dict[int, float] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            layer = int(row["layer"])
            mean = float(row["mean_l2_norm"])
            out[layer] = mean
    return out

@torch.no_grad()
def rescale_to_target_norm(v: torch.Tensor, target: float, eps: float = 1e-8) -> torch.Tensor:
    """
    v: [d] tensor
    returns v scaled so that ||v||_2 == target (approximately).
    """
    denom = v.norm().clamp_min(eps)
    return v * (float(target) / denom)