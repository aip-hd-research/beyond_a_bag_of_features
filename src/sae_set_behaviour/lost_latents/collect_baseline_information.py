from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import csv
import random
from sae_lens import SAE
import torch
from sae_set_behaviour.lost_latents.construct_text_sequence import TextPair
from sae_set_behaviour.lost_latents.construct_text_sequence import build_pairs_setup_A, build_pairs_setup_B


def summarize_vector(x: torch.Tensor) -> Tuple[float, float]:
    x = x.detach().float().cpu()
    if x.numel() == 0:
        return float("nan"), float("nan")
    return float(x.mean().item()), float(x.std(unbiased=False).item())


def _get_Wenc_dirs(sae: SAE, feat_ids: torch.Tensor) -> torch.Tensor:
    W = sae.W_enc
    d_sae = int(sae.cfg.d_sae)

    if W.shape[0] == d_sae:
        return W[feat_ids, :]
    if W.shape[1] == d_sae:
        return W[:, feat_ids].T
    raise ValueError(f"Unexpected W_enc shape: {tuple(W.shape)} vs d_sae={d_sae}")

def _get_Wdec_dirs(sae: SAE, feat_ids: torch.Tensor) -> torch.Tensor:
    W = sae.W_dec
    d_sae = int(sae.cfg.d_sae)

    if W.shape[0] == d_sae:
        return W[feat_ids, :]
    if W.shape[1] == d_sae:
        return W[:, feat_ids].T
    raise ValueError(f"Unexpected W_dec shape: {tuple(W.shape)} vs d_sae={d_sae}")

def build_random_mismatch_pairs(rows: List[Dict[str, str]], k: int, setup_name: str, seed: int) -> List[TextPair]:
    rng = random.Random(seed)

    if setup_name == "A_fixed_base":
        left_pairs = build_pairs_setup_A(rows, k)
        right_pairs = build_pairs_setup_A(rows, k)
    elif setup_name == "B_stepwise":
        left_pairs = build_pairs_setup_B(rows, k)
        right_pairs = build_pairs_setup_B(rows, k)
    else:
        raise ValueError(f"Unknown setup_name: {setup_name}")

    if not left_pairs or not right_pairs:
        return []

    right_data = [(p.composed_text, p.noun) for p in right_pairs]
    rng.shuffle(right_data)

    return [
        TextPair(
            base_text=lp.base_text,
            composed_text=right_text,
            noun=lp.noun,
            composed_noun=right_noun,
        )
        for lp, (right_text, right_noun) in zip(left_pairs, right_data)
    ]


def write_rows_to_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def summarize_eval(setup_name: str, pooling: str, layer: int, k: int, r: Dict[str, Any]) -> Dict[str, Any]:
    lost_rate = r["lost_rate_per_pair"].float()
    lost_count = r["lost_count_per_pair"].float()
    active_x = r["active_x_per_pair"].float()
    active_xy = r["active_xy_per_pair"].float()

    mean_lr, std_lr = summarize_vector(lost_rate)
    mean_lc, std_lc = summarize_vector(lost_count)
    mean_ax, std_ax = summarize_vector(active_x)
    mean_axy, std_axy = summarize_vector(active_xy)

    return {
        "setup": setup_name,
        "pooling": pooling,
        "layer": int(layer),
        "hook_name": str(r["hook_name"]),
        "k": int(k),
        "n_pairs": int(lost_rate.numel()),
        "mean_active_x": mean_ax,
        "std_active_x": std_ax,
        "mean_active_xy": mean_axy,
        "std_active_xy": std_axy,
        "mean_lost_count": mean_lc,
        "std_lost_count": std_lc,
        "mean_lost_rate": mean_lr,
        "std_lost_rate": std_lr,
    }