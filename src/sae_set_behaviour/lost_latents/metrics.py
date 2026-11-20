from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig

from transformer_lens import HookedTransformer
from sae_lens import SAE

from dataclasses import dataclass
from sae_set_behaviour.lost_latents.construct_text_sequence import last_non_padding_token_index


@torch.no_grad()
def compute_noun_union_signature(
    sae: SAE,
    word_acts: torch.Tensor,
    word_mask: torch.Tensor,
    sae_device: str = "cuda",
    encode_batch_size: int = 128,
) -> torch.Tensor:
    """
    Encode all noun-token activations through the SAE and return the union of active
    features over noun tokens (a feature is active if it fires on ANY noun token).

    word_acts: [N, W, d_model] CPU float32
    word_mask: [N, W] int8
    Returns: [N, d_sae] bool, CPU
    """
    N, W, d_model = word_acts.shape
    sae_dtype = sae.W_enc.dtype

    union_batches: List[torch.Tensor] = []
    for i in range(0, N, encode_batch_size):
        acts_b = word_acts[i:i + encode_batch_size]   # [B, W, d_model]
        mask_b = word_mask[i:i + encode_batch_size]   # [B, W]
        B = acts_b.shape[0]

        acts_flat = acts_b.reshape(B * W, d_model).to(sae_device, dtype=sae_dtype)
        binary_flat = (sae.encode(acts_flat) > 0)     # [B*W, d_sae] bool
        binary = binary_flat.view(B, W, -1)            # [B, W, d_sae]

        mask_exp = mask_b.to(sae_device, dtype=torch.bool).unsqueeze(-1)
        union_b = (binary & mask_exp).any(dim=1)       # [B, d_sae] bool

        union_batches.append(union_b.cpu())
        del acts_flat, binary_flat, binary, acts_b, mask_b, mask_exp, union_b

    return torch.cat(union_batches, dim=0)


def _cat_or_empty(chunks: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, dtype=torch.float32)


@torch.no_grad()
def general_and_lost_amplitude_noun_union(
    sae: SAE,
    word_acts_x: torch.Tensor,
    word_mask_x: torch.Tensor,
    word_acts_xy: torch.Tensor,
    word_mask_xy: torch.Tensor,
    sae_device: str = "cuda",
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Amplitude counterpart to compute_noun_union_signature, fused across X and X+Y and
    reduced to active entries within each batch -- so peak memory (GPU and CPU) scales
    with `batch_size`, never with the pair count N or d_sae alone. Encoding stays in the
    SAE's native dtype (typically bf16) until the final (already-small) extraction.

    word_acts_x/xy: [N, W, d_model] CPU float32 (W may differ between x and xy)
    word_mask_x/xy: [N, W] int8
    Returns:
      general_values: 1-D float32 CPU tensor -- amplitude (on X) of every latent
                       active on X's noun-token union, across all pairs.
      lost_values:    1-D float32 CPU tensor -- amplitude (on X) of the subset of
                       those latents no longer active on X+Y's noun-token union.
                       Subset of general_values by construction.
    """
    N = word_acts_x.shape[0]
    sae_dtype = sae.W_enc.dtype

    general_chunks: List[torch.Tensor] = []
    lost_chunks: List[torch.Tensor] = []

    for i in range(0, N, batch_size):
        acts_x_b = word_acts_x[i:i + batch_size]
        mask_x_b = word_mask_x[i:i + batch_size]
        acts_xy_b = word_acts_xy[i:i + batch_size]
        mask_xy_b = word_mask_xy[i:i + batch_size]
        B, Wx, d_model = acts_x_b.shape
        Wxy = acts_xy_b.shape[1]

        encoded_x = sae.encode(acts_x_b.reshape(B * Wx, d_model).to(sae_device, dtype=sae_dtype)).view(B, Wx, -1)
        mask_x_exp = mask_x_b.to(sae_device, dtype=torch.bool).unsqueeze(-1)
        encoded_x.masked_fill_(~mask_x_exp, float("-inf"))
        amp_x_b, _ = encoded_x.max(dim=1)          # [B, d_sae], sae_dtype; -inf where never active
        sig_x_b = amp_x_b > 0                        # [B, d_sae] bool
        del encoded_x, mask_x_exp

        encoded_xy = sae.encode(acts_xy_b.reshape(B * Wxy, d_model).to(sae_device, dtype=sae_dtype)).view(B, Wxy, -1)
        mask_xy_exp = mask_xy_b.to(sae_device, dtype=torch.bool).unsqueeze(-1)
        sig_xy_b = ((encoded_xy > 0) & mask_xy_exp).any(dim=1)  # [B, d_sae] bool
        del encoded_xy, mask_xy_exp

        lost_mask_b = sig_x_b & (~sig_xy_b)

        general_chunks.append(amp_x_b[sig_x_b].float().cpu())
        lost_chunks.append(amp_x_b[lost_mask_b].float().cpu())
        del acts_x_b, mask_x_b, acts_xy_b, mask_xy_b, amp_x_b, sig_x_b, sig_xy_b, lost_mask_b

    return _cat_or_empty(general_chunks), _cat_or_empty(lost_chunks)


@torch.no_grad()
def general_and_lost_amplitude_pooled(
    sae: SAE,
    x: torch.Tensor,
    xy: torch.Tensor,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Amplitude counterpart to signature_infos for single-vector-per-pair pooling (e.g.
    mean_pool), fused across X and X+Y and reduced to active entries within each batch --
    unlike encoding the full [N, d_model] tensor at once, this bounds peak memory by
    `batch_size` rather than the pair count N.

    x, xy: [N, d_model] (any device/dtype; moved onto the SAE's device/dtype internally)
    Returns: (general_values, lost_values) -- see general_and_lost_amplitude_noun_union.
    """
    N = x.shape[0]
    sae_device = sae.W_enc.device
    sae_dtype = sae.W_enc.dtype

    general_chunks: List[torch.Tensor] = []
    lost_chunks: List[torch.Tensor] = []

    for i in range(0, N, batch_size):
        x_b = x[i:i + batch_size].to(sae_device, dtype=sae_dtype)
        xy_b = xy[i:i + batch_size].to(sae_device, dtype=sae_dtype)

        amp_x_b = sae.encode(x_b)          # [B, d_sae], sae_dtype
        sig_x_b = amp_x_b > 0
        sig_xy_b = sae.encode(xy_b) > 0
        lost_mask_b = sig_x_b & (~sig_xy_b)

        general_chunks.append(amp_x_b[sig_x_b].float().cpu())
        lost_chunks.append(amp_x_b[lost_mask_b].float().cpu())
        del x_b, xy_b, amp_x_b, sig_x_b, sig_xy_b, lost_mask_b

    return _cat_or_empty(general_chunks), _cat_or_empty(lost_chunks)


@torch.no_grad()
def signature_infos(*, sae: SAE, x: torch.Tensor, eps: float = 1e-2, compute_preact_masks: bool = False,
    w_norm: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor]]:

    sae_activations = sae.encode(x)  # [batch, d_sae]
    signature_pos = (sae_activations > 0)
    del sae_activations

    if not compute_preact_masks:
        return signature_pos, None, None, None, None, None

    W_enc = sae.W_enc
    if W_enc.device != x.device or W_enc.dtype != x.dtype:
        W_enc = W_enc.to(device=x.device, dtype=x.dtype)

    b_enc = sae.b_enc
    if b_enc.device != x.device or b_enc.dtype != x.dtype:
        b_enc = b_enc.to(device=x.device, dtype=x.dtype)

    if hasattr(sae, "threshold") and getattr(sae, "threshold") is not None:
        threshold = sae.threshold
        if threshold.device != x.device or threshold.dtype != x.dtype:
            threshold = threshold.to(device=x.device, dtype=x.dtype)
        if threshold.ndim == 1:
            threshold = threshold.unsqueeze(0)
    else:
        threshold = torch.full((1, W_enc.shape[1]), float(eps), device=x.device, dtype=x.dtype)

    pre_activations = (x @ W_enc) + b_enc

    preact_pos_mask = (pre_activations > threshold)
    preact_neg_mask = (pre_activations < -threshold)
    preact_active_any_mask = (pre_activations.abs() > threshold)

    n_any = preact_active_any_mask.sum(dim=1).to(torch.float32)
    n_pos = preact_pos_mask.sum(dim=1).to(torch.float32)

    pos_fraction_pre_bias = torch.where(n_any > 0, n_pos / n_any, torch.zeros_like(n_any))

    d_sae = float(pre_activations.shape[1])
    inactive_fraction_pre_bias = 1.0 - (n_any / d_sae)

    return signature_pos, pos_fraction_pre_bias, inactive_fraction_pre_bias, preact_pos_mask, preact_neg_mask, preact_active_any_mask


@torch.no_grad()
def signature_infos_tokenwise(*, sae: SAE, acts: torch.Tensor, eps: float,
) -> Tuple[
    torch.Tensor,  # signature_tok: [B,S,d_sae]
    torch.Tensor,  # preact_pos_tok:   [B,S,d_sae] bool
    torch.Tensor,  # preact_neg_tok:   [B,S,d_sae] bool
    torch.Tensor]: # preact_any_tok:   [B,S,d_sae] bool
    """
    Applies signature_infos tokenwise to activations [B,S,d_in].
    Returns:
      signature_tok: SAE semantic signature (post-bias, binary)
      preact_pos_tok: pre-activation > threshold
      preact_neg_tok: pre-activation < -threshold
      preact_any_tok: abs(pre-activation) > threshold
    """
    B, S, d_in = acts.shape
    flat = acts.reshape(B * S, d_in)
    
    signature_flat, _pos_fraction, _inactive_fraction, preact_pos_flat, preact_neg_flat, preact_any_flat = signature_infos(
        sae=sae,
        x=flat,
        eps=eps,
        compute_preact_masks=True,
    )
    d_sae = signature_flat.shape[1]
    signature_tok = signature_flat.reshape(B, S, d_sae)
    preact_pos_tok = preact_pos_flat.reshape(B, S, d_sae)
    preact_neg_tok = preact_neg_flat.reshape(B, S, d_sae)
    preact_any_tok = preact_any_flat.reshape(B, S, d_sae)

    return signature_tok, preact_pos_tok, preact_neg_tok, preact_any_tok


@torch.no_grad()
def lost_from_signature(*, signature_tok: torch.Tensor, tok_mask: torch.Tensor, ) -> torch.Tensor:
    """lost[j] = union_sig[j] & ~sig_last[j]
    returned as BoolTensor[B,d_sae]"""
    B, S, d_sae = signature_tok.shape
    tok_mask = tok_mask.to(device=signature_tok.device, dtype=torch.bool)
    last_idx = last_non_padding_token_index(tok_mask).to(signature_tok.device)
    batch_idx = torch.arange(B, device=signature_tok.device)

    union_sig = (signature_tok & tok_mask.unsqueeze(-1)).any(dim=1)   # [B,d_sae]
    sig_last = signature_tok[batch_idx, last_idx, :]                  # [B,d_sae]

    return union_sig & (~sig_last)


@torch.no_grad()
def neg_rate_among_lost_last_token(*, signature_tok: torch.Tensor, preact_neg_tok: torch.Tensor, tok_mask: torch.Tensor) -> torch.Tensor:
    """
    Lost is now defined from SAE semantic signature:
      union_sig[j] = OR_s signature_tok[s,j] over non-pad tokens
      sig_last[j]  = signature_tok[last,j]
      lost[j]      = union_sig[j] & ~sig_last[j]
    Diagnostic numerator uses negative SAE pre-activation at the last token:
      neg_rate = (#lost with negative pre-activation at last token) / (#lost)
    """
    B, S, d_sae = signature_tok.shape

    tok_mask = tok_mask.to(device=signature_tok.device, dtype=torch.bool)
    last_idx = last_non_padding_token_index(tok_mask).to(signature_tok.device)
    batch_idx = torch.arange(B, device=signature_tok.device)

    lost = lost_from_signature(signature_tok=signature_tok, tok_mask=tok_mask)  # [B,d_sae]
    preact_neg_last = preact_neg_tok[batch_idx, last_idx, :]                    # [B,d_sae]

    n_lost = lost.sum(dim=1).to(torch.float32)
    n_neg = (lost & preact_neg_last).sum(dim=1).to(torch.float32)

    return torch.where(n_lost > 0, n_neg / n_lost, torch.zeros_like(n_lost))


@torch.no_grad()
def zero_rate_among_lost_last_token(*, signature_tok: torch.Tensor, preact_any_tok: torch.Tensor, tok_mask: torch.Tensor) -> torch.Tensor:
    """
    Lost is defined from SAE semantic signature:
      union_sig[j] = OR_s signature_tok[s,j] over non-pad tokens
      sig_last[j]  = signature_tok[last,j]
      lost[j]      = union_sig[j] & ~sig_last[j]
    Zero-rate among lost uses inactive pre-activation at the last token:
      zero_last[j] = abs(pre_last[j]) <= threshold
                   = ~preact_any_last[j]
      zero_rate = (#lost with zero_last) / (#lost)
    """
    B, S, d_sae = signature_tok.shape

    tok_mask = tok_mask.to(device=signature_tok.device, dtype=torch.bool)
    last_idx = last_non_padding_token_index(tok_mask).to(signature_tok.device)
    batch_idx = torch.arange(B, device=signature_tok.device)

    lost = lost_from_signature(signature_tok=signature_tok, tok_mask=tok_mask)  # [B,d_sae]
    preact_any_last = preact_any_tok[batch_idx, last_idx, :]                    # [B,d_sae]
    zero_last = ~preact_any_last                                                # [B,d_sae]

    n_lost = lost.sum(dim=1).to(torch.float32)
    n_zero = (lost & zero_last).sum(dim=1).to(torch.float32)

    return torch.where(n_lost > 0, n_zero / n_lost, torch.zeros_like(n_lost))

@torch.no_grad()
def feature_set_F_not_in_union_but_neg_last(*, signature_tok: torch.Tensor, preact_neg_tok: torch.Tensor, tok_mask: torch.Tensor) -> torch.Tensor:
    """
    F[j] = (~union_sig[j]) & (negative pre-activation at last token)
    where:
      union_sig uses SAE semantic signature over tokens
      preact_neg_last uses pre-activation < -threshold at the last token
    Returns: BoolTensor[B,d_sae]
    """
    B, S, d_sae = signature_tok.shape

    tok_mask = tok_mask.to(device=signature_tok.device, dtype=torch.bool)
    last_idx = last_non_padding_token_index(tok_mask).to(signature_tok.device)
    batch_idx = torch.arange(B, device=signature_tok.device)

    union_sig = (signature_tok & tok_mask.unsqueeze(-1)).any(dim=1)   # [B,d_sae]
    preact_neg_last = preact_neg_tok[batch_idx, last_idx, :]          # [B,d_sae]
    return (~union_sig) & preact_neg_last

#@torch.no_grad()
#def prev_layer_rates_for_lost(*, prev_preact_pos_tok: torch.Tensor, prev_preact_any_tok: torch.Tensor, tok_mask: torch.Tensor, lost_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#    tok_mask = tok_mask.to(device=prev_preact_pos_tok.device, dtype=torch.bool)
#    lost_mask = lost_mask.to(device=prev_preact_pos_tok.device, dtype=torch.bool)

#    tok_mask4 = tok_mask.unsqueeze(1).unsqueeze(-1)

#    never_pos_prev = ((~prev_preact_pos_tok) | (~tok_mask4)).all(dim=(1, 2))
#    all_zero_prev = ((~prev_preact_any_tok) | (~tok_mask4)).all(dim=(1, 2))

#    n_lost = lost_mask.sum(dim=1).to(torch.float32)
#    never_pos_in_lost = (lost_mask & never_pos_prev).sum(dim=1).to(torch.float32)
#    all_zero_in_lost = (lost_mask & all_zero_prev).sum(dim=1).to(torch.float32)

#    never_pos_rate = torch.where(n_lost > 0, never_pos_in_lost / n_lost, torch.zeros_like(n_lost))
#    all_zero_rate = torch.where(n_lost > 0, all_zero_in_lost / n_lost, torch.zeros_like(n_lost))
#    return never_pos_rate, all_zero_rate

@torch.no_grad()
def prev_layer_masks(*, prev_preact_neg_tok: torch.Tensor, prev_preact_any_tok: torch.Tensor, prev_signature_tok: torch.Tensor, tok_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tok_mask = tok_mask.to(device=prev_signature_tok.device, dtype=torch.bool)

    tok_mask4 = tok_mask.unsqueeze(1).unsqueeze(-1)

    prev_sig_active = (prev_signature_tok & tok_mask4).any(dim=(1, 2))
    always_neg_prev = (prev_preact_neg_tok | (~tok_mask4)).all(dim=(1, 2))
    all_zero_prev = ((~prev_preact_any_tok) | (~tok_mask4)).all(dim=(1, 2))

    return prev_sig_active, always_neg_prev, all_zero_prev


@torch.no_grad()
def grouped_rates_among_lost(*, lost_mask: torch.Tensor, prev_sig_active: torch.Tensor, always_neg_prev: torch.Tensor, all_zero_prev: torch.Tensor, preact_neg_last: torch.Tensor, preact_any_last: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lost_mask = lost_mask.to(device=prev_sig_active.device, dtype=torch.bool)

    prev_active_lost = lost_mask & prev_sig_active
    never_prev_active_lost = lost_mask & (~prev_sig_active)

    inactive_last = ~preact_any_last

    n_lost = lost_mask.sum(dim=1).to(torch.float32)

    n_prev_active_lost = prev_active_lost.sum(dim=1).to(torch.float32)
    n_never_prev_active_lost = never_prev_active_lost.sum(dim=1).to(torch.float32)

    prev_active_among_lost_rate = torch.where(n_lost > 0, n_prev_active_lost / n_lost, torch.zeros_like(n_lost))
    never_prev_active_among_lost_rate = torch.where(n_lost > 0, n_never_prev_active_lost / n_lost, torch.zeros_like(n_lost))

    n_inactive_last_among_prev_active_lost = (prev_active_lost & inactive_last).sum(dim=1).to(torch.float32)
    n_negative_last_among_prev_active_lost = (prev_active_lost & preact_neg_last).sum(dim=1).to(torch.float32)

    inactive_last_among_prev_active_lost_rate = torch.where(
        n_prev_active_lost > 0,
        n_inactive_last_among_prev_active_lost / n_prev_active_lost,
        torch.zeros_like(n_prev_active_lost))
    negative_last_among_prev_active_lost_rate = torch.where(
        n_prev_active_lost > 0,
        n_negative_last_among_prev_active_lost / n_prev_active_lost,
        torch.zeros_like(n_prev_active_lost))

    n_all_zero_among_never_prev_active_lost = (never_prev_active_lost & all_zero_prev).sum(dim=1).to(torch.float32)
    n_always_neg_among_never_prev_active_lost = (never_prev_active_lost & always_neg_prev).sum(dim=1).to(torch.float32)

    always_inactive_among_never_prev_active_lost_rate = torch.where(
        n_never_prev_active_lost > 0,
        n_all_zero_among_never_prev_active_lost / n_never_prev_active_lost,
        torch.zeros_like(n_never_prev_active_lost))
    always_negative_among_never_prev_active_lost_rate = torch.where(
        n_never_prev_active_lost > 0,
        n_always_neg_among_never_prev_active_lost / n_never_prev_active_lost,
        torch.zeros_like(n_never_prev_active_lost))

    return prev_active_among_lost_rate, never_prev_active_among_lost_rate, inactive_last_among_prev_active_lost_rate, negative_last_among_prev_active_lost_rate, always_inactive_among_never_prev_active_lost_rate, always_negative_among_never_prev_active_lost_rate

#@torch.no_grad()
#def prev_sig_active_rate_among_lost(*, prev_signature_tok: torch.Tensor, tok_mask: torch.Tensor, lost_mask: torch.Tensor) -> torch.Tensor:
#    """Among lost features, fraction that were SAE-active at any token position
#    in any previous layer."""
#    tok_mask = tok_mask.to(device=prev_signature_tok.device, dtype=torch.bool)
#    lost_mask = lost_mask.to(device=prev_signature_tok.device, dtype=torch.bool)
#    prev_sig_active = (prev_signature_tok & tok_mask.unsqueeze(1).unsqueeze(-1)).any(dim=(1, 2))   # [B,d_sae]
#    n_lost = lost_mask.sum(dim=1).to(torch.float32)
#    n_prev_sig_active = (lost_mask & prev_sig_active).sum(dim=1).to(torch.float32)
#    return torch.where(n_lost > 0, n_prev_sig_active / n_lost, torch.zeros_like(n_lost))

def lost_features_count(signature_x: torch.Tensor, signature_xy: torch.Tensor) -> torch.Tensor:
    """
    Since signature is binary, we can compute lost features by boolean logic:
    lost = signature_x & ~signature_xy
    """
    lost_per_dim = signature_x & (~signature_xy)
    return lost_per_dim.sum(dim=1).to(torch.float32)


def lost_features_rate(signature_x: torch.Tensor, signature_xy: torch.Tensor) -> torch.Tensor:
    """
    Measures proportion of features active in X but lost in X+Y, normalized by active features in X.
    """
    lost = lost_features_count(signature_x, signature_xy)
    active_x = signature_x.sum(dim=1).to(torch.float32)
    return torch.where(active_x > 0, lost / active_x, torch.zeros_like(lost))




@torch.no_grad()
def cosim_extrema_over_ensemble(dirs: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    dirs: [N, K, D]
    Returns: maxcos_by_k: [N, K] mincos_by_k: [N, K]
      - column 0 (k=1) is NaN
      - for k>=2, extrema are computed over cosine similarities between the
        first sampled direction and the directions 2..k, excluding self-similarity.
    """
    dirs_n = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(eps)
    base = dirs_n[:, 0:1, :]
    cos_all = (dirs_n * base).sum(dim=-1)

    N, K = cos_all.shape
    maxcos_by_k = torch.full((N, K), torch.nan, device=cos_all.device, dtype=cos_all.dtype)
    mincos_by_k = torch.full((N, K), torch.nan, device=cos_all.device, dtype=cos_all.dtype)

    if K >= 2:
        cos_rest = cos_all[:, 1:]
        maxcos_by_k[:, 1:] = torch.cummax(cos_rest, dim=1).values
        mincos_by_k[:, 1:] = torch.cummin(cos_rest, dim=1).values

    return maxcos_by_k, mincos_by_k