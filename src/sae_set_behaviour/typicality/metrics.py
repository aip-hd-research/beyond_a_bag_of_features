from typing import List

import torch
from scipy.stats import spearmanr
from transformer_lens import HookedTransformer
from typing import Dict, List, Any
import pandas as pd

def spearman(x: List[float], y: List[float]) -> float:
    """Spearman correlation; returns NaN if undefined (e.g. constant vector)."""
    if len(x) < 2:
        return float("nan")
    rho, _p = spearmanr(x, y)
    return float(rho) if rho is not None else float("nan")


def cosine_sim_matrix_to_vector(X: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    Xn = X / (X.norm(dim=1, keepdim=True) + eps)
    yn = y / (y.norm() + eps)
    return Xn @ yn


@torch.no_grad()
def build_static_token_embedding_cache(model: HookedTransformer, texts: List[str], *, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """
    Cache normalized token embeddings from W_E for each unique text.
    Returns:
      cache[text] = Tensor[num_tokens, d_model] on CPU
    """
    W_E = model.W_E
    cache: Dict[str, torch.Tensor] = {}

    for t in texts:
        if t in cache:
            continue

        ids: List[int] = model.tokenizer(t, add_special_tokens=False)["input_ids"]
        if len(ids) == 0:
            cache[t] = torch.zeros((0, W_E.shape[1]), dtype=torch.float32)
            continue

        tok_ids = torch.tensor(ids, device=W_E.device, dtype=torch.long)
        tok_embs = W_E[tok_ids]
        tok_embs = tok_embs / (tok_embs.norm(dim=-1, keepdim=True) + eps)
        cache[t] = tok_embs.detach().float().cpu()

    return cache


@torch.no_grad()
def apt_cosine_static_E_from_cache(
    cache: Dict[str, torch.Tensor],
    u: str,
    v: str,
) -> float:
    """
    apm-style cosine similarity using cached static token embeddings from W_E.

    Let S[t,l] = cos( e_u[t], e_v[l] ), where e_u[t] is the normalized W_E embedding
    of token t in u.

    apt(u -> v) = mean_t max_l S[t,l]
    apt_sym(u,v) = 0.5 * (apt(u -> v) + apt(v -> u))

    Returns NaN if either sequence is empty.
    """
    u_tok = cache[u]
    v_tok = cache[v]

    if u_tok.shape[0] == 0 or v_tok.shape[0] == 0:
        return float("nan")

    S = u_tok @ v_tok.T
    apt_uv = S.max(dim=1).values.mean()
    apt_vu = S.max(dim=0).values.mean()

    return float((0.5 * (apt_uv + apt_vu)).item())


@torch.no_grad()
def apt_cosine_static_E(model: HookedTransformer, u: str, v: str, *, eps: float = 1e-8) -> float:
    """
    Backward-compatible non-cached wrapper.
    """
    cache = build_static_token_embedding_cache(model, [u, v], eps=eps)
    return apt_cosine_static_E_from_cache(cache, u, v)


@torch.no_grad()
def embed_static_E(model: HookedTransformer, texts: List[str], pooling: str = "avg") -> torch.Tensor:
    """
    Static embeddings from the model input embedding matrix W_E.
    For each text: tokenize into token IDs, gather W_E, pool across tokens.
    Returns: [N, d_model] on CPU.
    """
    if pooling not in {"avg", "first", "last"}:
        raise ValueError(f"static_pooling must be one of avg/first/last, got {pooling!r}")

    W_E = model.W_E
    outs: List[torch.Tensor] = []

    for t in texts:
        ids = model.tokenizer(t, add_special_tokens=False)["input_ids"]
        if len(ids) == 0:
            outs.append(torch.zeros(model.cfg.d_model, device=W_E.device))
            continue

        tok_ids = torch.tensor(ids, device=W_E.device, dtype=torch.long)
        tok_embs = W_E[tok_ids]

        if pooling == "avg":
            v = tok_embs.mean(dim=0)
        elif pooling == "first":
            v = tok_embs[0]
        else:
            v = tok_embs[-1]

        outs.append(v)

    return torch.stack(outs, dim=0).detach().float().cpu()



def compute_cosim_table(model: HookedTransformer, by_category: Dict[str, List[Dict[str, Any]]], categories: List[str], items: List[str], static_pooling: str) -> pd.DataFrame:
    """
    Build per-item table with cosine similarity between item and category name.

    static_pooling:
      - "avg" / "first" / "last": cosine between pooled static W_E embeddings
      - "apm_sym": APM-style symmetric cosine between token sequences (no pooling)

    Returns df_item with columns:
      category, item, typicality, subdataset, cosim_item_category
    """
    rows_out: List[Dict[str, Any]] = []

    if static_pooling in {"avg", "first", "last"}:
        cat_emb = embed_static_E(model, categories, pooling=static_pooling)
        item_emb = embed_static_E(model, items, pooling=static_pooling)
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        item_to_idx = {it: i for i, it in enumerate(items)}

        for cat in categories:
            rows = by_category[cat]
            y = cat_emb[cat_to_idx[cat]]
            cat_items = [r["item"] for r in rows]
            X = torch.stack([item_emb[item_to_idx[it]] for it in cat_items], dim=0)
            cosim = cosine_sim_matrix_to_vector(X, y).tolist()

            for r, s in zip(rows, cosim):
                rows_out.append({
                    "category": cat,
                    "item": r["item"],
                    "typicality": float(r["typicality"]),
                    "subdataset": r["subdataset"],
                    "cosim_item_category": float(s),
                })

        return pd.DataFrame(rows_out)

    if static_pooling == "apt_sym":
        all_texts = list(dict.fromkeys(categories + items))
        static_cache = build_static_token_embedding_cache(model, all_texts)

        for cat in categories:
            rows = by_category[cat]
            for r in rows:
                s = apt_cosine_static_E_from_cache(static_cache, r["item"], cat)
                rows_out.append({
                    "category": cat,
                    "item": r["item"],
                    "typicality": float(r["typicality"]),
                    "subdataset": r["subdataset"],
                    "cosim_item_category": float(s),
                })

        return pd.DataFrame(rows_out)

    raise ValueError(f"Unknown static_pooling: {static_pooling}")


def apm_sym_from_binary_activations(A: torch.Tensor, mask_A: torch.Tensor, B: torch.Tensor, mask_B: torch.Tensor, feature_chunk_size: int = 4096) -> torch.Tensor:
    """
    APM_sym for paired sequences in a batch.
    Supports either:
      - A: [N, TA, D], B: [N, TB, D]
      - A: [1, TA, D], B: [N, TB, D]   (broadcast A across the batch)

    mask_A must match the batch dimension of A.
    mask_B must match the batch dimension of B.

    Returns: [N] float
    """
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError(f"A and B must be rank-3, got {A.shape=} and {B.shape=}")
    if mask_A.ndim != 2 or mask_B.ndim != 2:
        raise ValueError(f"mask_A and mask_B must be rank-2, got {mask_A.shape=} and {mask_B.shape=}")
    if A.shape[2] != B.shape[2]:
        raise ValueError(f"Feature dimensions must match, got {A.shape[2]=} and {B.shape[2]=}")

    if A.shape[0] != B.shape[0] and A.shape[0] != 1:
        raise ValueError(f"Batch sizes must either match or A must have batch size 1, got {A.shape[0]=} and {B.shape[0]=}")
    if mask_A.shape[0] != A.shape[0]:
        raise ValueError(f"mask_A batch size must match A, got {mask_A.shape[0]=} and {A.shape[0]=}")
    if mask_B.shape[0] != B.shape[0]:
        raise ValueError(f"mask_B batch size must match B, got {mask_B.shape[0]=} and {B.shape[0]=}")

    N = B.shape[0]
    TA = A.shape[1]
    TB = B.shape[1]
    D = A.shape[2]

    counts = torch.zeros((N, TA, TB), device=B.device, dtype=torch.float32)

    if A.shape[0] == 1:
        for start in range(0, D, feature_chunk_size):
            end = min(start + feature_chunk_size, D)

            A_chunk = A[:, :, start:end].to(torch.float32)
            B_chunk = B[:, :, start:end].to(torch.float32)

            counts += torch.einsum("atd,ild->itl", A_chunk, B_chunk)

            del A_chunk, B_chunk
    else:
        for start in range(0, D, feature_chunk_size):
            end = min(start + feature_chunk_size, D)

            A_chunk = A[:, :, start:end].to(torch.float32)
            B_chunk = B[:, :, start:end].to(torch.float32)

            counts += torch.einsum("itd,ild->itl", A_chunk, B_chunk)

            del A_chunk, B_chunk

    if mask_A.shape[0] == 1:
        mask_A_f = mask_A.to(torch.float32).expand(N, -1)
    else:
        mask_A_f = mask_A.to(torch.float32)
    mask_B_f = mask_B.to(torch.float32)

    counts = counts * mask_B_f.unsqueeze(1)
    max_over_B = counts.max(dim=-1).values
    denom_A = mask_A_f.sum(dim=-1).clamp_min(1.0)
    apm_A_B = (max_over_B * mask_A_f).sum(dim=-1) / denom_A

    counts_rev = counts.transpose(1, 2)
    counts_rev = counts_rev * mask_A_f.unsqueeze(1)
    max_over_A = counts_rev.max(dim=-1).values
    denom_B = mask_B_f.sum(dim=-1).clamp_min(1.0)
    apm_B_A = (max_over_A * mask_B_f).sum(dim=-1) / denom_B

    return 0.5 * (apm_A_B + apm_B_A)