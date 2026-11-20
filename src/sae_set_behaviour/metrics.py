from collections import defaultdict
from typing import Dict, List, Tuple
from jaxtyping import Bool, Float
from sae_lens import SAE
import torch
import numpy as np
from tqdm import tqdm
from torch.nn.functional import normalize

from sae_set_behaviour.util import divide_into_batches

def iou(a: Bool[torch.Tensor, "... N"], b: Bool[torch.Tensor, "... N"]) -> float:
    a = a.bool() # no-op if already of the correct type
    b = b.bool()
    return ((a * b).sum(dim=-1) / (a + b).sum(dim=-1).clamp_min(1)).mean().detach().cpu().item()

def intersection_size(a: Bool[torch.Tensor, "... N"], b: Bool[torch.Tensor, "... N"]) -> float:
    a = a.bool() # no-op if already of the correct type
    b = b.bool()
    return (a * b).sum(dim=-1).float().mean().detach().cpu().item()

def union_size(a: Bool[torch.Tensor, "... N"], b: Bool[torch.Tensor, "... N"]) -> float:
    a = a.bool() # no-op if already of the correct type
    b = b.bool()
    return (a + b).sum(dim=-1).float().mean().detach().cpu().item()

def l0_norm(a: Float[torch.Tensor, "... N"]) -> Float[torch.Tensor, "..."]:
    return (a != 0).sum(dim=-1).float()

def mean_and_max_cosine_similarity(a: Float[torch.Tensor, "BATCH N"], dtype: torch.dtype = torch.bfloat16, device: str = "cpu", batch_size: int = 32) -> Tuple[float, float]:
    """faster if you need both mean and max cosine similarity"""
    a = a.to(dtype).to(device)
    a = normalize(a, dim=-1)
    batch_means = []
    batch_maxs = []
    for i, batch in tqdm(enumerate(divide_into_batches(a, batch_size)), desc="Computing mean and max cosine similarity"):
        batch = batch.to(dtype).to(device)
        batch_cossims = batch @ a.T
        for j in range(len(batch)):
            batch_cossims[j, i * batch_size + j] = 0
        batch_means.append(torch.mean(batch_cossims).detach().cpu().item())
        batch_maxs.append(torch.max(batch_cossims).detach().cpu().item())
    return np.mean(batch_means), np.mean(batch_maxs)

def calculate_mean_sae_activations(activations: Dict[str, torch.Tensor], saes: List[SAE], device: str = "cpu", batch_size: int = 32, include_inactive_tokens: bool = False) -> Dict[str, Float[torch.Tensor, "sae_hidden_dim"]]:
    sae_activations = defaultdict(list)
    for sae in saes:
        sae.to(device)
        for batch in divide_into_batches(activations[sae.cfg.metadata.hook_name], batch_size):
            batch = batch.to(device)
            if include_inactive_tokens:
                sae_activations[sae.cfg.metadata.hook_name].append(sae.encode(batch).mean(dim=0).detach().cpu())
            else:   
                batch_encoded = sae.encode(batch)
                mean_activation = batch_encoded.sum(dim=0) / (batch_encoded > 0).sum(dim=0).clamp_min(1e-8)
                sae_activations[sae.cfg.metadata.hook_name].append(mean_activation.detach().cpu())
        sae.to("cpu")
    return {k: torch.stack(v).mean(dim=0) for k, v in sae_activations.items()} 


@torch.no_grad()
def most_negative_cosine_partners(
    W: Float[torch.Tensor, "n d"],
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    query_batch_size: int = 1024,
) -> tuple[Float[torch.Tensor, "n"], torch.Tensor]:
    """
    For each row i in W, find j that minimizes cosine_similarity(W[i], W[j]) with j != i.

    Performance notes:
    - Does not materialize the full n x n similarity matrix.
    - Assumes a full similarity *row* fits in memory (i.e., n floats).
    - Uses batched matmul over query rows for throughput.

    Returns:
    - min_vals: shape [n], float32 CPU tensor of the minimum cosine similarity for each i
    - argmins: shape [n], int64 CPU tensor of the argmin partner index j for each i
    """
    Wn = normalize(W.detach().cpu(), dim=-1)
    Wd = Wn.to(device=device, dtype=dtype)

    n = Wd.shape[0]
    min_vals = torch.empty((n,), dtype=torch.float32, device="cpu")
    argmins = torch.empty((n,), dtype=torch.int64, device="cpu")

    for q0 in range(0, n, query_batch_size):
        q1 = min(q0 + query_batch_size, n)
        Q = Wd[q0:q1]  # [q, d]
        sims = (Q @ Wd.T).to(torch.float32)  # [q, n]

        # Exclude self-similarity.
        qi = torch.arange(q1 - q0, device=device)
        jj = torch.arange(q0, q1, device=device)
        sims[qi, jj] = float("inf")

        chunk_min_vals, chunk_argmins = torch.min(sims, dim=1)
        min_vals[q0:q1] = chunk_min_vals.detach().cpu()
        argmins[q0:q1] = chunk_argmins.detach().cpu()

        del Q, sims, chunk_min_vals, chunk_argmins

    return min_vals, argmins