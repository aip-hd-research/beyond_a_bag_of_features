"""
Utility for sampling near-orthogonal feature directions from a SAE decoder matrix.
Intended for use with Gemma-Scope SAE W_dec weights.
"""

import torch


def sample_near_orthogonal_latents(
    W_dec: torch.Tensor,
    N_max: int,
    threshold: float = 0.2,
    seed: int = 0,
) -> torch.Tensor:
    """
    Greedily select up to N_max rows from W_dec that are pairwise near-orthogonal,
    i.e. all pairwise |cos_sim| < threshold.

    Algorithm: shuffle rows randomly, then iterate — add each candidate row iff
    its absolute cosine similarity with every already-selected row is < threshold.

    Args:
        W_dec:     (n_latents, d_model) decoder weight matrix (need not be normalised).
        N_max:     Maximum number of vectors to select.
        threshold: Maximum allowed |cos_sim| between any two selected vectors.
        seed:      Random seed for the initial shuffle.

    Returns:
        (K, d_model) tensor of L2-normalised selected vectors, K <= N_max.

    Raises:
        ValueError if fewer than 2 vectors can be found (threshold too strict).
    """
    normed = W_dec.float() / W_dec.float().norm(dim=1, keepdim=True).clamp(min=1e-8)

    torch.manual_seed(seed)
    perm = torch.randperm(normed.shape[0])
    shuffled = normed[perm]

    selected_rows: list[torch.Tensor] = []
    sel_mat: torch.Tensor | None = None

    for i in range(shuffled.shape[0]):
        if len(selected_rows) >= N_max:
            break
        v = shuffled[i]
        if sel_mat is None or (sel_mat @ v).abs().max().item() < threshold:
            selected_rows.append(v)
            sel_mat = v.unsqueeze(0) if sel_mat is None else torch.cat([sel_mat, v.unsqueeze(0)], dim=0)

    if len(selected_rows) < 2:
        raise ValueError(
            f"Only {len(selected_rows)} near-orthogonal vector(s) found with "
            f"threshold={threshold}. Increase threshold or use a larger W_dec."
        )

    result = torch.stack(selected_rows, dim=0)  # (K, d_model)
    print(
        f"Sampled {result.shape[0]} near-orthogonal latents "
        f"(threshold={threshold}, d_model={result.shape[1]})"
    )
    return result
