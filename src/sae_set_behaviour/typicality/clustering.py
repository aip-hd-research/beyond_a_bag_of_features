from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_mutual_info_score


def build_word_category_labels(
    words: List[str],
    categories: List[str],
    by_category: Dict[str, List[Dict[str, Any]]],
) -> np.ndarray:
    """Ground-truth category index for each word (categories map to themselves)."""
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    word_to_label: Dict[str, int] = {c: cat_to_idx[c] for c in categories}
    for cat, rows in by_category.items():
        for row in rows:
            word_to_label[row["item"]] = cat_to_idx[cat]
    return np.array([word_to_label[w] for w in words], dtype=int)


def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarities for rows of X."""
    Xn = _l2_normalize_rows(np.asarray(X, dtype=np.float64))
    return np.clip(Xn @ Xn.T, -1.0, 1.0)


def jaccard_similarity_matrix(B: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard similarities for binary (or binarized) feature rows."""
    B = (np.asarray(B) > 0).astype(np.float64)
    inter = B @ B.T
    row_sums = B.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - inter
    union = np.maximum(union, 1.0)
    return inter / union


def similarity_to_distance(sim: np.ndarray) -> np.ndarray:
    """Convert a similarity matrix to a precomputed distance matrix (d = 1 - sim)."""
    D = 1.0 - np.asarray(sim, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def l1_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise L1 (Manhattan) distances for rows of X."""
    return cdist(np.asarray(X, dtype=np.float64), np.asarray(X, dtype=np.float64), metric="cityblock")


def symmetric_kl_distance_matrix(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Pairwise symmetric KL divergence for rows of X, treated as unnormalized
    non-negative distributions (L1-normalized with epsilon smoothing).

    Computed via the cross-entropy identity KL(P||Q) = H(P) - P . logQ so that
    no [N, N, D] tensor is ever materialized: sym_KL[i,j] =
    0.5*(H[i]+H[j]) - 0.5*(P @ logP.T + (P @ logP.T).T)[i,j].
    """
    P = np.asarray(X, dtype=np.float64) + eps
    P = P / P.sum(axis=1, keepdims=True)
    logP = np.log(P)

    H = (P * logP).sum(axis=1)          # [N]
    cross = P @ logP.T                  # [N, N], cross[i,j] = sum_d P[i,d]*logP[j,d]

    D = 0.5 * (H[:, None] + H[None, :]) - 0.5 * (cross + cross.T)
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def symmetric_kl_softmax_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Pairwise symmetric KL divergence for rows of X, softmax-normalized into
    distributions (rather than L1-normalized) — same cross-entropy-identity
    trick as symmetric_kl_distance_matrix, no [N, N, D] tensor materialized.
    Softmax outputs are strictly positive in exact arithmetic, but the
    smallest components can underflow to exactly 0.0 in floating point for
    widely-spread rows (even after the max-subtraction below prevents
    overflow), so they're floored before the log to avoid -inf/NaN.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.max(axis=1, keepdims=True)  # numerical stability (overflow)
    expX = np.exp(X)
    P = expX / expX.sum(axis=1, keepdims=True)
    P = np.maximum(P, 1e-30)  # guard against underflow
    logP = np.log(P)

    H = (P * logP).sum(axis=1)
    cross = P @ logP.T

    D = 0.5 * (H[:, None] + H[None, :]) - 0.5 * (cross + cross.T)
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def dot_product_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Pairwise dot-product 'distance' for rows of X: unbounded similarity
    sim = X @ X.T is converted to a valid non-negative distance via
    D = max(sim) - sim, preserving rank order (higher dot product = smaller distance).
    """
    sim = np.asarray(X, dtype=np.float64) @ np.asarray(X, dtype=np.float64).T
    D = sim.max() - sim
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def _kmedoids_plusplus_init(D: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-medoids++ initialization on a precomputed distance matrix."""
    n = D.shape[0]
    medoids = np.empty(k, dtype=int)
    medoids[0] = int(rng.integers(n))
    closest = D[:, medoids[0]].copy()

    for j in range(1, k):
        probs = closest ** 2
        total = probs.sum()
        if total <= 0:
            medoids[j] = int(rng.choice(n, replace=False, size=1)[0])
        else:
            medoids[j] = int(rng.choice(n, p=probs / total))
        closest = np.minimum(closest, D[:, medoids[j]])
    return medoids


def kmedoids_labels(
    distance: np.ndarray,
    k: int,
    *,
    random_state: int = 0,
    max_iter: int = 300,
) -> np.ndarray:
    """
    k-medoids (PAM-style) on a precomputed pairwise distance matrix.

    Works with any symmetric dissimilarity matrix (cosine distance, 1 - Jaccard, …).
    """
    D = np.asarray(distance, dtype=np.float64)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError(f"distance must be square, got shape {D.shape}")
    if k > n:
        raise ValueError(f"k={k} cannot exceed n_samples={n}")
    if not np.allclose(D, D.T, atol=1e-8):
        raise ValueError("distance must be symmetric")

    rng = np.random.default_rng(random_state)
    medoids = _kmedoids_plusplus_init(D, k, rng)
    labels = np.full(n, -1, dtype=int)

    for _ in range(max_iter):
        dist_to_medoids = D[:, medoids]
        labels_new = np.argmin(dist_to_medoids, axis=1)
        if np.array_equal(labels_new, labels):
            break
        labels = labels_new

        new_medoids = np.empty(k, dtype=int)
        for j in range(k):
            cluster_idx = np.where(labels == j)[0]
            if cluster_idx.size == 0:
                new_medoids[j] = medoids[j]
                continue
            sub = D[np.ix_(cluster_idx, cluster_idx)]
            new_medoids[j] = cluster_idx[int(np.argmin(sub.sum(axis=1)))]

        if np.array_equal(new_medoids, medoids):
            medoids = new_medoids
            break
        medoids = new_medoids

    return labels


def clustering_ami_from_distance(
    distance: np.ndarray,
    true_labels: np.ndarray,
    k: int,
    *,
    random_state: int = 0,
) -> float:
    pred = kmedoids_labels(distance, k, random_state=random_state)
    return float(adjusted_mutual_info_score(true_labels, pred))


def clustering_ami_from_similarity(
    similarity: np.ndarray,
    true_labels: np.ndarray,
    k: int,
    *,
    random_state: int = 0,
) -> float:
    return clustering_ami_from_distance(
        similarity_to_distance(similarity),
        true_labels,
        k,
        random_state=random_state,
    )


def _per_layer_ami_rows(
    *,
    layer_prefix: str,
    layers: List[int],
    layer_similarities: Dict[int, np.ndarray],
    true_labels: np.ndarray,
    k: int,
    n_words: int,
    max_label: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    layer_amis: List[float] = []

    for layer in layers:
        ami = clustering_ami_from_similarity(layer_similarities[layer], true_labels, k)
        layer_amis.append(ami)
        rows.append({
            "representation": f"{layer_prefix}_layer_{layer}",
            "ami": ami,
            "n_words": n_words,
            "n_categories": k,
        })

    rows.append({
        "representation": max_label,
        "ami": max(layer_amis) if layer_amis else float("nan"),
        "n_words": n_words,
        "n_categories": k,
    })
    return rows


def _per_layer_ami_rows_from_distance(
    *,
    layer_prefix: str,
    layers: List[int],
    layer_distances: Dict[int, np.ndarray],
    true_labels: np.ndarray,
    k: int,
    n_words: int,
    max_label: str,
) -> List[Dict[str, Any]]:
    """Twin of _per_layer_ami_rows for metrics that are already distances
    (L1, symmetric KL, dot-product), skipping the similarity->distance step."""
    rows: List[Dict[str, Any]] = []
    layer_amis: List[float] = []

    for layer in layers:
        ami = clustering_ami_from_distance(layer_distances[layer], true_labels, k)
        layer_amis.append(ami)
        rows.append({
            "representation": f"{layer_prefix}_layer_{layer}",
            "ami": ami,
            "n_words": n_words,
            "n_categories": k,
        })

    rows.append({
        "representation": max_label,
        "ami": max(layer_amis) if layer_amis else float("nan"),
        "n_words": n_words,
        "n_categories": k,
    })
    return rows


def compute_clustering_ami_table(
    *,
    words: List[str],
    categories: List[str],
    by_category: Dict[str, List[Dict[str, Any]]],
    layers: List[int],
    avg_residual_by_layer: Dict[int, torch.Tensor],
    static_emb: torch.Tensor,
    union_by_layer: Optional[Dict[int, torch.Tensor]] = None,
    emb_jaccard_union_by_layer: Optional[Dict[int, torch.Tensor]] = None,
    mean_latent_by_layer: Optional[Dict[int, torch.Tensor]] = None,
    emb_mean_latent_by_layer: Optional[Dict[int, torch.Tensor]] = None,
) -> pd.DataFrame:
    """
    k-medoids (k = #categories) on pairwise similarities via distance = 1 - sim.

    Contextual and static embeddings use cosine similarity; optional SAE union
  vectors use Jaccard. Reports per-layer AMI and max over layers for each metric.
    """
    k = len(categories)
    true_labels = build_word_category_labels(words, categories, by_category)
    n_words = len(words)

    rows: List[Dict[str, Any]] = []

    ctx_sims = {
        layer: cosine_similarity_matrix(avg_residual_by_layer[layer].numpy())
        for layer in layers
    }
    rows.extend(_per_layer_ami_rows(
        layer_prefix="contextual",
        layers=layers,
        layer_similarities=ctx_sims,
        true_labels=true_labels,
        k=k,
        n_words=n_words,
        max_label="contextual_max_layer_ami",
    ))

    static_ami = clustering_ami_from_similarity(
        cosine_similarity_matrix(static_emb.numpy()),
        true_labels,
        k,
    )
    rows.append({
        "representation": "static_embedding",
        "ami": static_ami,
        "n_words": n_words,
        "n_categories": k,
    })

    if union_by_layer is not None:
        jac_sims = {
            layer: jaccard_similarity_matrix(union_by_layer[layer].numpy())
            for layer in layers
        }
        rows.extend(_per_layer_ami_rows(
            layer_prefix="jaccard",
            layers=layers,
            layer_similarities=jac_sims,
            true_labels=true_labels,
            k=k,
            n_words=n_words,
            max_label="jaccard_max_layer_ami",
        ))

    if emb_jaccard_union_by_layer is not None:
        emb_jac_layers = sorted(emb_jaccard_union_by_layer.keys())
        emb_jac_sims = {
            layer: jaccard_similarity_matrix(emb_jaccard_union_by_layer[layer].numpy())
            for layer in emb_jac_layers
        }
        rows.extend(_per_layer_ami_rows(
            layer_prefix="emb_jaccard",
            layers=emb_jac_layers,
            layer_similarities=emb_jac_sims,
            true_labels=true_labels,
            k=k,
            n_words=n_words,
            max_label="emb_jaccard_max_layer_ami",
        ))

    for latents, name_prefix in (
        (mean_latent_by_layer, ""),
        (emb_mean_latent_by_layer, "emb_"),
    ):
        if latents is None:
            continue
        metric_layers = layers if name_prefix == "" else sorted(latents.keys())
        arrays = {layer: latents[layer].numpy() for layer in metric_layers}

        for metric_name, distance_fn in (
            ("l1", l1_distance_matrix),
            ("kl", symmetric_kl_distance_matrix),
            ("klsm", symmetric_kl_softmax_distance_matrix),
            ("dot", dot_product_distance_matrix),
        ):
            prefix = f"{name_prefix}{metric_name}"
            rows.extend(_per_layer_ami_rows_from_distance(
                layer_prefix=prefix,
                layers=metric_layers,
                layer_distances={layer: distance_fn(arrays[layer]) for layer in metric_layers},
                true_labels=true_labels,
                k=k,
                n_words=n_words,
                max_label=f"{prefix}_max_layer_ami",
            ))

        saecos_prefix = f"{name_prefix}saecos"
        saecos_sims = {layer: cosine_similarity_matrix(arrays[layer]) for layer in metric_layers}
        rows.extend(_per_layer_ami_rows(
            layer_prefix=saecos_prefix,
            layers=metric_layers,
            layer_similarities=saecos_sims,
            true_labels=true_labels,
            k=k,
            n_words=n_words,
            max_label=f"{saecos_prefix}_max_layer_ami",
        ))

    return pd.DataFrame(rows)


def split_clustering_ami_tables(
    df_clustering_ami: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split combined AMI table into static, contextual, jaccard, and emb_jaccard tables."""
    rep = df_clustering_ami["representation"]

    df_static = df_clustering_ami[rep == "static_embedding"].reset_index(drop=True)
    df_contextual = df_clustering_ami[rep.str.startswith("contextual_")].reset_index(drop=True)
    is_jaccard = rep.str.startswith("jaccard_")
    is_emb_jaccard = rep.str.startswith("emb_jaccard_")
    df_jaccard = df_clustering_ami[is_jaccard & ~is_emb_jaccard].reset_index(drop=True)
    df_emb_jaccard = df_clustering_ami[is_emb_jaccard].reset_index(drop=True)

    return df_static, df_contextual, df_jaccard, df_emb_jaccard
