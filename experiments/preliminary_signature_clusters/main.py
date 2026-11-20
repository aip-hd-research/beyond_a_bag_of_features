from typing import List
from collections import defaultdict
from dataclasses import dataclass

import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
import urllib.error
import urllib.request
from jaxtyping import Float
import torch
import wandb
from datasets import Dataset, load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
import hydra
from tqdm import tqdm
from termcolor import colored
from sae_set_behaviour import metrics
from sae_set_behaviour.util import (
    SetupConfig,
    WandbConfig,
    collect_activations,
    divide_into_batches,
    infer_layer_int,
    init_wandb,
    setup,
)
from enum import Enum
from typing import Union

class Layer(Enum):
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"


class HeatmapColorScale(Enum):
    """Token heatmap: low → high activation intensity."""
    BLACK_RED = "black_red"
    WHITE_RED = "white_red"

@dataclass
class Config(SetupConfig):
    base_save_dir: str = "results/preliminary_signature_clusters"
    layer: Union[int, Layer] = Layer.MIDDLE
    n_samples: int = 30_000
    n_shuffle: int = 500_000
    seed: int = 0
    prefix: str = "Fact: "
    flush_cache: bool = False
    intersection_block_size: int = 0

ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config", config_path="../../configs")
def main(args: Config) -> None:
    dataset = (
        load_dataset("EleutherAI/pile", revision="refs/convert/parquet", data_dir="all/partial-train", split="train", streaming=True)
    )
    # dataset = deduplicate(dataset).rename_column("claim", "text")
    dataset = dataset.select_columns(["text"]).map(add_prefix, fn_kwargs={"prefix": args.prefix})
    buffer = [sample for sample, _ in zip(dataset, tqdm(range(args.n_shuffle)))]
    rng = random.Random(args.seed)
    dataset = Dataset.from_list(rng.sample(buffer, args.n_samples))
    model, saes, dataloader, save_dir = setup(args, dataset)
    if len(saes) == 0:
        raise ValueError("No SAEs were loaded. Check `sae.layers` in your config.")

    args = post_process_args(args)
    selected_sae = saes[args.layer]
    assert args.layer == infer_layer_int(sae_hook_name=selected_sae.cfg.metadata.hook_name, sae_obj=selected_sae)

    cache_key = config_hash(args)

    max_counts, max_heatmaps, sym_max_counts, activations = calculate_pairwise_distances(
        model,
        dataloader,
        selected_sae,
        args.device,
        args.prefix,
        save_dir,
        args.flush_cache,
        args.intersection_block_size,
        cache_key,
    )

    samples = [sample["text"] for sample in dataset]
    samples_name = f"samples_{cache_key}.txt" if cache_key else "samples.txt"
    save(samples, save_dir / Path(samples_name))

    display_results(
        sym_max_counts,
        max_heatmaps,
        samples,
        args.prefix,
        model.tokenizer,
        HeatmapColorScale.BLACK_RED,
    )

def deduplicate(dataset: Dataset) -> Dataset:
    """Works only on Fever dataset."""
    current_id = 0

    def filter_fn(sample):
        nonlocal current_id
        if sample["id"] != current_id:
            current_id = sample["id"]
            return True
        return False

    return dataset.filter(filter_fn)

def add_prefix(sample, prefix):
    sample["text"] = f"{prefix}{sample['text'][:100]}"
    return sample


def display_results(
    max_counts,
    max_heatmaps,
    samples,
    prefix,
    tokenizer,
    color_scale: HeatmapColorScale = HeatmapColorScale.BLACK_RED,
):
    centroid_id = 0
    threshold = 25
    while True:
        cluster_ids, cluster_samples, centroid_sample, cluster_max_counts = get_cluster(
            samples,
            max_counts,
            centroid_id,
            threshold,
        )
        display_cluster(
            cluster_ids,
            cluster_samples,
            centroid_id,
            centroid_sample,
            max_heatmaps,
            tokenizer,
            threshold,
            prefix,
            max_counts,
            cluster_max_counts,
            color_scale,
        )
        centroid_id, threshold, color_scale = get_command(
            centroid_id, cluster_ids, threshold, color_scale
        )


def get_cluster(
    dataset,
    max_counts,
    centroid_id,
    threshold,
):
    centroid = dataset[centroid_id]
    cluster_mask = max_counts[centroid_id] >= threshold
    cluster_ids = cluster_mask.nonzero().squeeze(dim=-1)
    scores = max_counts[centroid_id, cluster_ids]
    sort_order = torch.argsort(scores, descending=True)
    cluster_ids = cluster_ids[sort_order]
    cluster_max_counts = scores[sort_order].tolist()
    cluster_samples = [dataset[idx] for idx in cluster_ids]

    return cluster_ids, cluster_samples, centroid, cluster_max_counts



def _heatmap_row(max_heatmaps, centroid_id, sample_id) -> torch.Tensor:
    """Return the dense `[max_seq_len]` heatmap for a (centroid, sample) pair.

    `max_heatmaps` is stored as a list of sparse 2D tensors (one per
    centroid) to keep CPU memory tractable; this helper performs the
    list-index + sparse-row-select + densify dance.
    """
    return max_heatmaps[int(centroid_id)][int(sample_id)].to_dense()


def display_cluster(
    cluster_ids,
    cluster_samples,
    centroid_id,
    centroid_sample,
    max_heatmaps,
    tokenizer,
    threshold,
    prefix_template,
    max_counts,
    cluster_max_counts,
    color_scale: HeatmapColorScale,
):
    centroid_max = float(max_counts[centroid_id, centroid_id])
    idx_width = len(str(max(len(cluster_samples) - 1, 0)))
    count_width = (
        max((len(f"{c:.3f}") for c in cluster_max_counts), default=0)
    )
    content = "\n".join(
        f"{i:>{idx_width}} (max_count={cluster_max_counts[i]:>{count_width}.3f}): "
        f"{apply_heatmap(sample, _heatmap_row(max_heatmaps, centroid_id, cluster_ids[i]), tokenizer, prefix_template, color_scale)}"
        for i, sample in enumerate(cluster_samples)
    )
    scale_label = "black→red" if color_scale == HeatmapColorScale.BLACK_RED else "white→red"
    print(f"""
Threshold: {threshold}
Heatmap colors: {scale_label} (enter "c" to toggle)
Centroid (max_count={centroid_max:.3f}): {centroid_sample}
Cluster members:
{content}
Please enter either
a) the number of one of the cluster members.
"t=x" where x is the new threshold to be used.
"id=x" where x is the id of a sample from the dataset.
"c" to toggle heatmap colors between black→red and white→red.
""")


def _heatmap_rgb(intensity: float, color_scale: HeatmapColorScale) -> tuple[int, int, int]:
    """intensity in [0, 1]: low activation → high."""
    t = max(0.0, min(1.0, intensity))
    if color_scale == HeatmapColorScale.BLACK_RED:
        return (int(255 * t), 0, 0)
    # white (255,255,255) → red (255,0,0)
    g = int(255 * (1.0 - t))
    return (255, g, g)


def apply_heatmap(
    sample,
    heatmap,
    tokenizer,
    prefix_template="",
    color_scale: HeatmapColorScale = HeatmapColorScale.BLACK_RED,
):
    sample = sample[len(prefix_template):]
    tokens = tokenizer.tokenize(sample)
    max_heat = float(heatmap.max())
    if max_heat <= 0:
        max_heat = 1.0
    colored_sample = colored(prefix_template, "white")
    offset = 0
    for token, heat in zip(tokens, heatmap):
        token_length = len(token)
        t = float(heat) / max_heat
        r, g, b = _heatmap_rgb(t, color_scale)
        colored_sample += colored(sample[offset : offset + token_length], (r, g, b))
        offset += token_length

    return colored_sample.replace("\n", "")


def get_command(sample_id, cluster_ids, threshold, color_scale: HeatmapColorScale):
    user_command = input().strip()
    if user_command == "c":
        next_scale = (
            HeatmapColorScale.WHITE_RED
            if color_scale == HeatmapColorScale.BLACK_RED
            else HeatmapColorScale.BLACK_RED
        )
        return sample_id, threshold, next_scale
    if "t=" in user_command:
        return sample_id, float(user_command[2:]), color_scale
    if "id=" in user_command:
        return int(user_command[3:]), threshold, color_scale
    return cluster_ids[int(user_command)], threshold, color_scale


def calculate_pairwise_distances(
    model,
    dataloader,
    sae,
    device,
    prefix,
    save_dir: Path,
    flush_cache: bool,
    block_size: int = 0,
    cache_key: str = "",
):
    hook_prefix = sae.cfg.metadata.hook_name.replace('.', '_')
    file_prefix = f"{hook_prefix}_{cache_key}" if cache_key else hook_prefix
    max_counts_path = save_dir / Path(f"{file_prefix}_max_counts.pt")
    max_heatmaps_path = save_dir / Path(f"{file_prefix}_max_heatmaps.pt")
    sym_max_counts_path = save_dir / Path(f"{file_prefix}_sym_max_counts.pt")

    activations = collect_activations_cached(
        model, dataloader, sae, device, prefix, save_dir, flush_cache, cache_key
    )

    if (
        max_counts_path.exists()
        and max_heatmaps_path.exists()
        and sym_max_counts_path.exists()
        and not flush_cache
    ):
        max_counts = torch.load(max_counts_path)
        max_heatmaps = torch.load(max_heatmaps_path)
        sym_max_counts = torch.load(sym_max_counts_path)
        return max_counts, max_heatmaps, sym_max_counts, activations

    max_counts, max_heatmaps = calculate_intersections(
        activations.to(device=device), block_size=block_size
    )
    sym_max_counts = (max_counts + max_counts.T) / 2
    torch.save(max_counts, max_counts_path)
    torch.save(max_heatmaps, max_heatmaps_path)
    torch.save(sym_max_counts, sym_max_counts_path)

    return max_counts, max_heatmaps, sym_max_counts, activations


def collect_activations_cached(
    model,
    dataloader,
    sae,
    device,
    prefix,
    save_dir: Path,
    flush_cache: bool,
    cache_key: str = "",
):
    """Collect (and pad) SAE activations over the dataloader, caching the result on disk."""
    hook_prefix = sae.cfg.metadata.hook_name.replace('.', '_')
    file_prefix = f"{hook_prefix}_{cache_key}" if cache_key else hook_prefix
    activations_path = save_dir / Path(f"{file_prefix}_activations.pt")

    if activations_path.exists() and not flush_cache:
        return torch.load(activations_path)

    with torch.no_grad():
        activations = []
        offset = len(model.tokenizer.encode(prefix))
        for batch in tqdm(dataloader):
            _, cache = model.run_with_cache(batch["input_ids"])

            sae.to(device)
            activations.append(
                (
                    (sae.encode(cache[sae.cfg.metadata.hook_name][:, offset:]) > 0)
                    * batch["attention_mask"][:, offset:].unsqueeze(dim=-1).to(device=device)
                ).to_sparse().detach().cpu()
            )
            sae.to("cpu")

        max_length = max(batch.shape[-2] for batch in activations)
        activations = pad_sequences(activations, max_length)

    model.to("cpu")
    sae.to("cpu")

    os.makedirs(activations_path.parent, exist_ok=True)
    torch.save(activations, activations_path)

    return activations

def calculate_intersections(activations, block_size: int = 0):
    """Compute per-sample intersection-based max counts and heatmaps.

    The per-token intersection count against every (sample, position) is
    reformulated as a single sparse matrix-vector multiply. With the
    (sample, position) sparse dims flattened into a single row dim, we have

        counts[i, j] = <activation, activations[i, j]>
                     = (activations_2d @ activation)[i * max_seq_len + j].

    This avoids materialising the `[n_samples, max_seq_len, n_features]`
    elementwise intermediate (the original OOM source) and is substantially
    faster than the broadcast+elementwise version, because PyTorch's
    `torch.sparse.mm` is heavily optimised.

    `block_size` chunks the matmul along the row (sample) dimension to bound
    peak memory for the dense `[block_size, max_seq_len]` output;
    `block_size <= 0` runs the matmul in one shot (recommended unless memory
    is tight).
    """
    n_samples, max_seq_len, n_features = activations.shape
    if block_size <= 0 or block_size > n_samples:
        block_size = n_samples

    # Flatten the leading (sample, position) sparse dims into a single sparse
    # 2D matrix so one matvec produces the full `counts` tensor.
    if activations.is_sparse:
        coo = activations.coalesce()
        sample_idx, pos_idx, feat_idx = coo._indices()
        flat_idx = torch.stack([sample_idx * max_seq_len + pos_idx, feat_idx])
        activations_2d = torch.sparse_coo_tensor(
            flat_idx,
            coo._values().float(),
            (n_samples * max_seq_len, n_features),
        ).coalesce()
    else:
        activations_2d = activations.reshape(
            n_samples * max_seq_len, n_features
        ).float()

    # Precompute the row-index tensors used to slice `activations_2d` per
    # block. With a single block (the common case) we keep the full matrix
    # to skip the `index_select` entirely.
    block_ranges: List[tuple[int, int, torch.Tensor | None]] = []
    if block_size >= n_samples:
        block_ranges.append((0, n_samples, None))
    else:
        for start in range(0, n_samples, block_size):
            end = min(start + block_size, n_samples)
            row_indices = torch.arange(
                start * max_seq_len, end * max_seq_len, device=activations.device
            )
            block_ranges.append((start, end, row_indices))

    max_counts = []
    # Per-centroid heatmaps are very sparse (only positions where `counts`
    # ties the per-sample max contribute). Storing them as a list of sparse
    # 2D tensors keeps the cumulative CPU memory bounded by the actual
    # non-zero count rather than `n_samples * n_samples * max_seq_len`.
    max_heatmaps: List[torch.Tensor] = []

    for sequence in tqdm(activations):
        seq_max_counts: torch.Tensor | None = None
        seq_max_heatmap: torch.Tensor | None = None
        for idx, activation in enumerate(sequence):
            activation_col = activation.to_dense().float().unsqueeze(-1)
            if (activation_col == 0).all():  # first padding token
                break

            token_maxima = torch.empty(
                n_samples, device=activations.device, dtype=torch.float
            )
            token_heatmap = torch.empty(
                n_samples, max_seq_len, device=activations.device, dtype=torch.float
            )
            for start, end, row_indices in block_ranges:
                block_mat = (
                    activations_2d
                    if row_indices is None
                    else torch.index_select(activations_2d, 0, row_indices)
                )
                counts = (
                    torch.sparse.mm(block_mat, activation_col)
                    .squeeze(-1)
                    .view(end - start, max_seq_len)
                )
                maxima_block, _ = get_maxima(counts)
                heatmap_block = (
                    counts  # take the intersection count
                    * (maxima_block.unsqueeze(dim=-1) == counts)  # only where == max
                    * (maxima_block.unsqueeze(dim=-1) > 0)  # and the max is non-zero
                )
                token_maxima[start:end] = maxima_block
                token_heatmap[start:end] = heatmap_block

            if seq_max_counts is None:  # first non-padding token of this sequence
                seq_max_counts = token_maxima
                seq_max_heatmap = token_heatmap
            else:
                seq_max_counts += token_maxima
                seq_max_heatmap += token_heatmap
        seq_max_counts /= idx

        # Sparsify on-device (cheap when most entries are zero) then move to
        # CPU. Keep `max_counts` dense; it's a 2D matrix and typically not
        # sparse.
        max_counts.append(seq_max_counts.detach().cpu())
        max_heatmaps.append(seq_max_heatmap.detach().to_sparse().cpu())

    return torch.stack(max_counts), max_heatmaps


def get_maxima(counts):
    return torch.max(counts, dim=-1, keepdim=False)


def pad_sequences(batches: List[Float[torch.Tensor, "d_batch d_seq d_activations"]], max_len: int):
    padded_batches = []
    for batch in tqdm(batches):
        d_batch, d_seq, d_activations = batch.shape
        padding_shape = (d_batch, max_len - d_seq, d_activations)
        padded_batches.append(torch.concatenate((batch, torch.zeros(padding_shape, device=batch.device, dtype=batch.dtype).to_sparse()), dim=-2))
    return torch.concatenate(padded_batches)


def save(dataset: List[str], path: Path = Path("samples.txt")):
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        f.writelines([d + "\n" for d in dataset])


# Config fields that do not affect the value of cached tensors and should
# therefore be excluded from the cache key (paths, cache flags, parallelism,
# memory-only knobs, etc.).
_CACHE_HASH_EXCLUDED_KEYS: tuple[str, ...] = (
    "base_save_dir",
    "flush_cache",
    "intersection_block_size",
    "num_workers",
)


def config_hash(args: Config, length: int = 10) -> str:
    """Return a short, deterministic hash of the (relevant fields of the) config.

    Used as a component of the cache file names so that changing inputs that
    affect the cached tensors invalidates the cache automatically. Fields
    listed in `_CACHE_HASH_EXCLUDED_KEYS` are skipped because they do not
    influence the cached values.
    """
    container = OmegaConf.to_container(args, resolve=True)
    if isinstance(container, dict):
        for key in _CACHE_HASH_EXCLUDED_KEYS:
            container.pop(key, None)
    payload = json.dumps(container, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def post_process_args(args: Config) -> Config:
    if args.layer == Layer.MIDDLE:
        args.layer = args.sae.layers[len(args.sae.layers) // 2]
    elif args.layer == Layer.EARLY:
        args.layer = args.sae.layers[len(args.sae.layers) // 4]
    elif args.layer == Layer.LATE:
        args.layer = args.sae.layers[3 * len(args.sae.layers) // 4]
    return args

if __name__ == "__main__":
    main()
