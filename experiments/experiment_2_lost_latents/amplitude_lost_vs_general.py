from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import math
import re

import numpy as np
import pandas as pd
import torch
import hydra
import wandb
from omegaconf import DictConfig, MISSING
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from sae_lens import SAE
from datasets import Dataset

from transformer_lens import HookedTransformer

from sae_set_behaviour.util import SetupConfig, WandbConfig, init_wandb, setup
from sae_set_behaviour.lost_latents.generate_figure import plot_amplitude_histogram
from sae_set_behaviour.lost_latents.construct_text_sequence import TextPair, PAPER_TEMPLATE
from sae_set_behaviour.lost_latents.metrics import general_and_lost_amplitude_noun_union, general_and_lost_amplitude_pooled
from sae_set_behaviour.lost_latents.construct_embedding import get_activation_collector, collect_noun_word_activations
from sae_set_behaviour.lost_latents.collect_baseline_information import write_rows_to_csv, summarize_vector
from sae_set_behaviour.lost_latents.statistics import estimate_bin_width_from_pilot, build_histogram_bin_edges
from sae_set_behaviour.lost_latents.dataset_cases import DataCase, get_data_case, build_pairs_for_case, resolve_csv_path


@dataclass
class Config(SetupConfig):
    base_save_dir: str = "results/experiment_2_lost_latents"
    wandb: WandbConfig = MISSING
    data_cases: List[str] = field(default_factory=lambda: ["english", "german", "human_typicality"])
    data_csv: str = ""  # overrides the CSV; only applied when data_cases has exactly one entry
    plot_layers: Optional[List[int]] = None
    k_values: Optional[List[int]] = None  # e.g. [1, 5] to only run those k's (saves compute); default: every k in 1..case.k_max
    setup_name: str = "A_fixed_base"  # which X -> X+Y pairing to use as the "general input" / lost-feature source
    pooling: str = "last_token"
    text_template: str = PAPER_TEMPLATE
    text_prefix: str = ""
    max_length: int = 32
    experiment_batch_size: int = 32
    row_batch_size: Optional[int] = None  # unused here (this script batches via experiment_batch_size); declared only so shared config yamls (e.g. base.yaml) that set it don't break this script's structured config.
    sae_device: str = ""
    bin_pilot_size: int = 10  # number of rows used to estimate a per-layer bin width before the full sweep
    target_bins: int = 75  # aimed-for bin count (50-100) over the pilot's typical (1st-99th pct) range
    max_bins: int = 2000  # safety cap; widens bins rather than dropping outliers if exceeded
    sae_encode_batch_size: int = 64  # bounds peak GPU/CPU memory of the SAE-encode step; independent of experiment_batch_size (which is the LLM forward-pass batch size)
    log_images_to_wandb: bool = True
    log_bin_tables_to_wandb: bool = False  # per-(layer,k) bin tables are already saved as local CSVs; off by default to keep wandb sync light over a full sweep
    log_combined_bin_table_to_wandb: bool = True  # one long-format table per data case (all layers x k, with layer/k columns) instead of one upload per (layer,k)
    log_raw_values_to_wandb: bool = True  # raw general/lost amplitudes (float32 .npz, one per data case) as a wandb artifact, so the data can be re-binned/re-analysed later

ConfigStore.instance().store(name="config", node=Config)


def build_dummy_dataset() -> Dataset:
    return Dataset.from_dict({"text": ["dummy"]})


@torch.no_grad()
def collect_amplitude_values_for_layer(
    model: HookedTransformer,
    sae_by_layer: Dict[int, SAE],
    args: DictConfig,
    layer: int,
    pairs: List[TextPair],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (general_values, lost_values): 1-D float tensors of SAE latent activation
    amplitudes, restricted to the target (noun) tokens.
      general_values: amplitude (on X) of every latent active on X's target token(s) --
                       the "general input" population.
      lost_values:    amplitude (on X) of the subset of those latents that are no longer
                       active on X+Y's target token(s) -- the "lost" population. A subset
                       of general_values by construction.
    """
    device = args.device
    texts_x = [p.base_text for p in pairs]
    texts_xy = [p.composed_text for p in pairs]
    nouns_x = [p.noun_for_base for p in pairs]
    nouns_xy = [p.noun_for_composed for p in pairs]

    sae = sae_by_layer[layer]
    sae_device = str(getattr(args, "sae_device", "") or device)
    sae = sae.to(sae_device)
    sae.eval()
    sae_batch_size = int(getattr(args, "sae_encode_batch_size", 64) or 64)

    try:
        hook_name = sae.cfg.metadata.hook_name

        if args.pooling == "last_token":
            word_acts_x, word_mask_x = collect_noun_word_activations(
                model=model, texts=texts_x, nouns=nouns_x, hook_name=hook_name,
                device=device, max_length=args.max_length, batch_size=args.experiment_batch_size,
            )
            word_acts_xy, word_mask_xy = collect_noun_word_activations(
                model=model, texts=texts_xy, nouns=nouns_xy, hook_name=hook_name,
                device=device, max_length=args.max_length, batch_size=args.experiment_batch_size,
            )
            # word_acts_*/word_mask_* stay on CPU (d_model-sized, cheap); the SAE-encode
            # step below moves+encodes them in `sae_batch_size` chunks so peak GPU/CPU
            # memory scales with the batch size, not with the pair count or d_sae alone.
            general_values, lost_values = general_and_lost_amplitude_noun_union(
                sae, word_acts_x, word_mask_x, word_acts_xy, word_mask_xy,
                sae_device=str(sae.W_enc.device), batch_size=sae_batch_size,
            )
            del word_acts_x, word_mask_x, word_acts_xy, word_mask_xy
        else:
            collect_activations = get_activation_collector(args.pooling)
            activations_x = collect_activations(
                model=model, texts=texts_x, hook_name=hook_name, device=device,
                max_length=args.max_length, batch_size=args.experiment_batch_size,
            ).cpu()
            activations_xy = collect_activations(
                model=model, texts=texts_xy, hook_name=hook_name, device=device,
                max_length=args.max_length, batch_size=args.experiment_batch_size,
            ).cpu()
            general_values, lost_values = general_and_lost_amplitude_pooled(
                sae, activations_x, activations_xy, batch_size=sae_batch_size,
            )
            del activations_x, activations_xy

        return general_values, lost_values
    finally:
        sae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def estimate_layer_bin_width(
    model: HookedTransformer,
    sae_by_layer: Dict[int, SAE],
    args: DictConfig,
    rows: List[Dict[str, str]],
    case: DataCase,
    setup_name: str,
    layer: int,
    pilot_size: int,
    target_bins: int,
) -> float:
    """Cheap pilot pass over the first `pilot_size` rows (k=1) to pick a bin width
    that reflects this layer/SAE's activation scale, reused across all k for this layer."""
    pilot_rows = rows[:max(1, pilot_size)]
    pilot_pairs = build_pairs_for_case(
        rows=pilot_rows, case=case, setup_name=setup_name, k=1,
        base_template=str(args.text_template or ""), base_prefix=str(args.text_prefix or ""),
    )
    if not pilot_pairs:
        return 0.0

    general_values, lost_values = collect_amplitude_values_for_layer(model, sae_by_layer, args, layer, pilot_pairs)
    combined = torch.cat([general_values, lost_values]) if lost_values.numel() else general_values
    return estimate_bin_width_from_pilot(combined, target_bins=target_bins)


@hydra.main(version_base=None, config_name="config", config_path="../../configs")
def main(args: DictConfig) -> None:
    init_wandb(args.wandb, args, job_type="amplitude_lost_vs_general")
    run = wandb.run

    dummy_dataset = build_dummy_dataset()
    model, saes, _, _ = setup(args, dummy_dataset)
    model.eval()

    sae_by_layer: Dict[int, SAE] = {int(layer): sae for layer, sae in zip(args.sae.layers, saes)}

    case_names = list(getattr(args, "data_cases", None) or ["english"])
    explicit_csv = getattr(args, "data_csv", None)
    if explicit_csv and len(case_names) > 1:
        print(f"Ignoring data_csv override {explicit_csv!r}: only applies when data_cases has a single entry")
        explicit_csv = None

    plot_layers: List[int] = [int(x) for x in (list(args.plot_layers) if args.plot_layers else list(args.sae.layers))]
    setup_name = str(args.setup_name)
    requested_k_values = list(args.k_values) if getattr(args, "k_values", None) else None
    bin_pilot_size = int(args.bin_pilot_size)
    target_bins = int(args.target_bins)
    max_bins = int(args.max_bins)

    for case_name in case_names:
        try:
            case = get_data_case(str(case_name))
            csv_path = resolve_csv_path(case, explicit_csv, to_absolute_path)
            rows = case.load_rows(csv_path)

            if requested_k_values is not None:
                k_values = [int(k) for k in requested_k_values if 1 <= int(k) <= case.k_max]
                skipped = sorted(set(int(k) for k in requested_k_values) - set(k_values))
                if skipped:
                    print(f"[{case.name}] skipping requested k_values {skipped} (outside 1..{case.k_max})")
            else:
                k_values = list(range(1, case.k_max + 1))

            if not k_values:
                print(f"[{case.name}] no valid k_values to run, skipping case")
                continue

            # Several SAEs share one release (e.g. all gemma-scope-2 widths/L0s), so the
            # SAE id goes into the path as well -- otherwise they overwrite each other.
            sae_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", str(args.sae.id_template).replace("{}", "L"))
            output_dir = (Path(args.base_save_dir) / args.sae.model.name / args.sae.release / sae_tag / case.name / f"pooling={args.pooling}" / "amplitude_plots")
            output_dir.mkdir(parents=True, exist_ok=True)

            summary_rows: List[Dict[str, Any]] = []
            combined_bin_dfs: List[pd.DataFrame] = []
            raw_values: Dict[str, np.ndarray] = {}

            for layer in plot_layers:
                layer = int(layer)

                bin_width = estimate_layer_bin_width(
                    model=model, sae_by_layer=sae_by_layer, args=args, rows=rows, case=case,
                    setup_name=setup_name, layer=layer, pilot_size=bin_pilot_size, target_bins=target_bins,
                )

                for k in k_values:
                    pairs = build_pairs_for_case(
                        rows=rows, case=case, setup_name=setup_name, k=k,
                        base_template=str(args.text_template or ""), base_prefix=str(args.text_prefix or ""),
                    )
                    if not pairs:
                        continue

                    general_values, lost_values = collect_amplitude_values_for_layer(
                        model=model, sae_by_layer=sae_by_layer, args=args, layer=layer, pairs=pairs,
                    )
                    if general_values.numel() == 0:
                        print(f"[{case.name}] layer={layer} k={k}: no active latents found, skipping")
                        continue

                    layer_bin_width = bin_width
                    if not (layer_bin_width > 0 and math.isfinite(layer_bin_width)):
                        # Pilot couldn't establish a scale (e.g. too few active latents in the
                        # pilot rows) -- fall back to estimating from this k's own full data.
                        fallback_combined = torch.cat([general_values, lost_values]) if lost_values.numel() else general_values
                        layer_bin_width = estimate_bin_width_from_pilot(fallback_combined, target_bins=target_bins)
                    if not (layer_bin_width > 0 and math.isfinite(layer_bin_width)):
                        print(f"[{case.name}] layer={layer} k={k}: degenerate activation values, skipping")
                        continue

                    combined_values = torch.cat([general_values, lost_values]) if lost_values.numel() else general_values
                    edges = build_histogram_bin_edges(combined_values, bin_width=layer_bin_width, max_bins=max_bins)

                    general_counts, _ = np.histogram(general_values.numpy(), bins=edges)
                    if lost_values.numel():
                        lost_counts, _ = np.histogram(lost_values.numpy(), bins=edges)
                    else:
                        lost_counts = np.zeros(len(edges) - 1, dtype=np.int64)

                    general_mean, general_std = summarize_vector(general_values)
                    lost_mean, lost_std = summarize_vector(lost_values)

                    plot_path = output_dir / f"{setup_name}__pooling={args.pooling}__layer={layer}__k={k}__amplitude_hist.png"
                    plot_amplitude_histogram(
                        bin_edges=edges,
                        general_counts=general_counts,
                        lost_counts=lost_counts,
                        general_stats=(general_mean, general_std),
                        lost_stats=(lost_mean, lost_std),
                        title=f"{case.name} / {setup_name} / layer {layer} / k={k}",
                        save_path=plot_path,
                    )

                    bin_df = pd.DataFrame({
                        "bin_left": edges[:-1],
                        "bin_right": edges[1:],
                        "count_general": general_counts,
                        "count_lost": lost_counts,
                        "density_general": general_counts / max(int(general_counts.sum()), 1) / np.diff(edges),
                        "density_lost": lost_counts / max(int(lost_counts.sum()), 1) / np.diff(edges),
                    })
                    bin_csv = plot_path.with_suffix(".csv")
                    bin_df.to_csv(bin_csv, index=False)
                    combined_bin_dfs.append(bin_df.assign(layer=layer, k=k))
                    raw_values[f"layer={layer}__k={k}__general"] = general_values.float().numpy()
                    raw_values[f"layer={layer}__k={k}__lost"] = lost_values.float().numpy()

                    summary_rows.append({
                        "case": case.name,
                        "setup": setup_name,
                        "pooling": str(args.pooling),
                        "layer": layer,
                        "k": k,
                        "bin_width": float(layer_bin_width),
                        "n_bins": int(len(edges) - 1),
                        "n_general": int(general_values.numel()),
                        "mean_general": general_mean,
                        "std_general": general_std,
                        "n_lost": int(lost_values.numel()),
                        "mean_lost": lost_mean,
                        "std_lost": lost_std,
                    })

                    # Per-(layer,k) bin data/plots are always saved locally (CSV + PNG) above.
                    # What goes to wandb is controlled separately since a full sweep can mean
                    # hundreds of (layer,k) combos, and each Image/Table is a separate upload.
                    log_payload: Dict[str, Any] = {
                        f"{case.name}/{args.pooling}/amplitude/{setup_name}/mean_general": general_mean,
                        f"{case.name}/{args.pooling}/amplitude/{setup_name}/std_general": general_std,
                        f"{case.name}/{args.pooling}/amplitude/{setup_name}/mean_lost": lost_mean,
                        f"{case.name}/{args.pooling}/amplitude/{setup_name}/std_lost": lost_std,
                        f"{case.name}/{args.pooling}/amplitude/{setup_name}/layer": layer,
                        f"{case.name}/{args.pooling}/amplitude/{setup_name}/k": k,
                    }
                    if bool(args.log_images_to_wandb):
                        log_payload[f"{case.name}/{args.pooling}/amplitude/{setup_name}/layer={layer}/k={k}/hist_plot"] = wandb.Image(str(plot_path))
                    if bool(args.log_bin_tables_to_wandb):
                        log_payload[f"{case.name}/{args.pooling}/amplitude/{setup_name}/layer={layer}/k={k}/hist_data"] = wandb.Table(dataframe=bin_df)
                    wandb.log(log_payload)
                    print(f"[{case.name}] layer={layer} k={k}: n_general={general_values.numel()} n_lost={lost_values.numel()} bins={len(edges) - 1} -> {plot_path}")

            summary_csv = output_dir / f"amplitude_summary__pooling={args.pooling}.csv"
            write_rows_to_csv(summary_csv, summary_rows)
            print(f"Wrote: {summary_csv}")

            if summary_rows:
                wandb.log({f"{case.name}/{args.pooling}/amplitude_summary_table": wandb.Table(dataframe=pd.DataFrame(summary_rows))})

            if combined_bin_dfs:
                combined_df = pd.concat(combined_bin_dfs, ignore_index=True)
                combined_df = combined_df[["layer", "k"] + [c for c in combined_df.columns if c not in ("layer", "k")]]
                combined_csv = output_dir / f"amplitude_bins_all__pooling={args.pooling}.csv"
                combined_df.to_csv(combined_csv, index=False)
                if bool(getattr(args, "log_combined_bin_table_to_wandb", True)):
                    wandb.log({f"{case.name}/{args.pooling}/amplitude_bins_table": wandb.Table(dataframe=combined_df)})

            if raw_values:
                raw_npz = output_dir / f"amplitude_raw_values__pooling={args.pooling}.npz"
                np.savez_compressed(raw_npz, **raw_values)
                if bool(getattr(args, "log_raw_values_to_wandb", True)):
                    artifact = wandb.Artifact(
                        name=re.sub(r"[^A-Za-z0-9._-]+", "_", f"amplitude_raw__{run.name}__{case.name}__{args.pooling}"),
                        type="amplitude_raw_values",
                        metadata={"case": case.name, "pooling": str(args.pooling), "setup": setup_name,
                                  "model": str(args.sae.model.name), "release": str(args.sae.release),
                                  "id_template": str(args.sae.id_template),
                                  "keys": "layer={L}__k={k}__{general|lost}"},
                    )
                    artifact.add_file(str(raw_npz))
                    run.log_artifact(artifact)
        except Exception as e:
            print(f"[{case_name}] failed, skipping to next data case: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    run.finish()


if __name__ == "__main__":
    main()
