from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import torch
import hydra
import wandb
import time
from omegaconf import DictConfig, MISSING
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from sae_lens import SAE
from datasets import Dataset

from transformer_lens import HookedTransformer

from sae_set_behaviour.util import SetupConfig, WandbConfig, init_wandb, setup
from sae_set_behaviour.lost_latents.generate_figure import plot_layer_blocks_with_k_colors, plot_against_k, plot_metric_against_layer
from sae_set_behaviour.lost_latents.construct_text_sequence import TextPair, last_non_padding_token_index, PAPER_TEMPLATE
from sae_set_behaviour.lost_latents.metrics import signature_infos, lost_features_rate, lost_features_count, compute_noun_union_signature
from sae_set_behaviour.lost_latents.construct_embedding import get_activation_collector, collect_noun_word_activations
from sae_set_behaviour.lost_latents.collect_baseline_information import write_rows_to_csv, summarize_eval
from sae_set_behaviour.lost_latents.dataset_cases import DataCase, get_data_case, build_pairs_for_case, build_random_mismatch_pairs_for_case, resolve_csv_path
from sae_set_behaviour.lost_latents.logging import _log_plot_and_data, _metric_dicts_to_df

@dataclass
class Config(SetupConfig):
    base_save_dir: str = "results/experiment_2_lost_latents"
    wandb: WandbConfig = MISSING
    data_cases: List[str] = field(default_factory=lambda: ["english", "german", "human_typicality"])
    data_csv: str = ""  # overrides the CSV; only applied when data_cases has exactly one entry
    plot_layers: Optional[List[int]] = None# = field(default_factory=lambda: [2, 7, 12, 17])
    k_values: Optional[List[int]] = None  # e.g. [1, 5] to only run those k's (saves compute); default: every k in 1..case.k_max
    pooling: str = "last_token"
    text_template: str = PAPER_TEMPLATE
    text_prefix: str = ""
    baseline_seed: int = 0
    run_shovel_demo: bool = True
    eps: float = 0.01
    mixture_p_neg: float = 0.6
    max_length: int = 32
    experiment_batch_size: int = 32
    row_batch_size: Optional[int] = None  # unused here (this script batches via experiment_batch_size); declared only so shared config yamls (e.g. base.yaml) that set it don't break this script's structured config.
    sae_device: str = ""

ConfigStore.instance().store(name="config", node=Config)

def build_dummy_dataset() -> Dataset:
    return Dataset.from_dict({"text": ["dummy"]})

@hydra.main(version_base=None, config_name="config", config_path="../../configs")
def main(args: DictConfig) -> None:
    init_wandb(args.wandb, args, job_type="lost_rate_by_input")
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

    plot_layers: List[int] = list(args.plot_layers) if args.plot_layers else list(args.sae.layers)
    plot_layers = [int(x) for x in plot_layers]

    baseline_seed = int(getattr(args, "baseline_seed", 0))
    requested_k_values = list(args.k_values) if getattr(args, "k_values", None) else None

    for case_name in case_names:
        try:
            case = get_data_case(str(case_name))
            csv_path = resolve_csv_path(case, explicit_csv, to_absolute_path)
            rows = case.load_rows(csv_path)

            if requested_k_values is not None:
                case_k_values = [int(k) for k in requested_k_values if 1 <= int(k) <= case.k_max]
                skipped = sorted(set(int(k) for k in requested_k_values) - set(case_k_values))
                if skipped:
                    print(f"[{case.name}] skipping requested k_values {skipped} (outside 1..{case.k_max})")
                if not case_k_values:
                    print(f"[{case.name}] no valid k_values to run, skipping case")
                    continue
            else:
                case_k_values = list(range(1, case.k_max + 1))

            output_dir = (Path(args.base_save_dir)/ args.sae.model.name/ args.sae.release/ case.name/ f"pooling={args.pooling}"/ "plots")
            output_dir.mkdir(parents=True, exist_ok=True)

            all_results_rows: List[Dict[str, Any]] = []
            all_baseline_rows: List[Dict[str, Any]] = []

            for setup_idx, setup_name in enumerate(["A_fixed_base", "B_stepwise"]):
                mean_by_layer_by_k, std_by_layer_by_k, results_rows, baseline_rows = run_setup(
                    model=model,
                    sae_by_layer=sae_by_layer,
                    args=args,
                    rows=rows,
                    case=case,
                    setup_name=setup_name,
                    plot_layers=plot_layers,
                    baseline_seed=baseline_seed,
                    k_values=case_k_values)

                # With k_max=1 (e.g. human_typicality) A_fixed_base and B_stepwise are the
                # same single base->composed comparison (there's no chain to be "fixed" vs
                # "stepwise" about) -- still run both for a uniform pipeline, but only
                # plot/log/save the first to avoid duplicate rows under two setup labels.
                if case.k_max == 1 and setup_idx > 0:
                    continue

                if case.k_max == 1:
                    # Only one comparison per row (no k-dimension to sweep) -- one mean/std
                    # per layer, so plot lost rate vs. layer instead of vs. k.
                    plot_path = output_dir / f"{setup_name}__pooling={args.pooling}__layer_vs_lost_rate.png"
                    plot_metric_against_layer(
                        layers=plot_layers,
                        mean_by_layer={layer: mean_by_layer_by_k[layer][1] for layer in plot_layers},
                        std_by_layer={layer: std_by_layer_by_k[layer][1] for layer in plot_layers},
                        label="lost rate",
                        title=f"{case.name} / {setup_name}",
                        ylabel="lost rate $\mathcal{R}$",
                        save_path=plot_path,
                        show_error_bars=True)
                else:
                    plot_path = output_dir / f"{setup_name}__pooling={args.pooling}__layer_blocks_mean_lost_rate_vs_k.png"
                    k_values = sorted({k for layer in plot_layers for k in mean_by_layer_by_k[layer].keys()})
                    plot_against_k(
                        selected_layers=plot_layers,
                        mean_by_layer_by_k=mean_by_layer_by_k,
                        std_by_layer_by_k=std_by_layer_by_k,
                        title=f"{case.name} / {setup_name}",
                        ylabel="lost rate $\mathcal{R}$",
                        save_path=plot_path,
                        k_values=k_values,
                        show_error_bars=False)
                print(f"save_path: {plot_path}")
                _log_plot_and_data(plot_path=plot_path, mean_by_layer_by_k=mean_by_layer_by_k, std_by_layer_by_k=std_by_layer_by_k, metric_name="lost_rate", mix_mode=f"{case.name}/{args.pooling}", setup_name=setup_name)

                all_results_rows.extend(results_rows)
                all_baseline_rows.extend(baseline_rows)

            results_csv = output_dir / f"results__pooling={args.pooling}.csv"
            baseline_csv = output_dir / f"baseline_random__pooling={args.pooling}.csv"

            write_rows_to_csv(results_csv, all_results_rows)
            write_rows_to_csv(baseline_csv, all_baseline_rows)

            print(f"Wrote: {results_csv}")
            print(f"Wrote: {baseline_csv}")

            wandb.log({
                f"{case.name}/{args.pooling}/results_table": wandb.Table(dataframe=pd.DataFrame(all_results_rows)),
                f"{case.name}/{args.pooling}/baseline_random_table": wandb.Table(dataframe=pd.DataFrame(all_baseline_rows))})
        except Exception as e:
            print(f"[{case_name}] failed, skipping to next data case: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    run.finish()



def run_setup(model: HookedTransformer, sae_by_layer: Dict[int, SAE], args: DictConfig, rows: List[Dict[str, str]], case: DataCase, setup_name: str, plot_layers: List[int], baseline_seed: int, k_values: Optional[List[int]] = None):
    mean_by_layer_by_k: Dict[int, Dict[int, float]] = {int(layer): {} for layer in plot_layers}
    std_by_layer_by_k: Dict[int, Dict[int, float]] = {int(layer): {} for layer in plot_layers}
    results_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    ks = list(k_values) if k_values is not None else list(range(1, case.k_max + 1))

    for layer in plot_layers:
        layer = int(layer)

        text_template = str(getattr(args, "text_template", "") or "")
        text_prefix   = str(getattr(args, "text_prefix",   "") or "")

        for k in ks:
            pairs = build_pairs_for_case(
                rows=rows, case=case, setup_name=setup_name, k=k,
                base_template=text_template, base_prefix=text_prefix,
            )

            if pairs:
                r = run_eval_for_single_layer(model, sae_by_layer, args, layer, pairs)
                row = summarize_eval(setup_name, str(args.pooling), layer, k, r)

                results_rows.append(row)
                mean_by_layer_by_k[layer][k] = float(row["mean_lost_rate"])
                std_by_layer_by_k[layer][k] = float(row["std_lost_rate"])

            baseline_pairs = build_random_mismatch_pairs_for_case(
                rows=rows,
                case=case,
                k=k,
                setup_name=setup_name,
                seed=int(baseline_seed + 10_000 * layer + 100 * k),
                base_template=text_template,
                base_prefix=text_prefix,
            )

            if baseline_pairs:
                br = run_eval_for_single_layer(model, sae_by_layer, args, layer, baseline_pairs)
                brow = summarize_eval(setup_name, str(args.pooling), layer, k, br)
                brow["baseline_type"] = "random_mismatch"
                baseline_rows.append(brow)

    return mean_by_layer_by_k, std_by_layer_by_k, results_rows,baseline_rows   

@torch.no_grad()
def evaluate_set_consistency_for_layers(
    model: HookedTransformer,
    sae_by_layer: Dict[int, SAE],
    args: DictConfig,
    pairs: List[TextPair]) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Returns results keyed by layer index (int), each with:
      - lost_count_per_pair: Tensor[n_pairs]
      - lost_rate_per_pair:  Tensor[n_pairs]
      - active_x_per_pair:   Tensor[n_pairs]
      - active_xy_per_pair:  Tensor[n_pairs]
      - mean_lost_rate:      scalar tensor
    """
    device = args.device

    texts_x  = [p.base_text for p in pairs]
    texts_xy = [p.composed_text for p in pairs]
    nouns_x  = [p.noun_for_base for p in pairs]
    nouns_xy = [p.noun_for_composed for p in pairs]

    results_by_layer: Dict[int, Dict[str, torch.Tensor]] = {}

    for layer in args.sae.layers:
        layer = int(layer)
        sae = sae_by_layer[layer]
        _sae_device = str(getattr(args, "sae_device", "") or device)
        sae = sae.to(_sae_device)
        sae.eval()

        try:
            hook_name = sae.cfg.metadata.hook_name

            sae_dev = str(sae.W_enc.device)
            if args.pooling == "last_token":
                word_acts_x, word_mask_x = collect_noun_word_activations(
                    model=model, texts=texts_x, nouns=nouns_x,
                    hook_name=hook_name, device=device,
                    max_length=args.max_length, batch_size=args.experiment_batch_size,
                )
                signature_x = compute_noun_union_signature(sae, word_acts_x, word_mask_x, sae_device=sae_dev)
                del word_acts_x, word_mask_x

                word_acts_xy, word_mask_xy = collect_noun_word_activations(
                    model=model, texts=texts_xy, nouns=nouns_xy,
                    hook_name=hook_name, device=device,
                    max_length=args.max_length, batch_size=args.experiment_batch_size,
                )
                signature_xy = compute_noun_union_signature(sae, word_acts_xy, word_mask_xy, sae_device=sae_dev)
                del word_acts_xy, word_mask_xy
            else:
                collect_activations = get_activation_collector(args.pooling)
                activations_x = collect_activations(
                    model=model, texts=texts_x, hook_name=hook_name,
                    device=device, max_length=args.max_length, batch_size=args.experiment_batch_size,
                ).to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
                signature_x = signature_infos(sae=sae, x=activations_x, eps=args.eps, compute_preact_masks=False)[0]
                del activations_x

                activations_xy = collect_activations(
                    model=model, texts=texts_xy, hook_name=hook_name,
                    device=device, max_length=args.max_length, batch_size=args.experiment_batch_size,
                ).to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
                signature_xy = signature_infos(sae=sae, x=activations_xy, eps=args.eps, compute_preact_masks=False)[0]
                del activations_xy

            lost_count = lost_features_count(signature_x, signature_xy).detach().cpu()
            lost_rate = lost_features_rate(signature_x, signature_xy).detach().cpu()
            active_x = signature_x.sum(dim=1).to(torch.float32).detach().cpu()
            active_xy = signature_xy.sum(dim=1).to(torch.float32).detach().cpu()
            del signature_x, signature_xy

            results_by_layer[layer] = {
                "lost_count_per_pair": lost_count,
                "lost_rate_per_pair": lost_rate,
                "active_x_per_pair": active_x,
                "active_xy_per_pair": active_xy,
                "mean_lost_rate": lost_rate.float().mean(),
                "hook_name": hook_name,
            }
        finally:
            sae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results_by_layer

def run_eval_for_single_layer(model: HookedTransformer, sae_by_layer: Dict[int, SAE], args: DictConfig, layer: int, pairs: List[TextPair]) -> Dict[str, Any]:
    """
    Runs evaluate_set_consistency_for_layers but restricted to one layer.
    Returns the per-layer result dict (contains tensors + hook_name).
    """
    _layers = list(args.sae.layers)
    args.sae.layers = [int(layer)]
    try:
        results = evaluate_set_consistency_for_layers(model, sae_by_layer, args, pairs)
    finally:
        args.sae.layers = _layers
    return results[int(layer)]

if __name__ == "__main__":
    main()