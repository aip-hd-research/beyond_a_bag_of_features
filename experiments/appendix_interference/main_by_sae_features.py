from __future__ import annotations
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence

import gc
import hydra
import torch
import wandb
from datasets import Dataset
from omegaconf import DictConfig, MISSING
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_set_behaviour.util import SetupConfig, WandbConfig, init_wandb, setup, build_dummy_dataset
from sae_set_behaviour.lost_latents.metrics import cosim_extrema_over_ensemble
from sae_set_behaviour.lost_latents.generate_figure import plot_against_k
from sae_set_behaviour.lost_latents.construct_text_sequence import build_chain_texts, read_noun_attribute_rows, last_non_padding_token_index, PAPER_TEMPLATE
from sae_set_behaviour.lost_latents.collect_baseline_information import _get_Wdec_dirs
from sae_set_behaviour.lost_latents.feature_sampling import sample_coeffs
from sae_set_behaviour.lost_latents.construct_embedding import tokenize_texts, collect_noun_word_activations_multi_cpu
from sae_set_behaviour.lost_latents.logging import _log_plot_and_data


@dataclass
class Config(SetupConfig):
    base_save_dir: str = "results/appendix_interference"
    wandb: WandbConfig = MISSING
    data_csv: str = "data/data.csv"

    plot_layers: Optional[List[int]] = None#field(default_factory=lambda: [2, 7, 12, 17])

    text_template: str = PAPER_TEMPLATE
    text_prefix: str = ""
    max_length: int = 64
    eps: float = 0.01
    baseline_seed: int = 0

    mixture_trials: int = 50#100d
    mixture_k_max: int = 2000
    dist_width: float = 10.0
    mixture_modes: List[str] = field(default_factory=lambda: ["rand_signed", "rand_pos", "rand_neg"])

    synthetic_row_batch_size: int = 2
    synthetic_k_chunk_size: int = 16
    sae_eval_batch_size: int = 128
    activation_batch_size: int = 32
    eval_every_k: int = 10

ConfigStore.instance().store(name="config", node=Config)




def _prepare_rows(
    rows: Sequence[dict],
    k_max: int,
    text_prefix: str = "",
    text_template: str = "",
) -> Tuple[List[Tuple[str, List[str], List[str]]], List[str], List[str]]:
    prepared_rows: List[Tuple[str, List[str], List[str]]] = []
    base_texts: List[str] = []
    base_nouns: List[str] = []

    for row in rows:
        noun = row["noun"]
        attrs = [row.get(f"attr{i}", "") for i in range(1, k_max + 1)]
        chain_texts = build_chain_texts(noun, attrs)[: (k_max + 1)]

        if text_template and "{sequence}" in text_template:
            chain_texts = [text_template.format(sequence=t) for t in chain_texts]
        elif text_prefix:
            chain_texts = [f"{text_prefix}{t}" for t in chain_texts]

        prepared_rows.append((noun, attrs, chain_texts))
        base_texts.append(chain_texts[0])
        base_nouns.append(noun)

    return prepared_rows, base_texts, base_nouns


def _init_stat_tensors(plot_layers: List[int], k_max: int, device: Optional[str] = None) -> Tuple[
    Dict[int, torch.Tensor],
    Dict[int, torch.Tensor],
    Dict[int, torch.Tensor]]:
    kwargs = {"dtype": torch.float64}
    if device is not None:
        kwargs["device"] = device
    sums = {int(layer): torch.zeros(k_max + 1, **kwargs) for layer in plot_layers}
    sumsq = {int(layer): torch.zeros(k_max + 1, **kwargs) for layer in plot_layers}
    count = {int(layer): torch.zeros(k_max + 1, **kwargs) for layer in plot_layers}
    return sums, sumsq, count



@torch.no_grad()
def scale_to_norm_batched(x: torch.Tensor, ref_norms: torch.Tensor) -> torch.Tensor:
    denom = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = ref_norms.view(-1, 1, 1) / denom
    mask = (ref_norms > 0).view(-1, 1, 1)
    return torch.where(mask, x * scale, x)


@torch.no_grad()
def signature_infos_fast(*, sae: SAE, x: torch.Tensor) -> torch.Tensor:
    sae_activations = sae.encode(x)
    signature_pos = (sae_activations > 0)
    del sae_activations
    return signature_pos


@torch.no_grad()
def lost_features_rate_bool(signature_x: torch.Tensor, signature_xy: torch.Tensor) -> torch.Tensor:
    lost = signature_x & (~signature_xy)
    active_x = signature_x.sum(dim=1).to(torch.float32)
    lost_count = lost.sum(dim=1).to(torch.float32)
    return torch.where(active_x > 0, lost_count / active_x, torch.zeros_like(lost_count))





@torch.no_grad()
def eval_comp_flat_against_base_microbatched(*, sae: SAE, comp_flat: torch.Tensor, base_sig_single: torch.Tensor, flat_base_row_ids: torch.Tensor, sae_eval_batch_size: int) -> torch.Tensor:
    n_total = comp_flat.shape[0]

    lost_out = torch.empty(n_total, device=comp_flat.device, dtype=torch.float32)

    for start in range(0, n_total, sae_eval_batch_size):
        end = min(start + sae_eval_batch_size, n_total)

        x_mb = comp_flat[start:end]
        base_ids_mb = flat_base_row_ids[start:end]
        base_sig_mb = base_sig_single.index_select(0, base_ids_mb)

        sig_comp_mb = signature_infos_fast(sae=sae, x=x_mb)
        lost_mb = lost_features_rate_bool(base_sig_mb, sig_comp_mb)

        lost_out[start:end] = lost_mb

        del x_mb, base_ids_mb, base_sig_mb, sig_comp_mb, lost_mb

    return lost_out


@torch.no_grad()
def reduce_chunk_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = x.to(torch.float64)
    sum_by_c = y.sum(dim=(0, 1))
    sumsq_by_c = (y * y).sum(dim=(0, 1))
    count_by_c = torch.full((x.shape[2],), float(x.shape[0] * x.shape[1]), dtype=torch.float64, device=x.device)
    return sum_by_c, sumsq_by_c, count_by_c


def update_stat_block(sums: torch.Tensor, sumsq: torch.Tensor, count: torch.Tensor, k_start: int, sum_by_c: torch.Tensor, sumsq_by_c: torch.Tensor, count_by_c: torch.Tensor) -> None:
    k_end = k_start + sum_by_c.shape[0]
    sums[k_start:k_end] += sum_by_c
    sumsq[k_start:k_end] += sumsq_by_c
    count[k_start:k_end] += count_by_c


def finalize_stats(sums: Dict[int, torch.Tensor], sumsq: Dict[int, torch.Tensor], count: Dict[int, torch.Tensor], plot_layers: List[int], k_max: int, k_min_keep: int = 2) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
    mean_dict: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}
    std_dict: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}

    for layer in plot_layers:
        for k in range(k_min_keep, k_max + 1):
            n = float(count[layer][k].item())
            if n <= 0:
                continue

            mean = float((sums[layer][k] / count[layer][k]).item())
            if n <= 1:
                std = 0.0
            else:
                var = (sumsq[layer][k] - (sums[layer][k] ** 2) / count[layer][k]) / (count[layer][k] - 1.0)
                std = float(torch.sqrt(torch.clamp(var, min=0.0)).item())

            mean_dict[layer][k] = mean
            std_dict[layer][k] = std

    return mean_dict, std_dict


@hydra.main(version_base=None, config_name="config", config_path="../../configs")
def main(args: DictConfig) -> None:
    init_wandb(args.wandb, args)
    run = wandb.run

    device = str(args.device)
    sae_device = str(getattr(args, "sae_device", "") or args.device)

    plot_layers_cfg = getattr(args, "plot_layers", None)
    plot_layers: List[int] = list(plot_layers_cfg) if plot_layers_cfg is not None else list(args.sae.layers)
    plot_layers = [int(x) for x in plot_layers]

    mix_modes: List[str] = list(getattr(args, "mixture_modes", ["rand_signed", "rand_pos", "rand_neg"]))
    mix_modes = [m for m in mix_modes if m in {"rand_signed", "rand_pos", "rand_neg"}]

    trials = int(getattr(args, "mixture_trials", 100))
    k_max = int(getattr(args, "mixture_k_max", 300))
    seed0 = int(getattr(args, "baseline_seed", 0))
    dist_width = float(getattr(args, "dist_width", 10.0))
    max_length = int(getattr(args, "max_length", 64))
    eps = float(getattr(args, "eps", 1e-2))
    row_batch_size = int(getattr(args, "synthetic_row_batch_size", 2))
    k_chunk_size = int(getattr(args, "synthetic_k_chunk_size", 16))
    sae_eval_batch_size = int(getattr(args, "sae_eval_batch_size", 128))
    activation_batch_size = int(getattr(args, "activation_batch_size", 32))
    text_template = str(getattr(args, "text_template", "") or "")
    text_prefix   = str(getattr(args, "text_prefix",   "") or "")
    eval_every_k  = int(getattr(args, "eval_every_k", 1))
    if eval_every_k <= 0:
        raise ValueError(f"eval_every_k must be >= 1, got {eval_every_k}")

    eval_k_values = [1]
    eval_k_values.extend(range(1 + eval_every_k, k_max + 1, eval_every_k))
    if eval_k_values[-1] != k_max:
        eval_k_values.append(k_max)
    eval_k_values = sorted(set(eval_k_values))

    dummy_dataset = build_dummy_dataset()
    _layers = list(args.sae.layers)
    args.sae.layers = plot_layers
    try:
        model, saes, _, _ = setup(args, dummy_dataset)
    finally:
        args.sae.layers = _layers
    model.eval()

    sae_by_layer: Dict[int, SAE] = {}
    for layer, sae in zip(plot_layers, saes):
        sae_by_layer[int(layer)] = sae.to("cpu")

    del saes
    gc.collect()
    torch.cuda.empty_cache()

    hook_names_by_layer: Dict[int, str] = {layer: str(sae_by_layer[layer].cfg.metadata.hook_name) for layer in plot_layers}

    csv_path = Path(to_absolute_path(str(args.data_csv)))
    rows = read_noun_attribute_rows(csv_path)
    prepared_rows, base_texts, base_nouns = _prepare_rows(
        rows, k_max=k_max, text_prefix=text_prefix, text_template=text_template,
    )

    base_output_dir = Path(args.base_save_dir) / str(args.sae.model.name) / str(args.sae.release) / "decoder_mixtures_tensorized"

    lost_sums_by_mode: Dict[str, Dict[int, torch.Tensor]] = {}
    lost_sumsq_by_mode: Dict[str, Dict[int, torch.Tensor]] = {}
    lost_count_by_mode: Dict[str, Dict[int, torch.Tensor]] = {}

    #maxcos_sums, maxcos_sumsq, maxcos_count = _init_stat_tensors(plot_layers, k_max, device=sae_device)
    #mincos_sums, mincos_sumsq, mincos_count = _init_stat_tensors(plot_layers, k_max, device=sae_device)

    for mode in mix_modes:
        lost_sums_by_mode[mode], lost_sumsq_by_mode[mode], lost_count_by_mode[mode] = _init_stat_tensors(plot_layers, k_max, device=sae_device)

    base_word_acts_by_layer, base_word_mask = collect_noun_word_activations_multi_cpu(
        model=model,
        texts=base_texts,
        nouns=base_nouns,
        hook_names_by_layer=hook_names_by_layer,
        device=device,
        max_length=max_length,
        batch_size=activation_batch_size)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    for layer in plot_layers:
        print(layer)
        sae = sae_by_layer[layer]
        sae = sae.to(sae_device)
        sae.eval()

        base_word_acts_cpu = base_word_acts_by_layer[layer]  # [N, W, d_model]

        num_rows = len(prepared_rows)
        d_sae = int(sae.cfg.d_sae)

        for row_start in range(0, num_rows, row_batch_size):
            row_end = min(row_start + row_batch_size, num_rows)
            batch_size = row_end - row_start

            word_acts_b = base_word_acts_cpu[row_start:row_end]          # [B, W, d_model]
            mask_b = base_word_mask[row_start:row_end]                    # [B, W]

            # Last noun token: index within padded word acts = mask_b.sum - 1
            last_noun_idx = (mask_b.sum(dim=1).long() - 1).clamp_min(0)  # [B]
            bidx = torch.arange(word_acts_b.shape[0])
            base_vec_batch = word_acts_b[bidx, last_noun_idx, :].to(sae_device, dtype=sae.W_enc.dtype)
            row_target_norms = base_vec_batch.norm(dim=-1)
            base_sig_single = (sae.encode(base_vec_batch) > 0)

            feat_gen = torch.Generator(device=sae_device)
            feat_gen.manual_seed(seed0 + 10_000 * layer + row_start)

            feat_ids_all = torch.randint(
                low=0,
                high=d_sae,
                size=(batch_size, trials, k_max),
                device=sae_device,
                generator=feat_gen)

            #base_dir_ids = feat_ids_all[:, :, 0].reshape(-1)
            #dirs_for_cos_base = _get_Wdec_dirs(sae, base_dir_ids).to(sae_device)
            #dirs_for_cos_base_norm = dirs_for_cos_base.norm(dim=-1, keepdim=True).clamp_min_(1e-8)
            #dirs_for_cos_base.div_(dirs_for_cos_base_norm)
            #del dirs_for_cos_base_norm

            #running_max = None
            #running_min = None

            #for i in range(1, k_max):
            #    dir_ids_i = feat_ids_all[:, :, i].reshape(-1)
            #    dirs_i = _get_Wdec_dirs(sae, dir_ids_i).to(sae_device)
            #    dirs_i_norm = dirs_i.norm(dim=-1, keepdim=True).clamp_min_(1e-8)
            #    dirs_i.div_(dirs_i_norm)
            #    del dirs_i_norm
            #
            #    cos_i = (dirs_i * dirs_for_cos_base).sum(dim=-1)
            #
            #    if running_max is None:
            #        running_max = cos_i
            #        running_min = cos_i
            #    else:
            #        running_max = torch.maximum(running_max, cos_i)
            #        running_min = torch.minimum(running_min, cos_i)
            #
            #    k = i + 1
            #    if k in eval_k_values:
            #        maxcos_step = running_max.view(batch_size, trials).unsqueeze(-1)
            #        mincos_step = running_min.view(batch_size, trials).unsqueeze(-1)
            #
            #        max_sum, max_sumsq, max_count_chunk = reduce_chunk_stats(maxcos_step)
            #        min_sum, min_sumsq, min_count_chunk = reduce_chunk_stats(mincos_step)
            #
            #        update_stat_block(
            #            maxcos_sums[layer], maxcos_sumsq[layer], maxcos_count[layer], k,
            #            max_sum, max_sumsq, max_count_chunk)
            #        update_stat_block(
            #            mincos_sums[layer], mincos_sumsq[layer], mincos_count[layer], k,
            #            min_sum, min_sumsq, min_count_chunk)
            #
            #        del maxcos_step, mincos_step
            #        del max_sum, max_sumsq, max_count_chunk, min_sum, min_sumsq, min_count_chunk
            #
            #    del dir_ids_i, dirs_i, cos_i
            #
            #del base_dir_ids, dirs_for_cos_base, running_max, running_min

            mode_seed_offsets = {"rand_signed": 101, "rand_pos": 202, "rand_neg": 303}

            for mode in mix_modes:
                coeff_gen = torch.Generator(device=sae_device)
                coeff_gen.manual_seed(seed0 + 100_000 * layer + 1_000 * row_start + mode_seed_offsets[mode])

                coeffs_all = sample_coeffs(
                    trials=batch_size * trials,
                    k_max=k_max,
                    mode=mode,
                    dist_width=dist_width,
                    device=sae_device,
                    dtype=base_vec_batch.dtype,
                    generator=coeff_gen).reshape(batch_size, trials, k_max)

                v = base_vec_batch.unsqueeze(1).expand(batch_size, trials, -1).clone()

                for chunk_start in range(0, k_max, k_chunk_size):
                    chunk_end = min(chunk_start + k_chunk_size, k_max)
                    chunk_len = chunk_end - chunk_start

                    feat_ids_chunk = feat_ids_all[:, :, chunk_start:chunk_end].reshape(-1)
                    dirs_chunk = _get_Wdec_dirs(sae, feat_ids_chunk).to(sae_device)
                    dirs_chunk = dirs_chunk.reshape(batch_size, trials, chunk_len, -1)

                    eval_k_in_chunk = [k for k in eval_k_values if chunk_start + 1 <= k <= chunk_end]

                    if not eval_k_in_chunk:
                        for local_i in range(chunk_len):
                            global_i = chunk_start + local_i
                            v = v + coeffs_all[:, :, global_i:global_i+1] * dirs_chunk[:, :, local_i, :]
                            v = scale_to_norm_batched(v, row_target_norms)

                        del feat_ids_chunk, dirs_chunk
                        continue

                    eval_step_indices = {k - 1 for k in eval_k_in_chunk}
                    comp_states: List[torch.Tensor] = []

                    for local_i in range(chunk_len):
                        global_i = chunk_start + local_i
                        v = v + coeffs_all[:, :, global_i:global_i+1] * dirs_chunk[:, :, local_i, :]
                        v = scale_to_norm_batched(v, row_target_norms)

                        if global_i in eval_step_indices:
                            comp_states.append(v.clone())

                    comp_chunk = torch.stack(comp_states, dim=2)
                    eval_chunk_len = comp_chunk.shape[2]

                    comp_flat = comp_chunk.reshape(batch_size * trials * eval_chunk_len, -1)

                    flat_base_row_ids = torch.arange(batch_size, device=sae_device).view(batch_size, 1, 1).expand(batch_size, trials, eval_chunk_len).reshape(-1)

                    lost_flat = eval_comp_flat_against_base_microbatched(
                        sae=sae,
                        comp_flat=comp_flat,
                        base_sig_single=base_sig_single,
                        flat_base_row_ids=flat_base_row_ids,
                        sae_eval_batch_size=sae_eval_batch_size)

                    lost_chunk = lost_flat.view(batch_size, trials, eval_chunk_len)

                    lost_sum, lost_sumsq, lost_count_chunk = reduce_chunk_stats(lost_chunk)

                    for local_j, k in enumerate(eval_k_in_chunk):
                        update_stat_block(
                            lost_sums_by_mode[mode][layer],
                            lost_sumsq_by_mode[mode][layer],
                            lost_count_by_mode[mode][layer],
                            k,
                            lost_sum[local_j:local_j+1],
                            lost_sumsq[local_j:local_j+1],
                            lost_count_chunk[local_j:local_j+1])

                    del comp_states, comp_chunk, comp_flat, flat_base_row_ids
                    del lost_flat, lost_chunk
                    del lost_sum, lost_sumsq, lost_count_chunk
                    del feat_ids_chunk, dirs_chunk

                del coeffs_all, v

            del word_acts_b, mask_b, last_noun_idx, bidx, base_vec_batch, row_target_norms, base_sig_single, feat_ids_all

        sae.to("cpu")
        del sae
        del base_word_acts_cpu
        gc.collect()
        torch.cuda.empty_cache()

    #mean_maxcos, std_maxcos = finalize_stats(maxcos_sums, maxcos_sumsq, maxcos_count, plot_layers, k_max, k_min_keep=1)
    #mean_mincos, std_mincos = finalize_stats(mincos_sums, mincos_sumsq, mincos_count, plot_layers, k_max, k_min_keep=1)
    #
    #shared_output_dir = base_output_dir / "shared_direction_geometry" / "plots"
    #shared_output_dir.mkdir(parents=True, exist_ok=True)

    k_values = list(eval_k_values)

    #maxcos_plot_path = shared_output_dir / "A_fixed_base__k_vs_maxcos_lines.png"
    #plot_against_k(
    #    selected_layers=plot_layers,
    #    mean_by_layer_by_k=mean_maxcos,
    #    std_by_layer_by_k=std_maxcos,
    #    title="Maximum cosine similarity of sampled SAE directions",
    #    ylabel="max-cossim",
    #    save_path=maxcos_plot_path,
    #    k_values=k_values,
    #    show_error_bars=False,
    #    ylim=(0.0, 1.0))
    #_log_plot_and_data(plot_path=maxcos_plot_path, mean_by_layer_by_k=mean_maxcos, std_by_layer_by_k=std_maxcos, metric_name="maxcos", mix_mode="shared_direction_geometry", setup_name="A_fixed_base")
    #
    #mincos_plot_path = shared_output_dir / "A_fixed_base__k_vs_mincos_lines.png"
    #plot_against_k(
    #    selected_layers=plot_layers,
    #    mean_by_layer_by_k=mean_mincos,
    #    std_by_layer_by_k=std_mincos,
    #    title="Minimum cosine similarity of sampled SAE directions",
    #    ylabel="min-cossim",
    #    save_path=mincos_plot_path,
    #    k_values=k_values,
    #    show_error_bars=False,
    #    ylim=(-0.5, 0.5))
    #_log_plot_and_data(plot_path=mincos_plot_path, mean_by_layer_by_k=mean_mincos, std_by_layer_by_k=std_mincos, metric_name="mincos", mix_mode="shared_direction_geometry", setup_name="A_fixed_base")

    for mode in mix_modes:
        if mode == "rand_signed":
            mode_tag = f"rand_signed{dist_width:g}"
            title_suffix = "Signed decoder-direction perturbations"
        elif mode == "rand_pos":
            mode_tag = f"rand_pos{dist_width:g}"
            title_suffix = "Positive decoder-direction perturbations"
        elif mode == "rand_neg":
            mode_tag = f"rand_neg{dist_width:g}"
            title_suffix = "Negative decoder-direction perturbations"
        else:
            mode_tag = mode
            title_suffix = mode

        output_dir = base_output_dir / mode_tag / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        mean_lost, std_lost = finalize_stats(lost_sums_by_mode[mode], lost_sumsq_by_mode[mode], lost_count_by_mode[mode], plot_layers, k_max, k_min_keep=1)

        lost_plot_path = output_dir / "A_fixed_base__k_vs_lost_rate_lines.png"
        plot_against_k(
            selected_layers=plot_layers,
            mean_by_layer_by_k=mean_lost,
            std_by_layer_by_k=std_lost,
            title=f"Loss of active SAE features under {title_suffix.lower()}",
            ylabel="lost rate",
            save_path=lost_plot_path,
            k_values=k_values,
            show_error_bars=False,
            ylim=(0.0, 1.0))
        _log_plot_and_data(plot_path=lost_plot_path, mean_by_layer_by_k=mean_lost, std_by_layer_by_k=std_lost, metric_name="lost_rate", mix_mode=mode_tag, setup_name="A_fixed_base")

        for layer in plot_layers:
            if k_max in mean_lost[layer]:
                wandb.log({f"{mode_tag}/layer_{layer}/lost_rate_at_kmax": mean_lost[layer][k_max]})

    run.finish()


if __name__ == "__main__":
    main()