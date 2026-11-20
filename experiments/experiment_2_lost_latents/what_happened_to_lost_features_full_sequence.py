from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import random
import hydra
import pandas as pd
import torch
import wandb
from datasets import Dataset
from omegaconf import DictConfig, MISSING
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path

from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_set_behaviour.util import SetupConfig, WandbConfig, init_wandb, setup, build_dummy_dataset
from sae_set_behaviour.lost_latents.generate_figure import plot_against_k, plot_two_metrics_against_layer, plot_main_metric_with_inlay
from sae_set_behaviour.lost_latents.construct_text_sequence import last_non_padding_token_index, PAPER_TEMPLATE
from sae_set_behaviour.lost_latents.dataset_cases import DataCase, get_data_case, build_chain_texts_and_words_for_rows, resolve_csv_path
from sae_set_behaviour.lost_latents.metrics import signature_infos, signature_infos_tokenwise, grouped_rates_among_lost, compute_noun_union_signature
from sae_set_behaviour.lost_latents.construct_embedding import tokenize_texts, find_noun_token_spans, extract_noun_word_acts
from sae_set_behaviour.lost_latents.statistics import aggregate_over_k
from sae_set_behaviour.lost_latents.logging import _log_plot_and_data, _metric_dicts_to_df, _tensor_metric_dicts_to_df


@dataclass
class Config(SetupConfig):
    base_save_dir: str = "results/experiment_2_lost_latents"
    data_cases: List[str] = field(default_factory=lambda: ["english", "german", "human_typicality"])
    data_csv: str = ""  # overrides the CSV; only applied when data_cases has exactly one entry
    plot_layers: Optional[List[int]] = None
    text_template: str = PAPER_TEMPLATE
    max_length: int = 64
    eps: float = 0.1
    k_max: Optional[int] = None  # defaults to the data_case's k_max; only applied when data_cases has exactly one entry
    k_values: Optional[List[int]] = None  # unused here: this script computes every k in one joint forward pass per row-batch, so restricting k_values wouldn't save compute. Declared only so shared config yamls (e.g. base.yaml) that set it don't break this script's structured config.
    trials: int = 200
    baseline_seed: int = 0
    sae_device: str = ""  # if set, SAEs run on this device (e.g. cuda:1); otherwise on args.device
    row_batch_size: int = 4
    wandb: WandbConfig = MISSING


ConfigStore.instance().store(name="config", node=Config)



@torch.no_grad()
def collect_sequence_activations_multi(*, model: HookedTransformer, texts: List[str], hook_names: List[str], device: str, max_length: int = 128) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    input_ids, attention_mask = tokenize_texts(model=model, texts=texts, device=device, max_length=max_length)
    attention_mask = attention_mask.to(torch.bool)
    hook_name_set = set(hook_names)
    _, cache = model.run_with_cache(input_ids, names_filter=lambda n: n in hook_name_set)
    acts_by_hook = {hn: cache[hn].detach() for hn in hook_names}
    return acts_by_hook, attention_mask


@torch.no_grad()
def precompute_row_metrics_for_layer(*, model: HookedTransformer, sae: SAE, rows: List[Dict[str, str]], case: DataCase, device: str, sae_device: str,
    k_max: int, max_length: int, eps: float, prev_hook_names: List[str], row_batch_size: int, text_template: str = "") -> Tuple[
    Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    hook_target = str(sae.cfg.metadata.hook_name)
    all_hook_names = list(dict.fromkeys([hook_target] + prev_hook_names))

    chains_by_row, primary_words_by_row, content_words_by_row = build_chain_texts_and_words_for_rows(rows, case, base_template=text_template)
    n_rows = len(chains_by_row)

    neg_by_k_cpu: Dict[int, List[torch.Tensor]] = {k: [] for k in range(1, k_max + 1)}
    zero_by_k_cpu: Dict[int, List[torch.Tensor]] = {k: [] for k in range(1, k_max + 1)}

    prev_active_among_lost_by_k_cpu: Dict[int, List[torch.Tensor]] = {k: [] for k in range(1, k_max + 1)}
    never_prev_active_among_lost_by_k_cpu: Dict[int, List[torch.Tensor]] = {k: [] for k in range(1, k_max + 1)}

    inactive_last_among_prev_active_lost_by_k_cpu: Dict[int, List[torch.Tensor]] = {k: [] for k in range(1, k_max + 1)}
    negative_last_among_prev_active_lost_by_k_cpu: Dict[int, List[torch.Tensor]] = {k: [] for k in range(1, k_max + 1)}

    always_inactive_among_never_prev_active_lost_by_k_cpu: Dict[int, List[torch.Tensor]] = {k: [] for k in range(1, k_max + 1)}
    always_negative_among_never_prev_active_lost_by_k_cpu: Dict[int, List[torch.Tensor]] = {k: [] for k in range(1, k_max + 1)}

    for start in range(0, n_rows, row_batch_size):
        end = min(start + row_batch_size, n_rows)
        batch_chains = chains_by_row[start:end]
        batch_words  = primary_words_by_row[start:end]
        batch_content_words = content_words_by_row[start:end]
        flat_texts = [t for chain in batch_chains for t in chain]
        flat_nouns = [w for words in batch_words for w in words]
        flat_content_words = [cw for content in batch_content_words for cw in content]

        acts_by_hook, tok_mask = collect_sequence_activations_multi(
            model=model,
            texts=flat_texts,
            hook_names=all_hook_names,
            device=device,
            max_length=max_length,
        )

        b = end - start
        s = k_max + 1

        tok_mask_flat = tok_mask.to(device=sae_device, dtype=torch.bool)
        seq_len = tok_mask_flat.shape[1]
        tok_mask_seq = tok_mask_flat.view(b, s, seq_len)

        noun_spans = find_noun_token_spans(model.tokenizer, flat_texts, flat_nouns)

        acts_t = acts_by_hook[hook_target].to(sae_device)

        # Noun union signature for "lost" definition (noun-only, unchanged)
        word_acts_flat, word_mask_flat = extract_noun_word_acts(acts_t.cpu(), noun_spans)
        signature_union_flat = compute_noun_union_signature(
            sae, word_acts_flat, word_mask_flat, sae_device=sae_device
        )  # [b*s, d_sae] bool, CPU
        del word_acts_flat, word_mask_flat

        d_sae = signature_union_flat.shape[-1]
        signature_union = signature_union_flat.to(sae_device).view(b, s, d_sae)
        base_signature = signature_union[:, 0, :]
        del signature_union_flat

        # Content token mask for prev-hook analysis: every content word's span at each
        # chain step (noun + all attributes so far for noun-attr chains; just the
        # step's own word for e.g. the human-typicality category/instance case).
        content_tok_mask_flat = torch.zeros(b * s, seq_len, dtype=torch.bool, device=sae_device)
        max_content_words = max((len(ws) for ws in flat_content_words), default=0)
        for word_idx in range(max_content_words):
            relevant_flat_indices = [i for i, ws in enumerate(flat_content_words) if len(ws) > word_idx]
            if not relevant_flat_indices:
                continue
            relevant_texts = [flat_texts[i] for i in relevant_flat_indices]
            relevant_words = [flat_content_words[i][word_idx] for i in relevant_flat_indices]
            word_spans = find_noun_token_spans(model.tokenizer, relevant_texts, relevant_words)
            for flat_i, (span_s, span_e) in zip(relevant_flat_indices, word_spans):
                if span_e > span_s:
                    content_tok_mask_flat[flat_i, span_s:span_e] = True
        content_tok_mask_seq = content_tok_mask_flat.view(b, s, seq_len)
        content_valid_tok = content_tok_mask_seq.unsqueeze(-1)  # [b, s, seq_len, 1]

        preact_neg_content = None
        preact_zero_content = None

        prev_sig_active_acc = None
        always_neg_prev_acc = None
        all_zero_prev_acc = None

        if len(prev_hook_names) > 0:
            for hn in prev_hook_names:
                using_target_hook = (hn == hook_target)

                if using_target_hook:
                    acts_prev = acts_t
                else:
                    acts_prev = acts_by_hook.pop(hn).to(sae_device)

                sig_prev_flat, _preact_pos_prev_flat, preact_neg_prev_flat, preact_any_prev_flat = signature_infos_tokenwise(
                    sae=sae,
                    acts=acts_prev,
                    eps=eps,
                )

                if using_target_hook:
                    _cmask = content_tok_mask_flat.unsqueeze(-1)  # [b*s, seq_len, 1]
                    preact_neg_content = (preact_neg_prev_flat | ~_cmask).all(dim=1).view(b, s, d_sae)
                    preact_zero_content = ((~preact_any_prev_flat) | ~_cmask).all(dim=1).view(b, s, d_sae)
                    del _cmask

                sig_prev = sig_prev_flat.view(b, s, seq_len, d_sae)
                preact_neg_prev = preact_neg_prev_flat.view(b, s, seq_len, d_sae)
                preact_any_prev = preact_any_prev_flat.view(b, s, seq_len, d_sae)

                sig_any_tok  = (sig_prev & content_valid_tok).any(dim=2)               # [b, s, d_sae]
                neg_all_tok  = (preact_neg_prev | (~content_valid_tok)).all(dim=2)    # [b, s, d_sae]
                zero_all_tok = ((~preact_any_prev) | (~content_valid_tok)).all(dim=2) # [b, s, d_sae]

                if prev_sig_active_acc is None:
                    prev_sig_active_acc = sig_any_tok
                    always_neg_prev_acc = neg_all_tok
                    all_zero_prev_acc = zero_all_tok
                else:
                    prev_sig_active_acc |= sig_any_tok
                    always_neg_prev_acc &= neg_all_tok
                    all_zero_prev_acc &= zero_all_tok

                if not using_target_hook:
                    del acts_prev
                del sig_prev_flat, _preact_pos_prev_flat, preact_neg_prev_flat, preact_any_prev_flat
                del sig_prev, preact_neg_prev, preact_any_prev
                del sig_any_tok, neg_all_tok, zero_all_tok

        acts_by_hook.pop(hook_target, None)
        del acts_by_hook

        for k in range(1, k_max + 1):
            sig_k        = signature_union[:, k, :]
            preact_neg_k = preact_neg_content[:, k, :]   # neg at ALL content tokens of k-th seq
            preact_zero_k = preact_zero_content[:, k, :] # inactive at ALL content tokens of k-th seq

            lost_k = base_signature & (~sig_k)

            n_lost = lost_k.sum(dim=1).to(torch.float32)
            n_neg  = (lost_k & preact_neg_k).sum(dim=1).to(torch.float32)
            n_zero = (lost_k & preact_zero_k).sum(dim=1).to(torch.float32)

            neg_rate_k  = torch.where(n_lost > 0, n_neg  / n_lost, torch.zeros_like(n_lost))
            zero_rate_k = torch.where(n_lost > 0, n_zero / n_lost, torch.zeros_like(n_lost))

            neg_by_k_cpu[k].append(neg_rate_k.detach().cpu())
            zero_by_k_cpu[k].append(zero_rate_k.detach().cpu())

            if prev_sig_active_acc is None:
                prev_active_among_lost_rate = torch.zeros((b,), device=sae_device)
                never_prev_active_among_lost_rate = torch.zeros((b,), device=sae_device)
                inactive_last_among_prev_active_lost_rate = torch.zeros((b,), device=sae_device)
                negative_last_among_prev_active_lost_rate = torch.zeros((b,), device=sae_device)
                always_inactive_among_never_prev_active_lost_rate = torch.zeros((b,), device=sae_device)
                always_negative_among_never_prev_active_lost_rate = torch.zeros((b,), device=sae_device)
            else:
                prev_sig_active = prev_sig_active_acc[:, k, :]
                always_neg_prev = always_neg_prev_acc[:, k, :]
                all_zero_prev = all_zero_prev_acc[:, k, :]

                prev_active_among_lost_rate, never_prev_active_among_lost_rate, inactive_last_among_prev_active_lost_rate, negative_last_among_prev_active_lost_rate, always_inactive_among_never_prev_active_lost_rate, always_negative_among_never_prev_active_lost_rate = grouped_rates_among_lost(
                    lost_mask=lost_k,
                    prev_sig_active=prev_sig_active,
                    always_neg_prev=always_neg_prev,
                    all_zero_prev=all_zero_prev,
                    preact_neg_last=preact_neg_k,    # neg at ALL content tokens of k-th seq
                    preact_any_last=~preact_zero_k,  # active at ANY content token of k-th seq
                )

            prev_active_among_lost_by_k_cpu[k].append(prev_active_among_lost_rate.detach().cpu())
            never_prev_active_among_lost_by_k_cpu[k].append(never_prev_active_among_lost_rate.detach().cpu())
            inactive_last_among_prev_active_lost_by_k_cpu[k].append(inactive_last_among_prev_active_lost_rate.detach().cpu())
            negative_last_among_prev_active_lost_by_k_cpu[k].append(negative_last_among_prev_active_lost_rate.detach().cpu())
            always_inactive_among_never_prev_active_lost_by_k_cpu[k].append(always_inactive_among_never_prev_active_lost_rate.detach().cpu())
            always_negative_among_never_prev_active_lost_by_k_cpu[k].append(always_negative_among_never_prev_active_lost_rate.detach().cpu())

            del sig_k, preact_neg_k, preact_zero_k, lost_k
            del n_lost, n_neg, n_zero, neg_rate_k, zero_rate_k
            del prev_active_among_lost_rate, never_prev_active_among_lost_rate
            del inactive_last_among_prev_active_lost_rate, negative_last_among_prev_active_lost_rate
            del always_inactive_among_never_prev_active_lost_rate, always_negative_among_never_prev_active_lost_rate

        del tok_mask_flat, tok_mask_seq
        del content_tok_mask_flat, content_tok_mask_seq, content_valid_tok
        del signature_union, preact_neg_content, preact_zero_content, base_signature
        del acts_t

        if prev_sig_active_acc is not None:
            del prev_sig_active_acc, always_neg_prev_acc, all_zero_prev_acc

    neg_by_k = {k: torch.cat(v, dim=0) for k, v in neg_by_k_cpu.items()}
    zero_by_k = {k: torch.cat(v, dim=0) for k, v in zero_by_k_cpu.items()}

    prev_active_among_lost_by_k = {k: torch.cat(v, dim=0) for k, v in prev_active_among_lost_by_k_cpu.items()}
    never_prev_active_among_lost_by_k = {k: torch.cat(v, dim=0) for k, v in never_prev_active_among_lost_by_k_cpu.items()}

    inactive_last_among_prev_active_lost_by_k = {k: torch.cat(v, dim=0) for k, v in inactive_last_among_prev_active_lost_by_k_cpu.items()}
    negative_last_among_prev_active_lost_by_k = {k: torch.cat(v, dim=0) for k, v in negative_last_among_prev_active_lost_by_k_cpu.items()}

    always_inactive_among_never_prev_active_lost_by_k = {k: torch.cat(v, dim=0) for k, v in always_inactive_among_never_prev_active_lost_by_k_cpu.items()}
    always_negative_among_never_prev_active_lost_by_k = {k: torch.cat(v, dim=0) for k, v in always_negative_among_never_prev_active_lost_by_k_cpu.items()}

    return neg_by_k, zero_by_k, prev_active_among_lost_by_k, never_prev_active_among_lost_by_k, inactive_last_among_prev_active_lost_by_k, negative_last_among_prev_active_lost_by_k, always_inactive_among_never_prev_active_lost_by_k, always_negative_among_never_prev_active_lost_by_k


def sample_trials_from_row_metrics(*, row_metric_by_k: Dict[int, torch.Tensor], sampled_row_ids: List[int]) -> Dict[int, torch.Tensor]:
    idx = torch.tensor(sampled_row_ids, dtype=torch.long)
    out: Dict[int, torch.Tensor] = {}
    for k, vec in row_metric_by_k.items():
        out[k] = vec.index_select(0, idx)
    return out


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(args: DictConfig) -> None:
    init_wandb(args.wandb, args, job_type="lost_features_full_sequence")
    run = wandb.run

    device = str(args.device)
    sae_device = str(getattr(args, "sae_device", "") or args.device)

    plot_layers: List[int] = list(args.plot_layers) if args.plot_layers else list(args.sae.layers)
    plot_layers = [int(x) for x in plot_layers]

    case_names = list(getattr(args, "data_cases", None) or ["english"])
    explicit_k_max = getattr(args, "k_max", None) if len(case_names) == 1 else None
    explicit_csv = getattr(args, "data_csv", None) if len(case_names) == 1 else None

    trials = int(getattr(args, "trials", 200))
    seed0 = int(getattr(args, "baseline_seed", 0))
    max_length = int(getattr(args, "max_length", 64))
    eps = float(getattr(args, "eps", 1e-2))
    row_batch_size = int(getattr(args, "row_batch_size", 16))
    text_template = str(getattr(args, "text_template", "") or "")

    dummy_dataset = build_dummy_dataset()
    _layers = list(args.sae.layers)
    args.sae.layers = plot_layers
    try:
        model, saes, _, _ = setup(args, dummy_dataset)
    finally:
        args.sae.layers = _layers
    model.eval()

    sae_by_layer: Dict[int, SAE] = {int(layer): sae for layer, sae in zip(plot_layers, saes)}


    for case_name in case_names:
        try:
            case = get_data_case(str(case_name))
            k_max = int(explicit_k_max or case.k_max)
            csv_path = resolve_csv_path(case, explicit_csv, to_absolute_path)
            rows = case.load_rows(csv_path)
            output_dir = Path(args.base_save_dir) / str(args.sae.model.name) / str(args.sae.release) / case.name / "lost_lasttok_signature_preact" / "plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            metric_namespace = f"{case.name}/lost_lasttok_signature_preact"

            for setup_name in ["A_fixed_base"]:
                mean_neg_by_layer: Dict[int, float] = {}
                std_neg_by_layer: Dict[int, float] = {}
                std_neg_across_k_by_layer: Dict[int, float] = {}

                mean_zero_by_layer: Dict[int, float] = {}
                std_zero_by_layer: Dict[int, float] = {}
                std_zero_across_k_by_layer: Dict[int, float] = {}

                mean_neg_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}
                std_neg_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}
                mean_zero_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}
                std_zero_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}

                neg_tensor_by_layer_by_k: Dict[int, Dict[int, torch.Tensor]] = {layer: {} for layer in plot_layers}
                zero_tensor_by_layer_by_k: Dict[int, Dict[int, torch.Tensor]] = {layer: {} for layer in plot_layers}

                mean_prev_active_among_lost_by_layer: Dict[int, float] = {}
                std_prev_active_among_lost_by_layer: Dict[int, float] = {}
                mean_never_prev_active_among_lost_by_layer: Dict[int, float] = {}
                std_never_prev_active_among_lost_by_layer: Dict[int, float] = {}

                mean_prev_active_among_lost_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}
                std_prev_active_among_lost_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}
                mean_never_prev_active_among_lost_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}
                std_never_prev_active_among_lost_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}

                prev_active_among_lost_tensor_by_layer_by_k: Dict[int, Dict[int, torch.Tensor]] = {layer: {} for layer in plot_layers}
                never_prev_active_among_lost_tensor_by_layer_by_k: Dict[int, Dict[int, torch.Tensor]] = {layer: {} for layer in plot_layers}

                mean_inactive_last_among_prev_active_lost_by_layer: Dict[int, float] = {}
                std_inactive_last_among_prev_active_lost_by_layer: Dict[int, float] = {}
                mean_negative_last_among_prev_active_lost_by_layer: Dict[int, float] = {}
                std_negative_last_among_prev_active_lost_by_layer: Dict[int, float] = {}

                mean_inactive_last_among_prev_active_lost_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}
                std_inactive_last_among_prev_active_lost_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}
                mean_negative_last_among_prev_active_lost_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}
                std_negative_last_among_prev_active_lost_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}

                inactive_last_among_prev_active_lost_tensor_by_layer_by_k: Dict[int, Dict[int, torch.Tensor]] = {layer: {} for layer in plot_layers}
                negative_last_among_prev_active_lost_tensor_by_layer_by_k: Dict[int, Dict[int, torch.Tensor]] = {layer: {} for layer in plot_layers}

                mean_always_inactive_among_never_prev_active_lost_by_layer: Dict[int, float] = {}
                std_always_inactive_among_never_prev_active_lost_by_layer: Dict[int, float] = {}
                mean_always_negative_among_never_prev_active_lost_by_layer: Dict[int, float] = {}
                std_always_negative_among_never_prev_active_lost_by_layer: Dict[int, float] = {}

                mean_always_inactive_among_never_prev_active_lost_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}
                std_always_inactive_among_never_prev_active_lost_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}
                mean_always_negative_among_never_prev_active_lost_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}
                std_always_negative_among_never_prev_active_lost_by_layer_by_k: Dict[int, Dict[int, float]] = {layer: {} for layer in plot_layers}

                always_inactive_among_never_prev_active_lost_tensor_by_layer_by_k: Dict[int, Dict[int, torch.Tensor]] = {layer: {} for layer in plot_layers}
                always_negative_among_never_prev_active_lost_tensor_by_layer_by_k: Dict[int, Dict[int, torch.Tensor]] = {layer: {} for layer in plot_layers}

                for layer in plot_layers:
                    sae = sae_by_layer[layer]
                    sae = sae.to(sae_device)
                    sae.eval()
                    try:

                        hook_target = str(sae.cfg.metadata.hook_name)
                        pattern = f"blocks.{layer}."
                        suffix = hook_target.split(pattern, 1)[-1]
                        prev_hook_names = [f"blocks.{l}.{suffix}" for l in range(layer + 1)]

                        neg_by_row_by_k, zero_by_row_by_k, prev_active_among_lost_by_row_by_k, never_prev_active_among_lost_by_row_by_k, inactive_last_among_prev_active_lost_by_row_by_k, negative_last_among_prev_active_lost_by_row_by_k, always_inactive_among_never_prev_active_lost_by_row_by_k, always_negative_among_never_prev_active_lost_by_row_by_k = precompute_row_metrics_for_layer(
                            model=model,
                            sae=sae,
                            rows=rows,
                            case=case,
                            device=device,
                            sae_device=sae_device,
                            k_max=k_max,
                            max_length=max_length,
                            eps=eps,
                            prev_hook_names=prev_hook_names,
                            row_batch_size=row_batch_size,
                            text_template=text_template,
                        )

                        rng = random.Random(seed0 + 10_000 * layer)
                        sampled_row_ids = [rng.randrange(len(rows)) for _ in range(trials)]

                        neg_by_k = sample_trials_from_row_metrics(row_metric_by_k=neg_by_row_by_k, sampled_row_ids=sampled_row_ids)
                        zero_by_k = sample_trials_from_row_metrics(row_metric_by_k=zero_by_row_by_k, sampled_row_ids=sampled_row_ids)

                        prev_active_among_lost_by_k = sample_trials_from_row_metrics(row_metric_by_k=prev_active_among_lost_by_row_by_k, sampled_row_ids=sampled_row_ids)
                        never_prev_active_among_lost_by_k = sample_trials_from_row_metrics(row_metric_by_k=never_prev_active_among_lost_by_row_by_k, sampled_row_ids=sampled_row_ids)

                        inactive_last_among_prev_active_lost_by_k = sample_trials_from_row_metrics(row_metric_by_k=inactive_last_among_prev_active_lost_by_row_by_k, sampled_row_ids=sampled_row_ids)
                        negative_last_among_prev_active_lost_by_k = sample_trials_from_row_metrics(row_metric_by_k=negative_last_among_prev_active_lost_by_row_by_k, sampled_row_ids=sampled_row_ids)

                        always_inactive_among_never_prev_active_lost_by_k = sample_trials_from_row_metrics(row_metric_by_k=always_inactive_among_never_prev_active_lost_by_row_by_k, sampled_row_ids=sampled_row_ids)
                        always_negative_among_never_prev_active_lost_by_k = sample_trials_from_row_metrics(row_metric_by_k=always_negative_among_never_prev_active_lost_by_row_by_k, sampled_row_ids=sampled_row_ids)

                        neg_tensor_by_layer_by_k[layer] = neg_by_k
                        zero_tensor_by_layer_by_k[layer] = zero_by_k

                        prev_active_among_lost_tensor_by_layer_by_k[layer] = prev_active_among_lost_by_k
                        never_prev_active_among_lost_tensor_by_layer_by_k[layer] = never_prev_active_among_lost_by_k

                        inactive_last_among_prev_active_lost_tensor_by_layer_by_k[layer] = inactive_last_among_prev_active_lost_by_k
                        negative_last_among_prev_active_lost_tensor_by_layer_by_k[layer] = negative_last_among_prev_active_lost_by_k

                        always_inactive_among_never_prev_active_lost_tensor_by_layer_by_k[layer] = always_inactive_among_never_prev_active_lost_by_k
                        always_negative_among_never_prev_active_lost_tensor_by_layer_by_k[layer] = always_negative_among_never_prev_active_lost_by_k

                        for k in sorted(neg_by_k.keys()):
                            mean_neg_by_layer_by_k[layer][k] = float(neg_by_k[k].mean().item())
                            std_neg_by_layer_by_k[layer][k] = float(neg_by_k[k].std(unbiased=True).item())

                        for k in sorted(zero_by_k.keys()):
                            mean_zero_by_layer_by_k[layer][k] = float(zero_by_k[k].mean().item())
                            std_zero_by_layer_by_k[layer][k] = float(zero_by_k[k].std(unbiased=True).item())

                        for k in sorted(prev_active_among_lost_by_k.keys()):
                            mean_prev_active_among_lost_by_layer_by_k[layer][k] = float(prev_active_among_lost_by_k[k].mean().item())
                            std_prev_active_among_lost_by_layer_by_k[layer][k] = float(prev_active_among_lost_by_k[k].std(unbiased=True).item())

                        for k in sorted(never_prev_active_among_lost_by_k.keys()):
                            mean_never_prev_active_among_lost_by_layer_by_k[layer][k] = float(never_prev_active_among_lost_by_k[k].mean().item())
                            std_never_prev_active_among_lost_by_layer_by_k[layer][k] = float(never_prev_active_among_lost_by_k[k].std(unbiased=True).item())

                        for k in sorted(inactive_last_among_prev_active_lost_by_k.keys()):
                            mean_inactive_last_among_prev_active_lost_by_layer_by_k[layer][k] = float(inactive_last_among_prev_active_lost_by_k[k].mean().item())
                            std_inactive_last_among_prev_active_lost_by_layer_by_k[layer][k] = float(inactive_last_among_prev_active_lost_by_k[k].std(unbiased=True).item())

                        for k in sorted(negative_last_among_prev_active_lost_by_k.keys()):
                            mean_negative_last_among_prev_active_lost_by_layer_by_k[layer][k] = float(negative_last_among_prev_active_lost_by_k[k].mean().item())
                            std_negative_last_among_prev_active_lost_by_layer_by_k[layer][k] = float(negative_last_among_prev_active_lost_by_k[k].std(unbiased=True).item())

                        for k in sorted(always_inactive_among_never_prev_active_lost_by_k.keys()):
                            mean_always_inactive_among_never_prev_active_lost_by_layer_by_k[layer][k] = float(always_inactive_among_never_prev_active_lost_by_k[k].mean().item())
                            std_always_inactive_among_never_prev_active_lost_by_layer_by_k[layer][k] = float(always_inactive_among_never_prev_active_lost_by_k[k].std(unbiased=True).item())

                        for k in sorted(always_negative_among_never_prev_active_lost_by_k.keys()):
                            mean_always_negative_among_never_prev_active_lost_by_layer_by_k[layer][k] = float(always_negative_among_never_prev_active_lost_by_k[k].mean().item())
                            std_always_negative_among_never_prev_active_lost_by_layer_by_k[layer][k] = float(always_negative_among_never_prev_active_lost_by_k[k].std(unbiased=True).item())

                        neg_mean_layer, neg_std_layer, neg_std_across_k = aggregate_over_k(neg_by_k)
                        zero_mean_layer, zero_std_layer, zero_std_across_k = aggregate_over_k(zero_by_k)

                        mean_neg_by_layer[layer] = neg_mean_layer
                        std_neg_by_layer[layer] = neg_std_layer
                        std_neg_across_k_by_layer[layer] = neg_std_across_k

                        mean_zero_by_layer[layer] = zero_mean_layer
                        std_zero_by_layer[layer] = zero_std_layer
                        std_zero_across_k_by_layer[layer] = zero_std_across_k

                        prev_active_among_lost_mean_layer, prev_active_among_lost_std_layer, _ = aggregate_over_k(prev_active_among_lost_by_k)
                        never_prev_active_among_lost_mean_layer, never_prev_active_among_lost_std_layer, _ = aggregate_over_k(never_prev_active_among_lost_by_k)

                        inactive_last_among_prev_active_lost_mean_layer, inactive_last_among_prev_active_lost_std_layer, _ = aggregate_over_k(inactive_last_among_prev_active_lost_by_k)
                        negative_last_among_prev_active_lost_mean_layer, negative_last_among_prev_active_lost_std_layer, _ = aggregate_over_k(negative_last_among_prev_active_lost_by_k)

                        always_inactive_among_never_prev_active_lost_mean_layer, always_inactive_among_never_prev_active_lost_std_layer, _ = aggregate_over_k(always_inactive_among_never_prev_active_lost_by_k)
                        always_negative_among_never_prev_active_lost_mean_layer, always_negative_among_never_prev_active_lost_std_layer, _ = aggregate_over_k(always_negative_among_never_prev_active_lost_by_k)

                        mean_prev_active_among_lost_by_layer[layer] = prev_active_among_lost_mean_layer
                        std_prev_active_among_lost_by_layer[layer] = prev_active_among_lost_std_layer
                        mean_never_prev_active_among_lost_by_layer[layer] = never_prev_active_among_lost_mean_layer
                        std_never_prev_active_among_lost_by_layer[layer] = never_prev_active_among_lost_std_layer

                        mean_inactive_last_among_prev_active_lost_by_layer[layer] = inactive_last_among_prev_active_lost_mean_layer
                        std_inactive_last_among_prev_active_lost_by_layer[layer] = inactive_last_among_prev_active_lost_std_layer
                        mean_negative_last_among_prev_active_lost_by_layer[layer] = negative_last_among_prev_active_lost_mean_layer
                        std_negative_last_among_prev_active_lost_by_layer[layer] = negative_last_among_prev_active_lost_std_layer

                        mean_always_inactive_among_never_prev_active_lost_by_layer[layer] = always_inactive_among_never_prev_active_lost_mean_layer
                        std_always_inactive_among_never_prev_active_lost_by_layer[layer] = always_inactive_among_never_prev_active_lost_std_layer
                        mean_always_negative_among_never_prev_active_lost_by_layer[layer] = always_negative_among_never_prev_active_lost_mean_layer
                        std_always_negative_among_never_prev_active_lost_by_layer[layer] = always_negative_among_never_prev_active_lost_std_layer

                    finally:
                        sae.to("cpu")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                neg_k_plot_path = output_dir / f"{setup_name}__k_vs_neg_rate_among_lost.png"
                k_values_neg = sorted({k for layer in plot_layers for k in mean_neg_by_layer_by_k[layer].keys()})
                plot_against_k(
                    selected_layers=plot_layers,
                    mean_by_layer_by_k=mean_neg_by_layer_by_k,
                    std_by_layer_by_k=std_neg_by_layer_by_k,
                    title="Negative pre-activation among lost features",
                    ylabel="rate",
                    save_path=neg_k_plot_path,
                    k_values=k_values_neg,
                    show_error_bars=False,
                    ylim=(0.0, 1.0))
                _log_plot_and_data(plot_path=neg_k_plot_path, mean_by_layer_by_k=mean_neg_by_layer_by_k, std_by_layer_by_k=std_neg_by_layer_by_k, metric_name="neg_rate_among_lost", mix_mode=metric_namespace, setup_name=setup_name)

                zero_k_plot_path = output_dir / f"{setup_name}__k_vs_zero_rate_among_lost.png"
                k_values_zero = sorted({k for layer in plot_layers for k in mean_zero_by_layer_by_k[layer].keys()})
                plot_against_k(
                    selected_layers=plot_layers,
                    mean_by_layer_by_k=mean_zero_by_layer_by_k,
                    std_by_layer_by_k=std_zero_by_layer_by_k,
                    title="Inactive pre-activation among lost features",
                    ylabel="rate",
                    save_path=zero_k_plot_path,
                    k_values=k_values_zero,
                    show_error_bars=False,
                    ylim=(0.0, 1.0),
                )
                _log_plot_and_data(plot_path=zero_k_plot_path, mean_by_layer_by_k=mean_zero_by_layer_by_k, std_by_layer_by_k=std_zero_by_layer_by_k, metric_name="zero_rate_among_lost", mix_mode=metric_namespace, setup_name=setup_name)

                prev_active_k_plot_path = output_dir / f"{setup_name}__k_vs_prev_active_among_lost.png"
                k_values_prev_active = sorted({k for layer in plot_layers for k in mean_prev_active_among_lost_by_layer_by_k[layer].keys()})
                plot_against_k(
                    selected_layers=plot_layers,
                    mean_by_layer_by_k=mean_prev_active_among_lost_by_layer_by_k,
                    std_by_layer_by_k=std_prev_active_among_lost_by_layer_by_k,
                    title="Previously active among lost features",
                    ylabel="rate",
                    save_path=prev_active_k_plot_path,
                    k_values=k_values_prev_active,
                    show_error_bars=False,
                    ylim=(0.0, 1.0),
                )
                _log_plot_and_data(plot_path=prev_active_k_plot_path, mean_by_layer_by_k=mean_prev_active_among_lost_by_layer_by_k, std_by_layer_by_k=std_prev_active_among_lost_by_layer_by_k, metric_name="prev_active_among_lost", mix_mode=metric_namespace, setup_name=setup_name)

                never_prev_active_k_plot_path = output_dir / f"{setup_name}__k_vs_never_prev_active_among_lost.png"
                k_values_never_prev_active = sorted({k for layer in plot_layers for k in mean_never_prev_active_among_lost_by_layer_by_k[layer].keys()})
                plot_against_k(
                    selected_layers=plot_layers,
                    mean_by_layer_by_k=mean_never_prev_active_among_lost_by_layer_by_k,
                    std_by_layer_by_k=std_never_prev_active_among_lost_by_layer_by_k,
                    title="Never previously active among lost features",
                    ylabel="rate",
                    save_path=never_prev_active_k_plot_path,
                    k_values=k_values_never_prev_active,
                    show_error_bars=False,
                    ylim=(0.0, 1.0),
                )
                _log_plot_and_data(plot_path=never_prev_active_k_plot_path, mean_by_layer_by_k=mean_never_prev_active_among_lost_by_layer_by_k, std_by_layer_by_k=std_never_prev_active_among_lost_by_layer_by_k, metric_name="never_prev_active_among_lost", mix_mode=metric_namespace, setup_name=setup_name)

                inactive_last_prev_active_k_plot_path = output_dir / f"{setup_name}__k_vs_inactive_last_among_prev_active_lost.png"
                k_values_inactive_last_prev_active = sorted({k for layer in plot_layers for k in mean_inactive_last_among_prev_active_lost_by_layer_by_k[layer].keys()})
                plot_against_k(
                    selected_layers=plot_layers,
                    mean_by_layer_by_k=mean_inactive_last_among_prev_active_lost_by_layer_by_k,
                    std_by_layer_by_k=std_inactive_last_among_prev_active_lost_by_layer_by_k,
                    title="Inactive at last token among previously active lost features",
                    ylabel="rate",
                    save_path=inactive_last_prev_active_k_plot_path,
                    k_values=k_values_inactive_last_prev_active,
                    show_error_bars=False,
                    ylim=(0.0, 1.0),
                )
                _log_plot_and_data(plot_path=inactive_last_prev_active_k_plot_path, mean_by_layer_by_k=mean_inactive_last_among_prev_active_lost_by_layer_by_k, std_by_layer_by_k=std_inactive_last_among_prev_active_lost_by_layer_by_k, metric_name="inactive_last_among_prev_active_lost", mix_mode=metric_namespace, setup_name=setup_name)

                negative_last_prev_active_k_plot_path = output_dir / f"{setup_name}__k_vs_negative_last_among_prev_active_lost.png"
                k_values_negative_last_prev_active = sorted({k for layer in plot_layers for k in mean_negative_last_among_prev_active_lost_by_layer_by_k[layer].keys()})
                plot_against_k(
                    selected_layers=plot_layers,
                    mean_by_layer_by_k=mean_negative_last_among_prev_active_lost_by_layer_by_k,
                    std_by_layer_by_k=std_negative_last_among_prev_active_lost_by_layer_by_k,
                    title="Negative at last token among previously active lost features",
                    ylabel="rate",
                    save_path=negative_last_prev_active_k_plot_path,
                    k_values=k_values_negative_last_prev_active,
                    show_error_bars=False,
                    ylim=(0.0, 1.0),
                )
                _log_plot_and_data(plot_path=negative_last_prev_active_k_plot_path, mean_by_layer_by_k=mean_negative_last_among_prev_active_lost_by_layer_by_k, std_by_layer_by_k=std_negative_last_among_prev_active_lost_by_layer_by_k, metric_name="negative_last_among_prev_active_lost", mix_mode=metric_namespace, setup_name=setup_name)

                always_inactive_never_prev_active_k_plot_path = output_dir / f"{setup_name}__k_vs_always_inactive_among_never_prev_active_lost.png"
                k_values_always_inactive_never_prev_active = sorted({k for layer in plot_layers for k in mean_always_inactive_among_never_prev_active_lost_by_layer_by_k[layer].keys()})
                plot_against_k(
                    selected_layers=plot_layers,
                    mean_by_layer_by_k=mean_always_inactive_among_never_prev_active_lost_by_layer_by_k,
                    std_by_layer_by_k=std_always_inactive_among_never_prev_active_lost_by_layer_by_k,
                    title="Always inactive among never previously active lost features",
                    ylabel="rate",
                    save_path=always_inactive_never_prev_active_k_plot_path,
                    k_values=k_values_always_inactive_never_prev_active,
                    show_error_bars=False,
                    ylim=(0.0, 1.0),
                )
                _log_plot_and_data(plot_path=always_inactive_never_prev_active_k_plot_path, mean_by_layer_by_k=mean_always_inactive_among_never_prev_active_lost_by_layer_by_k, std_by_layer_by_k=std_always_inactive_among_never_prev_active_lost_by_layer_by_k, metric_name="always_inactive_among_never_prev_active_lost", mix_mode=metric_namespace, setup_name=setup_name)

                always_negative_never_prev_active_k_plot_path = output_dir / f"{setup_name}__k_vs_always_negative_among_never_prev_active_lost.png"
                k_values_always_negative_never_prev_active = sorted({k for layer in plot_layers for k in mean_always_negative_among_never_prev_active_lost_by_layer_by_k[layer].keys()})
                plot_against_k(
                    selected_layers=plot_layers,
                    mean_by_layer_by_k=mean_always_negative_among_never_prev_active_lost_by_layer_by_k,
                    std_by_layer_by_k=std_always_negative_among_never_prev_active_lost_by_layer_by_k,
                    title="Always negative among never previously active lost features",
                    ylabel="rate",
                    save_path=always_negative_never_prev_active_k_plot_path,
                    k_values=k_values_always_negative_never_prev_active,
                    show_error_bars=False,
                    ylim=(0.0, 1.0),
                )
                _log_plot_and_data(plot_path=always_negative_never_prev_active_k_plot_path, mean_by_layer_by_k=mean_always_negative_among_never_prev_active_lost_by_layer_by_k, std_by_layer_by_k=std_always_negative_among_never_prev_active_lost_by_layer_by_k, metric_name="always_negative_among_never_prev_active_lost", mix_mode=metric_namespace, setup_name=setup_name)

                layer_plot_path = output_dir / f"{setup_name}__layer_vs_neg_zero_rate_mean_over_k.png"
                plot_two_metrics_against_layer(
                    layers=plot_layers,
                    mean_metric1_by_layer=mean_neg_by_layer,
                    std_metric1_by_layer=std_neg_by_layer,
                    mean_metric2_by_layer=mean_zero_by_layer,
                    std_metric2_by_layer=std_zero_by_layer,
                    metric1_label="negative pre-activation",
                    metric2_label="inactive pre-activation",
                    title="Mean lost-feature diagnostics across layers",
                    ylabel="rate",
                    save_path=layer_plot_path,
                    show_error_bars=True,
                    metric1_color="cyan",
                    metric2_color="red",
                )
                layer_df = pd.concat([
                    _metric_dicts_to_df({layer: {0: mean_neg_by_layer[layer]} for layer in plot_layers}, {layer: {0: std_neg_by_layer[layer]} for layer in plot_layers}, "negative_preact_mean_over_k", metric_namespace, setup_name),
                    _metric_dicts_to_df({layer: {0: mean_zero_by_layer[layer]} for layer in plot_layers}, {layer: {0: std_zero_by_layer[layer]} for layer in plot_layers}, "inactive_preact_mean_over_k", metric_namespace, setup_name),
                ], ignore_index=True)
                layer_csv_path = layer_plot_path.with_suffix(".csv")
                layer_df.to_csv(layer_csv_path, index=False)
                wandb.log({
                    f"{metric_namespace}/{setup_name}/layer_vs_neg_zero_plot": wandb.Image(str(layer_plot_path)),
                    f"{metric_namespace}/{setup_name}/layer_vs_neg_zero_data": wandb.Table(dataframe=layer_df),
                })

                for k in range(1, k_max + 1):
                    prev_active_main_by_layer_k = {layer: mean_prev_active_among_lost_by_layer_by_k[layer][k] for layer in plot_layers}
                    prev_active_main_std_by_layer_k = {layer: std_prev_active_among_lost_by_layer_by_k[layer][k] for layer in plot_layers}

                    inactive_last_prev_active_by_layer_k = {layer: mean_inactive_last_among_prev_active_lost_by_layer_by_k[layer][k] for layer in plot_layers}
                    inactive_last_prev_active_std_by_layer_k = {layer: std_inactive_last_among_prev_active_lost_by_layer_by_k[layer][k] for layer in plot_layers}

                    negative_last_prev_active_by_layer_k = {layer: mean_negative_last_among_prev_active_lost_by_layer_by_k[layer][k] for layer in plot_layers}
                    negative_last_prev_active_std_by_layer_k = {layer: std_negative_last_among_prev_active_lost_by_layer_by_k[layer][k] for layer in plot_layers}

                    prev_active_inlay_plot_path = output_dir / f"{setup_name}__layer_vs_prev_active_with_inlay__k{k}.png"
                    plot_main_metric_with_inlay(
                        layers=plot_layers,
                        mean_main_by_layer=prev_active_main_by_layer_k,
                        std_main_by_layer=prev_active_main_std_by_layer_k,
                        mean_inlay_metric1_by_layer=inactive_last_prev_active_by_layer_k,
                        std_inlay_metric1_by_layer=inactive_last_prev_active_std_by_layer_k,
                        mean_inlay_metric2_by_layer=negative_last_prev_active_by_layer_k,
                        std_inlay_metric2_by_layer=negative_last_prev_active_std_by_layer_k,
                        main_label="previously active among lost",
                        inlay_metric1_label="inactive at last",
                        inlay_metric2_label="negative at last",
                        title=f"Previously active lost features (k={k})",
                        ylabel="rate",
                        save_path=prev_active_inlay_plot_path,
                        show_error_bars=True,
                        main_color="royalblue",
                        inlay_metric1_color="red",
                        inlay_metric2_color="cyan",
                    )
                    wandb.log({
                        f"{metric_namespace}/{setup_name}/layer_vs_prev_active_with_inlay_k_{k}": wandb.Image(str(prev_active_inlay_plot_path)),
                    })

                for k in range(1, k_max + 1):
                    never_prev_active_main_by_layer_k = {layer: mean_never_prev_active_among_lost_by_layer_by_k[layer][k] for layer in plot_layers}
                    never_prev_active_main_std_by_layer_k = {layer: std_never_prev_active_among_lost_by_layer_by_k[layer][k] for layer in plot_layers}

                    always_inactive_never_prev_active_by_layer_k = {layer: mean_always_inactive_among_never_prev_active_lost_by_layer_by_k[layer][k] for layer in plot_layers}
                    always_inactive_never_prev_active_std_by_layer_k = {layer: std_always_inactive_among_never_prev_active_lost_by_layer_by_k[layer][k] for layer in plot_layers}

                    always_negative_never_prev_active_by_layer_k = {layer: mean_always_negative_among_never_prev_active_lost_by_layer_by_k[layer][k] for layer in plot_layers}
                    always_negative_never_prev_active_std_by_layer_k = {layer: std_always_negative_among_never_prev_active_lost_by_layer_by_k[layer][k] for layer in plot_layers}

                    never_prev_active_inlay_plot_path = output_dir / f"{setup_name}__layer_vs_never_prev_active_with_inlay__k{k}.png"
                    plot_main_metric_with_inlay(
                        layers=plot_layers,
                        mean_main_by_layer=never_prev_active_main_by_layer_k,
                        std_main_by_layer=never_prev_active_main_std_by_layer_k,
                        mean_inlay_metric1_by_layer=always_inactive_never_prev_active_by_layer_k,
                        std_inlay_metric1_by_layer=always_inactive_never_prev_active_std_by_layer_k,
                        mean_inlay_metric2_by_layer=always_negative_never_prev_active_by_layer_k,
                        std_inlay_metric2_by_layer=always_negative_never_prev_active_std_by_layer_k,
                        main_label="never previously active among lost",
                        inlay_metric1_label="always inactive",
                        inlay_metric2_label="always negative",
                        title=f"Never previously active lost features (k={k})",
                        ylabel="rate",
                        save_path=never_prev_active_inlay_plot_path,
                        show_error_bars=True,
                        main_color="darkorange",
                        inlay_metric1_color="red",
                        inlay_metric2_color="cyan",
                    )
                    wandb.log({
                        f"{metric_namespace}/{setup_name}/layer_vs_never_prev_active_with_inlay_k_{k}": wandb.Image(str(never_prev_active_inlay_plot_path)),
                    })

                prev_active_inlay_mean_over_k_plot_path = output_dir / f"{setup_name}__layer_vs_prev_active_with_inlay_mean_over_k.png"
                plot_main_metric_with_inlay(
                    layers=plot_layers,
                    mean_main_by_layer=mean_prev_active_among_lost_by_layer,
                    std_main_by_layer=std_prev_active_among_lost_by_layer,
                    mean_inlay_metric1_by_layer=mean_inactive_last_among_prev_active_lost_by_layer,
                    std_inlay_metric1_by_layer=std_inactive_last_among_prev_active_lost_by_layer,
                    mean_inlay_metric2_by_layer=mean_negative_last_among_prev_active_lost_by_layer,
                    std_inlay_metric2_by_layer=std_negative_last_among_prev_active_lost_by_layer,
                    main_label="previously active among lost",
                    inlay_metric1_label="inactive at last",
                    inlay_metric2_label="negative at last",
                    title="Previously active lost features",
                    ylabel="rate",
                    save_path=prev_active_inlay_mean_over_k_plot_path,
                    show_error_bars=True,
                    main_color="royalblue",
                    inlay_metric1_color="red",
                    inlay_metric2_color="cyan",
                )
                wandb.log({
                    f"{metric_namespace}/{setup_name}/layer_vs_prev_active_with_inlay_mean_over_k": wandb.Image(str(prev_active_inlay_mean_over_k_plot_path)),
                })

                never_prev_active_inlay_mean_over_k_plot_path = output_dir / f"{setup_name}__layer_vs_never_prev_active_with_inlay_mean_over_k.png"
                plot_main_metric_with_inlay(
                    layers=plot_layers,
                    mean_main_by_layer=mean_never_prev_active_among_lost_by_layer,
                    std_main_by_layer=std_never_prev_active_among_lost_by_layer,
                    mean_inlay_metric1_by_layer=mean_always_inactive_among_never_prev_active_lost_by_layer,
                    std_inlay_metric1_by_layer=std_always_inactive_among_never_prev_active_lost_by_layer,
                    mean_inlay_metric2_by_layer=mean_always_negative_among_never_prev_active_lost_by_layer,
                    std_inlay_metric2_by_layer=std_always_negative_among_never_prev_active_lost_by_layer,
                    main_label="never previously active among lost",
                    inlay_metric1_label="always inactive",
                    inlay_metric2_label="always negative",
                    title="Never previously active lost features",
                    ylabel="rate",
                    save_path=never_prev_active_inlay_mean_over_k_plot_path,
                    show_error_bars=True,
                    main_color="darkorange",
                    inlay_metric1_color="red",
                    inlay_metric2_color="cyan",
                )
                wandb.log({
                    f"{metric_namespace}/{setup_name}/layer_vs_never_prev_active_with_inlay_mean_over_k": wandb.Image(str(never_prev_active_inlay_mean_over_k_plot_path)),
                })

                prev_active_raw_df = _tensor_metric_dicts_to_df(prev_active_among_lost_tensor_by_layer_by_k, "prev_active_among_lost", setup_name)
                never_prev_active_raw_df = _tensor_metric_dicts_to_df(never_prev_active_among_lost_tensor_by_layer_by_k, "never_prev_active_among_lost", setup_name)
                inactive_last_prev_active_raw_df = _tensor_metric_dicts_to_df(inactive_last_among_prev_active_lost_tensor_by_layer_by_k, "inactive_last_among_prev_active_lost", setup_name)
                negative_last_prev_active_raw_df = _tensor_metric_dicts_to_df(negative_last_among_prev_active_lost_tensor_by_layer_by_k, "negative_last_among_prev_active_lost", setup_name)
                always_inactive_never_prev_active_raw_df = _tensor_metric_dicts_to_df(always_inactive_among_never_prev_active_lost_tensor_by_layer_by_k, "always_inactive_among_never_prev_active_lost", setup_name)
                always_negative_never_prev_active_raw_df = _tensor_metric_dicts_to_df(always_negative_among_never_prev_active_lost_tensor_by_layer_by_k, "always_negative_among_never_prev_active_lost", setup_name)

                neg_raw_df = _tensor_metric_dicts_to_df(neg_tensor_by_layer_by_k, "neg_rate_among_lost", setup_name)
                zero_raw_df = _tensor_metric_dicts_to_df(zero_tensor_by_layer_by_k, "zero_rate_among_lost", setup_name)

                raw_df = pd.concat([
                    neg_raw_df,
                    zero_raw_df,
                    prev_active_raw_df,
                    never_prev_active_raw_df,
                    inactive_last_prev_active_raw_df,
                    negative_last_prev_active_raw_df,
                    always_inactive_never_prev_active_raw_df,
                    always_negative_never_prev_active_raw_df,
                ], ignore_index=True)
                raw_csv_path = output_dir / f"{setup_name}__raw_metrics_by_layer_k.csv"
                raw_df.to_csv(raw_csv_path, index=False)
                wandb.log({
                    f"{metric_namespace}/{setup_name}/raw_metrics": wandb.Table(dataframe=raw_df),
                })

        except Exception as e:
            print(f"[{case_name}] failed, skipping to next data case: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    run.finish()


if __name__ == "__main__":
    main()