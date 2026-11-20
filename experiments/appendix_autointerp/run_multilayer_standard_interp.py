#!/usr/bin/env python3
"""
Multi-layer standard auto-interpretability: lost vs control vs surviving.

Hydra-based, sweep-compatible via spawn_sweep_master.py.
Uses noun-union signatures with PAPER_TEMPLATE for feature identification.
Evaluates only at sae.interpretability_layers (or all sae.layers if not set).

Run from the repo root:
    pixi run -e autointerp python experiments/appendix_autointerp/run_multilayer_standard_interp.py \
        +appendix_autointerp=base sae=gemma-3-270m/16k_medium

Requires OPENAI_API_KEY in the environment.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import hydra
import wandb
from omegaconf import DictConfig, MISSING, OmegaConf
from hydra.core.config_store import ConfigStore

REPO_ROOT = Path(__file__).resolve().parents[1]

from sae_set_behaviour.util import SetupConfig, WandbConfig, init_wandb
from sae_set_behaviour.lost_latents.construct_text_sequence import (
    build_pairs,
    read_noun_attribute_rows,
    PAPER_TEMPLATE,
)
from sae_set_behaviour.lost_latents.metrics import compute_noun_union_signature
from sae_set_behaviour.lost_latents.construct_embedding import collect_noun_word_activations


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config(SetupConfig):
    base_save_dir: str = "results/appendix_autointerp"
    wandb: WandbConfig = MISSING
    data_csv: str = "data/data.csv"
    scratch_dir: str = "results/appendix_autointerp/scratch"
    n_eval: int = 100
    buffer: int = 3
    total_tokens: int = 500_000
    text_template: str = PAPER_TEMPLATE
    k_attr: int = 1
    eps: float = 0.01
    max_length: int = 32
    experiment_batch_size: int = 32
    seed: int = 42
    sae_device: str = ""  # if set, SAE runs on this device (e.g. cuda:1); otherwise falls back to args.device


ConfigStore.instance().store(name="config", node=Config)

GROUPS = ["lost", "control", "surviving"]
COLORS = {"lost": "C0", "control": "C1", "surviving": "C2"}
LABELS = {"lost": "Lost", "control": "Control (random)", "surviving": "Surviving"}


# ── Path helpers ──────────────────────────────────────────────────────────────

def _sae_slug(args: DictConfig) -> str:
    """Stable filesystem key for this SAE release × id_template."""
    id_tmpl = str(getattr(args.sae, "id_template", None) or "unknown")
    return f"{args.sae.release}__{id_tmpl.replace('{}', 'L')}"


def _out_dir(args: DictConfig) -> Path:
    return Path(args.base_save_dir) / _sae_slug(args)


def _idx_dir(args: DictConfig) -> Path:
    return Path(args.scratch_dir) / "interp_indices" / _sae_slug(args)


def _result_path(args: DictConfig, layer: int, group: str) -> Path:
    return _out_dir(args) / f"layer_{layer}" / f"{group}_features.json"


def _sae_id(args: DictConfig, layer: int) -> str:
    return args.sae.id_template.format(layer)


def _hook_device(model, layer: int) -> torch.device:
    """Return the device that layer `layer` lives on (handles n_devices splits)."""
    try:
        return next(model.blocks[layer].parameters()).device
    except (StopIteration, IndexError):
        return next(model.parameters()).device


def get_interpretability_layers(args: DictConfig) -> List[int]:
    interp = args.sae.interpretability_layers
    if interp is not None:
        return [int(l) for l in interp]
    return [int(l) for l in args.sae.layers]


# ── Results I/O ───────────────────────────────────────────────────────────────

def save_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scores = [v["score"] for v in results.values() if "score" in v]
    payload = {
        "stats": {
            "n_features": len(scores),
            "score_mean": float(sum(scores) / len(scores)) if scores else float("nan"),
            "score_std": float(torch.tensor(scores).std().item()) if len(scores) > 1 else float("nan"),
        },
        "per_feature": {
            str(k): {"score": v.get("score"), "explanation": v.get("explanation")}
            for k, v in results.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved: {path}")


def load_stats(path: Path) -> Optional[dict]:
    return json.loads(path.read_text())["stats"] if path.exists() else None


# ── Identify phase ────────────────────────────────────────────────────────────

def run_identify(args: DictConfig, layers: List[int]) -> None:
    """
    Classify SAE features as lost / control / surviving using noun-union signatures.

    Signatures are the union of active latents over all noun-word tokens under
    PAPER_TEMPLATE ("This is a {sequence}. "), matching the experiment_1 strategy.
    """
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    _p = Path(args.data_csv)
    csv_path = _p if _p.is_absolute() else REPO_ROOT / _p
    rows = read_noun_attribute_rows(csv_path)
    pairs = build_pairs(rows, "A_fixed_base", k=args.k_attr, text_template=args.text_template)

    texts_base = [p.base_text for p in pairs]
    texts_comp = [p.composed_text for p in pairs]
    nouns_base = [p.noun_for_base for p in pairs]
    nouns_comp = [p.noun_for_composed for p in pairs]

    n_devices  = int(getattr(args.sae.model, "n_devices",              1)     or 1)
    model_dtype = str(getattr(args.sae.model, "dtype",            "bfloat16") or "bfloat16")
    fold_ln                = bool(getattr(args.sae.model, "fold_ln",                True))
    center_writing_weights = bool(getattr(args.sae.model, "center_writing_weights", False))
    center_unembed         = bool(getattr(args.sae.model, "center_unembed",         False))

    model = HookedTransformer.from_pretrained(
        args.sae.model.name,
        device=args.device,
        dtype=model_dtype,
        n_devices=n_devices,
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        center_unembed=center_unembed,
    )
    model.eval()

    for layer in layers:
        idir = _idx_dir(args) / f"layer_{layer}"
        lost_f = idir / "lost_indices.txt"
        ctrl_f = idir / "control_indices.txt"
        surv_f = idir / "surviving_indices.txt"

        if lost_f.exists() and ctrl_f.exists() and surv_f.exists():
            print(f"Layer {layer}: index files exist, skipping identify.")
            continue

        print(f"\n{'='*60}\nLayer {layer}: identifying lost/control/surviving\n{'='*60}")

        sae = SAE.from_pretrained(
            release=args.sae.release,
            sae_id=_sae_id(args, layer),
            device="cpu",
        )
        sae_dev = str(args.sae_device or args.device)
        try:
            sae = sae.to(sae_dev)
        except torch.OutOfMemoryError:
            print(f"  Warning: OOM moving SAE to {sae_dev} — reloading on CPU.")
            del sae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sae = SAE.from_pretrained(
                release=args.sae.release,
                sae_id=_sae_id(args, layer),
                device="cpu",
            )
            sae_dev = "cpu"
        sae.eval()

        hook_name = sae.cfg.metadata.hook_name
        d_sae = int(sae.W_enc.shape[1])

        word_acts_b, word_mask_b = collect_noun_word_activations(
            model=model, texts=texts_base, nouns=nouns_base,
            hook_name=hook_name, device=args.device,
            max_length=args.max_length, batch_size=args.experiment_batch_size,
        )
        sig_b = compute_noun_union_signature(sae, word_acts_b, word_mask_b, sae_device=sae_dev)
        del word_acts_b, word_mask_b

        word_acts_c, word_mask_c = collect_noun_word_activations(
            model=model, texts=texts_comp, nouns=nouns_comp,
            hook_name=hook_name, device=args.device,
            max_length=args.max_length, batch_size=args.experiment_batch_size,
        )
        sig_c = compute_noun_union_signature(sae, word_acts_c, word_mask_c, sae_device=sae_dev)
        del word_acts_c, word_mask_c

        lost_count = (sig_b & ~sig_c).sum(dim=0).cpu().long()
        active_count = sig_b.sum(dim=0).cpu().long()
        del sig_b, sig_c
        sae.to("cpu")
        del sae

        lost_pool = torch.where(lost_count >= 1)[0]
        surv_pool = torch.where((active_count >= 1) & (lost_count == 0))[0]
        n_buf = args.n_eval * args.buffer

        def _sample(pool: torch.Tensor, seed_offset: int) -> List[int]:
            rng = torch.Generator()
            rng.manual_seed(args.seed + seed_offset)
            n = min(len(pool), n_buf)
            perm = torch.randperm(len(pool), generator=rng)
            return sorted(pool[perm[:n]].tolist())

        lost_idx = _sample(lost_pool, 0)
        ctrl_idx = _sample(torch.arange(d_sae), 1)
        surv_idx = _sample(surv_pool, 2)

        idir.mkdir(parents=True, exist_ok=True)
        for f, idx in [(lost_f, lost_idx), (ctrl_f, ctrl_idx), (surv_f, surv_idx)]:
            f.write_text("\n".join(str(i) for i in idx) + "\n")
        print(f"  {len(lost_idx)} lost / {len(ctrl_idx)} control / {len(surv_idx)} surviving")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Interp phase ──────────────────────────────────────────────────────────────

def run_interp(args: DictConfig, layers: List[int], api_key: str) -> None:
    import sae_bench.sae_bench_utils.activation_collection as activation_collection
    import sae_bench.sae_bench_utils.general_utils as general_utils
    from sae_bench.evals.autointerp.main import run_eval_single_sae
    from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
    from transformer_lens import HookedTransformer

    model_name = args.sae.model.name
    llm_dtype_str = str(getattr(args.sae.model, "dtype", "bfloat16") or "bfloat16")
    llm_dtype = general_utils.str_to_dtype(llm_dtype_str)
    llm_bs = activation_collection.LLM_NAME_TO_BATCH_SIZE.get(model_name, 8)
    artifacts = str(Path(args.scratch_dir) / "saebench_artifacts")

    _device = general_utils.setup_environment()
    n_devices = int(getattr(args.sae.model, "n_devices", 1) or 1)

    model: HookedTransformer = HookedTransformer.from_pretrained_no_processing(
        model_name, device=_device, dtype=llm_dtype, n_devices=n_devices,
    )

    for layer in layers:
        sid = _sae_id(args, layer)
        print(f"\n{'='*60}\nLayer {layer}  ({sid})\n{'='*60}")

        sae_dev = _hook_device(model, layer)
        _, sae, sparsity = general_utils.load_and_format_sae(args.sae.release, sid, "cpu")
        try:
            sae = sae.to(device=sae_dev, dtype=llm_dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"  Warning: OOM moving SAE to {sae_dev} — running SAE on CPU.")
            sae = sae.to(dtype=llm_dtype)
            sae_dev = torch.device("cpu")

        idir = _idx_dir(args) / f"layer_{layer}"
        candidates: Dict[str, List[int]] = {}
        for g in GROUPS:
            f = idir / f"{g}_indices.txt"
            if not f.exists():
                raise FileNotFoundError(f"Missing {f} — run identify first.")
            candidates[g] = [int(line) for line in f.read_text().splitlines() if line.strip()][:args.n_eval]

        for group in GROUPS:
            out = _result_path(args, layer, group)
            if out.exists():
                print(f"  Layer {layer}/{group}: done, skipping.")
                continue
            print(f"\n  --- {group} ---")
            cfg = AutoInterpEvalConfig(
                model_name=model_name,
                n_latents=None,
                override_latents=candidates[group],
                total_tokens=args.total_tokens,
                llm_dtype=llm_dtype_str,
                llm_batch_size=llm_bs,
            )
            results = run_eval_single_sae(
                config=cfg, sae=sae, model=model, device=sae_dev,
                artifacts_folder=artifacts, api_key=api_key, sae_sparsity=sparsity,
            )
            save_results(out, results)
            if wandb.run is not None:
                stats = load_stats(out)
                if stats:
                    wandb.log({
                        f"autointerp/{group}/layer_{layer}/score_mean": stats["score_mean"],
                        f"autointerp/{group}/layer_{layer}/score_std": stats["score_std"],
                        f"autointerp/{group}/layer_{layer}/n_features": stats["n_features"],
                    })

        del sae, sparsity
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Plot + summary ────────────────────────────────────────────────────────────

def print_table(args: DictConfig, layers: List[int]) -> None:
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    header = f"{'Layer':>7}  " + "  ".join(f"{LABELS[g]:>22}" for g in GROUPS)
    print(header)
    print("-" * len(header))
    for layer in layers:
        row = f"{layer:>7}  "
        for group in GROUPS:
            s = load_stats(_result_path(args, layer, group))
            row += f"  {s['score_mean']:.4f} ± {s['score_std']:.4f}    " if s else f"  {'n/a':>22}  "
        print(row)


def plot_results(args: DictConfig, layers: List[int]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = _out_dir(args) / "multilayer_scores"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(layers))
    width = 0.25
    offsets = {"lost": -width, "control": 0.0, "surviving": width}

    for group in GROUPS:
        means, stds = [], []
        for layer in layers:
            s = load_stats(_result_path(args, layer, group))
            means.append(s["score_mean"] if s else float("nan"))
            stds.append(s["score_std"] if s else 0.0)
        ax.bar(
            x + offsets[group], means, width, yerr=stds,
            label=LABELS[group], color=COLORS[group], capsize=4, alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Autointerp score (mean ± std)")
    ax.set_title(f"Standard autointerp by layer\n{args.sae.model.name}, {args.sae.release}")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()

    for fmt in ("pdf", "png"):
        p = out_path.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=200 if fmt == "png" else None)
        print(f"Saved: {p}")
    plt.close(fig)

    if wandb.run is not None:
        wandb.log({"multilayer_scores": wandb.Image(str(out_path.with_suffix(".png")))})


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_name="config", config_path="../../configs")
def main(args: DictConfig) -> None:
    init_wandb(args.wandb, args)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable")

    layers = get_interpretability_layers(args)
    print(f"Model: {args.sae.model.name}  |  Release: {args.sae.release}")
    print(f"Interpretability layers: {layers}")

    run_identify(args, layers)
    run_interp(args, layers, api_key)
    print_table(args, layers)
    plot_results(args, layers)

    if wandb.run is not None:
        summary = {}
        for group in GROUPS:
            scores = []
            for layer in layers:
                s = load_stats(_result_path(args, layer, group))
                if s:
                    summary[f"{group}/layer_{layer}/score_mean"] = s["score_mean"]
                    summary[f"{group}/layer_{layer}/score_std"] = s["score_std"]
                    scores.append(s["score_mean"])
            if scores:
                summary[f"{group}/mean_across_layers"] = sum(scores) / len(scores)
        wandb.run.summary.update(summary)

    wandb.finish()


if __name__ == "__main__":
    main()
