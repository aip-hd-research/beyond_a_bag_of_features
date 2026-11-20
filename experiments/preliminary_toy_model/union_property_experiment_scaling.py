"""
Union Property Scaling Experiment  (high-D, large-N)

Same hypothesis as the D=50 baseline, but at scales where the oversampled regime
(SAE width >> N) is actually reachable:

Setup:
- D specified via --d (default 1000)
- N in {100, 300, 500, 1000, 5000} orthogonal true features
- SAE widths sweep from clearly undersampled to clearly oversampled relative to N
- Pair evaluation is capped at --max-pairs (default 5000) random pairs per (SAE, N)
  to keep evaluation tractable for large N

Usage:
  python union_property_experiment_scaling.py --d 1000 --p 0.2 --training-tokens 50000000
  python union_property_experiment_scaling.py --d 1000 --p 0.2 --max-pairs 2000
  CUDA_VISIBLE_DEVICES=1 python union_property_experiment_scaling.py --d 1000 --p 0.2
"""

import argparse
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from hedging_paper.evals.union_property_eval import evaluate_union_property
from hedging_paper.saes.base_sae import BaseSAE, BaseSAEConfig, BaseSAERunnerConfig
from hedging_paper.toy_models.get_training_batch import get_training_batch

import hedging_paper.toy_models.plotting as _plotting
from hedging_paper.toy_models.plotting import SEABORN_RC_CONTEXT

_plotting.SEABORN_RC_CONTEXT["text.usetex"] = False
RC_CONTEXT = {**SEABORN_RC_CONTEXT, "text.usetex": False}

from hedging_paper.toy_models.toy_model import ToyModel
from hedging_paper.toy_models.train_toy_sae import train_toy_sae
from hedging_paper.util import DEFAULT_DEVICE

tqdm._instances.clear()  # type: ignore
plt.rcParams.update({"figure.dpi": 150})

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, default=1000, help="Hidden/activation dimension")
parser.add_argument("--p", type=float, default=0.2, help="Feature firing probability")
parser.add_argument("--training-tokens", type=int, default=50_000_000)
parser.add_argument("--max-pairs", type=int, default=5000,
                    help="Max pairs evaluated per (SAE, N); None = all pairs")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────

HIDDEN_DIM      = args.d
FIRING_PROB     = args.p
L1_COEFF        = 1e-2
TRAINING_TOKENS = args.training_tokens
MAX_PAIRS       = args.max_pairs
SEEDS           = [0, 1, 2]
ALL_N           = [100, 300, 500, 1000, 5000]

# Widths span from undersampled (width << min(N)) to oversampled (width >> max(N)).
WIDTHS        = [20, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 5000, 10000]
LABELED_TICKS = {50, 100, 200, 500, 1000, 2000, 5000, 10000}

p_tag     = f"p{int(FIRING_PROB * 100):03d}"
SAE_ROOT  = Path(f"results/preliminary_toy_model/saes/union_property_scaling_D{HIDDEN_DIM}_{p_tag}")
PLOT_ROOT = Path(f"results/preliminary_toy_model/plots/union_property_scaling_D{HIDDEN_DIM}_{p_tag}")

# ── Helper: save a set of plots for one N ─────────────────────────────────────

def save_plots(df: pd.DataFrame, num_features: int, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    df_metrics = df.melt(
        id_vars=["seed", "width"],
        value_vars=["jaccard", "recall", "precision"],
        var_name="metric",
        value_name="value",
    )
    with plt.rc_context(RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.lineplot(data=df_metrics, x="width", y="value", hue="metric",
                     ax=ax, errorbar="sd", marker="o", markersize=4)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(num_features, color="red", linestyle=":", linewidth=0.8)
        ax.set_xlabel("SAE width")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log")
        ax.set_xticks([w for w in WIDTHS if w in LABELED_TICKS])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plot_dir / "metrics_vs_width.pdf")
        plt.savefig(plot_dir / "metrics_vs_width.png")
        plt.close()

    with plt.rc_context(RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.lineplot(data=df, x="width", y="jaccard",
                     ax=ax, errorbar="sd", marker="o", markersize=4)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(num_features, color="red", linestyle=":", linewidth=0.8)
        ax.set_xlabel("SAE width")
        ax.set_ylabel("Union Jaccard")
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log")
        ax.set_xticks([w for w in WIDTHS if w in LABELED_TICKS])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(plot_dir / "jaccard_vs_width.pdf")
        plt.savefig(plot_dir / "jaccard_vs_width.png")
        plt.close()

    df_melt = df.melt(
        id_vars=["seed", "width"],
        value_vars=["extra", "lost"],
        var_name="type",
        value_name="count",
    )
    with plt.rc_context(RC_CONTEXT):
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.lineplot(data=df_melt, x="width", y="count", hue="type",
                     ax=ax, errorbar="sd", marker="o", markersize=4)
        ax.axvline(num_features, color="red", linestyle=":", linewidth=0.8)
        ax.set_xlabel("SAE width")
        ax.set_ylabel("Mean latents per pair")
        ax.set_xscale("log")
        ax.set_xticks([w for w in WIDTHS if w in LABELED_TICKS])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plot_dir / "extra_vs_lost_vs_width.pdf")
        plt.savefig(plot_dir / "extra_vs_lost_vs_width.png")
        plt.close()

    print(f"  Saved plots to {plot_dir}")


# ── Main loop over N ──────────────────────────────────────────────────────────

for NUM_FEATURES in ALL_N:
    print(f"\n{'='*60}")
    print(f"  N = {NUM_FEATURES} features  (D={HIDDEN_DIM}, p={FIRING_PROB}, max_pairs={MAX_PAIRS})")
    print(f"{'='*60}")

    base_dir = SAE_ROOT / f"N_{NUM_FEATURES}"
    base_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = PLOT_ROOT / f"N_{NUM_FEATURES}"

    toy_model_path = base_dir / "toy_model.pt"
    if toy_model_path.exists():
        toy_model = torch.load(toy_model_path, weights_only=False).to(DEFAULT_DEVICE)
        print("Loaded toy model from disk")
    else:
        toy_model = ToyModel(num_feats=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(DEFAULT_DEVICE)
        torch.save(toy_model, toy_model_path)
        print("Created and saved toy model")

    firing_probs = torch.full((NUM_FEATURES,), FIRING_PROB)
    generate_batch = partial(get_training_batch, firing_probabilities=firing_probs)

    sample = generate_batch(50_000)
    true_l0 = (sample > 0).float().sum(dim=-1).mean()
    print(f"True L0: {true_l0:.2f}  (expected ~{NUM_FEATURES * FIRING_PROB:.2f})")

    for seed in SEEDS:
        torch.manual_seed(seed)
        for width in WIDTHS:
            sae_path = base_dir / f"seed_{seed}/width_{width}"
            if sae_path.exists():
                print(f"  seed={seed} width={width}: already trained, skipping")
                continue

            print(f"  seed={seed} width={width}: training...")
            cfg = BaseSAERunnerConfig(
                context_size=500,
                d_in=HIDDEN_DIM,
                d_sae=width,
                l1_coefficient=L1_COEFF,
                normalize_sae_decoder=False,
                scale_sparsity_penalty_by_decoder_norm=True,
                init_encoder_as_decoder_transpose=True,
                apply_b_dec_to_input=True,
                b_dec_init_method="zeros",
            )
            sae = BaseSAE(BaseSAEConfig.from_sae_runner_config(cfg))
            train_toy_sae(
                sae,
                toy_model,
                generate_batch,
                training_tokens=TRAINING_TOKENS,
                device=DEFAULT_DEVICE,
            )
            sae.save_model_final(sae_path)
            print(f"  seed={seed} width={width}: saved to {sae_path}")

    print("Training complete.")

    records = []
    for seed in SEEDS:
        for width in WIDTHS:
            sae_path = base_dir / f"seed_{seed}/width_{width}"
            sae = BaseSAE.load_from_disk(str(sae_path), device=str(DEFAULT_DEVICE))
            sae.eval()

            metrics = evaluate_union_property(
                sae=sae,
                toy_model=toy_model,
                num_features=NUM_FEATURES,
                active_threshold=0.0,
                max_pairs=MAX_PAIRS,
                seed=seed,
            )
            records.append({"seed": seed, "width": width, **metrics})
            print(
                f"  seed={seed} width={width:5d}: "
                f"jaccard={metrics['jaccard']:.3f}  "
                f"recall={metrics['recall']:.3f}  "
                f"precision={metrics['precision']:.3f}  "
                f"pairs={metrics['num_pairs']}"
            )

    df = pd.DataFrame(records)
    save_plots(df, NUM_FEATURES, plot_dir)

    summary = (
        df.groupby("width")[["jaccard", "recall", "precision", "extra", "lost"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print(summary.to_string())

print("\nAll done.")
