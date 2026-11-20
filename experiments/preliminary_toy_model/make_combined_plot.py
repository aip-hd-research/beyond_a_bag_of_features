"""
Combined 3-panel figure: union property metrics.
Loads cached SAEs, re-runs evaluation, saves figure.

Usage:
  python make_combined_plot.py                          # defaults: D=50, p=0.5, N=10,50,100
  python make_combined_plot.py --p 0.2
  python make_combined_plot.py --d 500 --ns 500 1000 5000
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from hedging_paper.evals.union_property_eval import evaluate_union_property
from hedging_paper.saes.base_sae import BaseSAE
from hedging_paper.toy_models.toy_model import ToyModel
import hedging_paper.toy_models.plotting as _plotting
from hedging_paper.toy_models.plotting import SEABORN_RC_CONTEXT
from hedging_paper.util import DEFAULT_DEVICE

tqdm._instances.clear()  # type: ignore
_plotting.SEABORN_RC_CONTEXT["text.usetex"] = False
RC_CONTEXT = {**SEABORN_RC_CONTEXT, "text.usetex": False}

parser = argparse.ArgumentParser()
parser.add_argument("--p", type=float, default=0.5, help="Feature firing probability")
parser.add_argument("--d", type=int, default=50, help="Hidden/activation dimension")
parser.add_argument("--ns", type=int, nargs=3, default=[10, 50, 100],
                    metavar=("N1", "N2", "N3"), help="Three N values for panels a/b/c")
parser.add_argument("--max-pairs", type=int, default=5000,
                    help="Max pairs per SAE eval when C(N,2) exceeds this (default 5000)")
parser.add_argument("--sae-root", type=str, default=None,
                    help="Override SAE root directory (auto-detected if omitted)")
parser.add_argument("--plot-root", type=str, default=None,
                    help="Override plot output directory (auto-detected if omitted)")
parser.add_argument("--log-x", action="store_true",
                    help="Log-scale x-axis (use for scaling experiments with wide width range)")
parser.add_argument("--out-prefix", type=str, default="combined",
                    help="Output filename prefix (default: 'combined')")
args = parser.parse_args()

p_tag     = f"p{int(args.p * 100):03d}"
_d_prefix = "" if args.d == 50 else f"D{args.d}_"
SAE_ROOT  = Path(args.sae_root) if args.sae_root else \
            Path(f"results/preliminary_toy_model/saes/union_property_{_d_prefix}{p_tag}")
PLOT_ROOT = Path(args.plot_root) if args.plot_root else \
            Path(f"results/preliminary_toy_model/plots/union_property_{_d_prefix}{p_tag}")

# Auto-detect widths from the first available N directory on disk
def _detect_widths(sae_root: Path, ns: list[int]) -> list[int]:
    for n in ns:
        seed0 = sae_root / f"N_{n}/seed_0"
        if seed0.exists():
            ws = sorted(int(p.name.split("_")[1]) for p in seed0.iterdir()
                        if p.is_dir() and p.name.startswith("width_"))
            if ws:
                return ws
    return [8, 10, 12, 15, 20, 30, 40, 50, 70, 100]  # fallback

WIDTHS        = _detect_widths(SAE_ROOT, args.ns)
LABELED_TICKS = set(WIDTHS[::max(1, len(WIDTHS) // 7)])  # ~7 evenly spaced ticks

SEEDS        = [0, 1, 2]
PANEL_NS     = args.ns
PANEL_LABELS = ["a)", "b)", "c)"]

PLOT_ROOT.mkdir(parents=True, exist_ok=True)

# ── Collect data ──────────────────────────────────────────────────────────────

all_data: dict[int, pd.DataFrame] = {}

for N in PANEL_NS:
    base_dir = SAE_ROOT / f"N_{N}"
    toy_model = torch.load(base_dir / "toy_model.pt", weights_only=False)
    toy_model = toy_model.to(DEFAULT_DEVICE)
    toy_model.eval()

    records = []
    for seed in SEEDS:
        for width in WIDTHS:
            sae_path = base_dir / f"seed_{seed}/width_{width}"
            sae = BaseSAE.load_from_disk(str(sae_path), device=str(DEFAULT_DEVICE))
            sae.eval()
            total_pairs = N * (N - 1) // 2
            mp = args.max_pairs if total_pairs > args.max_pairs else None
            metrics = evaluate_union_property(
                sae=sae,
                toy_model=toy_model,
                num_features=N,
                active_threshold=0.0,
                max_pairs=mp,
            )
            records.append({"seed": seed, "width": width, **metrics})
            print(f"  N={N:3d} seed={seed} width={width:3d}: jaccard={metrics['jaccard']:.3f}")

    all_data[N] = pd.DataFrame(records)

# ── Helpers ───────────────────────────────────────────────────────────────────

metric_palette = {"Jaccard": "#1f77b4", "Recall": "#ff7f0e", "Precision": "#2ca02c"}


def _sci_fmt(x, _):
    if x < 1000:
        return str(int(x))
    exp = int(round(len(str(int(x))) - 1))
    coef = x / 10 ** exp
    coef_str = f"{coef:g}" if abs(coef - round(coef)) > 1e-9 else str(int(round(coef)))  # 1500 -> "1.5e3", not "2e3"
    return f"{coef_str}e{exp}"


def _stagger_crowded_ticks(ax, min_frac: float = 0.045, extra_pad: float = 8.0) -> None:
    """Shift every second x tick label of a crowded run down so neighbouring labels don't overlap."""
    locs = list(ax.get_xticks())
    to_axes = ax.transAxes.inverted()
    xs = [to_axes.transform(ax.transData.transform((v, 0)))[0] for v in locs]
    last = None
    for tick, x in sorted(zip(ax.xaxis.get_major_ticks(), xs), key=lambda p: p[1]):
        if last is not None and x - last < min_frac:
            tick.set_pad(tick.get_pad() + extra_pad)
            last = None
        else:
            last = x


# ── Build figure ──────────────────────────────────────────────────────────────

with plt.rc_context(RC_CONTEXT):
    fig, axes = plt.subplots(
        nrows=3, ncols=1,
        figsize=(5.8, 6.2),
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.12)

    for ax, label, N in zip(axes, PANEL_LABELS, PANEL_NS):
        df = all_data[N]
        df_m = df.melt(
            id_vars=["seed", "width"],
            value_vars=["jaccard", "recall", "precision"],
            var_name="metric",
            value_name="value",
        )
        df_m["metric"] = df_m["metric"].str.capitalize()

        sns.lineplot(
            data=df_m,
            x="width",
            y="value",
            hue="metric",
            style="metric",
            dashes={"Jaccard": (1, 0), "Recall": (5, 2), "Precision": (2, 1.5)},
            palette=metric_palette,
            ax=ax,
            errorbar="sd",
            marker="o",
            markersize=3.5,
            linewidth=1.2,
        )

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, zorder=0)
        ax.axvline(N, color="red", linestyle=":", linewidth=0.9, zorder=0)

        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Score", fontsize=9.5)
        ax.tick_params(axis="both", labelsize=8.5)
        if args.log_x:
            ax.set_xscale("log")
            ax.set_xticks([w for w in WIDTHS if w in LABELED_TICKS])
            ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(_sci_fmt))
        else:
            ax.set_xticks([w for w in WIDTHS if w in LABELED_TICKS])
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

        # panel label (top-left) + N annotation (bottom-left)
        ax.text(0.02, 0.93, label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")
        ax.text(0.02, 0.07, f"N = {N}", transform=ax.transAxes,
                fontsize=8, ha="left", va="bottom", color="gray")

        # legend only on top panel
        if ax is axes[0]:
            ax.legend(
                fontsize=8,
                loc="lower right",
                ncol=3,
                framealpha=0.7,
                handlelength=1.5,
            )
        else:
            ax.get_legend().remove()

        ax.set_xlabel("")

    axes[-1].set_xlabel("SAE width", fontsize=9.5)
    _stagger_crowded_ticks(axes[-1])

    ns_tag  = "_".join(str(n) for n in PANEL_NS)
    out_pdf = PLOT_ROOT / f"{args.out_prefix}_N{ns_tag}.pdf"
    out_png = PLOT_ROOT / f"{args.out_prefix}_N{ns_tag}.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()

print(f"\nSaved: {out_pdf}")
print(f"Saved: {out_png}")

# ── Build figure 2: combined (seed + pair-sampling) error bars ─────────────────

def _combined_err_df(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    For each width, compute the grand mean and combined uncertainty:
      combined_std² = between-seed variance + mean(within-seed SE²)
    where within-seed SE² = (σ_pairs / √n_pairs)² for each seed.
    """
    rows = []
    for width, grp in df.groupby("width"):
        for metric in metrics:
            means = grp[metric].to_numpy(dtype=float)
            stds  = grp[f"{metric}_std"].to_numpy(dtype=float)
            ns    = grp["num_pairs"].to_numpy(dtype=float)
            grand_mean   = means.mean()
            between_var  = float(np.var(means, ddof=1)) if len(means) > 1 else 0.0
            within_var   = float((stds ** 2 / ns).mean())
            combined_std = (between_var + within_var) ** 0.5
            rows.append({"width": int(width), "metric": metric.capitalize(),
                         "mean": grand_mean, "err": combined_std})
    return pd.DataFrame(rows)


_LINESTYLES = {"Jaccard": "-", "Recall": (0, (5, 2)), "Precision": (0, (2, 1.5))}

with plt.rc_context(RC_CONTEXT):
    fig2, axes2 = plt.subplots(nrows=3, ncols=1, figsize=(5.8, 6.2), sharex=True)
    fig2.subplots_adjust(hspace=0.12)

    for ax, label, N in zip(axes2, PANEL_LABELS, PANEL_NS):
        df      = all_data[N]
        err_df  = _combined_err_df(df, ["jaccard", "recall", "precision"])

        for metric, color in metric_palette.items():
            sub = err_df[err_df["metric"] == metric].sort_values("width")
            ax.plot(sub["width"], sub["mean"],
                    color=color, marker="o", markersize=3.5, linewidth=1.2,
                    linestyle=_LINESTYLES[metric], label=metric)
            ax.fill_between(sub["width"],
                            (sub["mean"] - sub["err"]).clip(0, 1.08),
                            (sub["mean"] + sub["err"]).clip(0, 1.08),
                            alpha=0.18, color=color)

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, zorder=0)
        ax.axvline(N, color="red", linestyle=":", linewidth=0.9, zorder=0)

        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Score", fontsize=9.5)
        ax.tick_params(axis="both", labelsize=8.5)
        if args.log_x:
            ax.set_xscale("log")
            ax.set_xticks([w for w in WIDTHS if w in LABELED_TICKS])
            ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(_sci_fmt))
        else:
            ax.set_xticks([w for w in WIDTHS if w in LABELED_TICKS])
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

        ax.text(0.02, 0.93, label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")
        ax.text(0.02, 0.07, f"N = {N}", transform=ax.transAxes,
                fontsize=8, ha="left", va="bottom", color="gray")

        if ax is axes2[0]:
            ax.legend(fontsize=8, loc="lower right", ncol=3,
                      framealpha=0.7, handlelength=1.5)
        elif ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_xlabel("")

    axes2[-1].set_xlabel("SAE width", fontsize=9.5)
    _stagger_crowded_ticks(axes2[-1])

    out2_pdf = PLOT_ROOT / f"{args.out_prefix}_N{ns_tag}_combined_err.pdf"
    out2_png = PLOT_ROOT / f"{args.out_prefix}_N{ns_tag}_combined_err.png"
    plt.savefig(out2_pdf, bbox_inches="tight")
    plt.savefig(out2_png, bbox_inches="tight", dpi=200)
    plt.close()

print(f"Saved: {out2_pdf}")
print(f"Saved: {out2_png}")
