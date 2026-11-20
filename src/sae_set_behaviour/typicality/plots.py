from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

LABEL_FONTSIZE = 20
LEGEND_FONTSIZE = 16
TITLE_FONTSIZE = 16
TICK_FONTSIZE = 12

FIGSIZE_SINGLE = (4.5, 14.0)
FIGSIZE_LAYER = (16, 14.0)
FIGSIZE_SQUARE = (14, 14)

LEFT = 0.30
BOTTOM = 0.06
HEIGHT = 0.90
RIGHT_MARGIN = 0.04
CBAR_PAD = 0.02
CBAR_WIDTH = 0.025


def make_ordered_categories_for_plots(df_cat_typ_apm_by_layer: pd.DataFrame, layers: List[int], max_categories: int) -> List[str]:
    layer_cols_short = [f"layer_{l}" for l in layers]
    order_df = df_cat_typ_apm_by_layer.copy()
    order_df["mean_rho"] = order_df[layer_cols_short].mean(axis=1)
    ordered = order_df.sort_values("mean_rho", ascending=False)["category"].tolist()
    return ordered[:max_categories]


def _make_axes(figsize: Tuple[float, float]):
    fig = plt.figure(figsize=figsize)
    ax_width = 1.0 - LEFT - RIGHT_MARGIN - CBAR_PAD - CBAR_WIDTH
    ax = fig.add_axes([LEFT, BOTTOM, ax_width, HEIGHT])
    cax = fig.add_axes([LEFT + ax_width + CBAR_PAD, BOTTOM, CBAR_WIDTH, HEIGHT])
    return fig, ax, cax


def _select_evenly_spaced_ticks(labels: List[str], max_ticks: int = 6) -> Tuple[List[int], List[str]]:
    n = len(labels)
    if n <= max_ticks:
        idx = list(range(n))
    else:
        idx = np.linspace(0, n - 1, num=max_ticks)
        idx = np.round(idx).astype(int).tolist()
        idx = sorted(set(idx))
        if idx[0] != 0:
            idx = [0] + idx
        if idx[-1] != n - 1:
            idx = idx + [n - 1]
        idx = sorted(set(idx))
        if len(idx) > max_ticks:
            idx = idx[:max_ticks - 1] + [n - 1]
            idx = sorted(set(idx))
    tick_labels = [labels[i] for i in idx]
    return idx, tick_labels


def plot_heatmap(mat: np.ndarray, row_labels: List[str], col_labels: List[str], save_path: Path, title: str, figsize: Tuple[float, float], xlabel: str, vmin: float = -1.0, vmax: float = 1.0, max_xticks: int = 6):
    fig, ax, cax = _make_axes(figsize)

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"Spearman $\rho$", fontsize=LABEL_FONTSIZE, labelpad=12)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    tick_idx, tick_labels = _select_evenly_spaced_ticks(col_labels, max_ticks=max_xticks)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=TICK_FONTSIZE)

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=TICK_FONTSIZE)

    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Category", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close(fig)


def plot_layer_layer_heatmap(df_ll: pd.DataFrame, save_path: Path, vmin: float = -1.0, vmax: float = 1.0, max_xticks: int = 6):
    mat = df_ll.to_numpy(dtype=float)
    labels = df_ll.columns.tolist()

    mask_upper = np.triu(np.ones_like(mat, dtype=bool), k=1)
    mat_plot = mat.copy()
    mat_plot[mask_upper] = np.nan

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white")

    fig, ax, cax = _make_axes(FIGSIZE_SQUARE)

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(mat_plot, aspect="equal", cmap=cmap, norm=norm, interpolation="nearest")

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"Spearman $\rho$", fontsize=LABEL_FONTSIZE, labelpad=12)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    tick_idx, tick_labels = _select_evenly_spaced_ticks(labels, max_ticks=max_xticks)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=TICK_FONTSIZE)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels(tick_labels, fontsize=TICK_FONTSIZE)

    ax.set_xlabel(r"Layer $l_1$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"Layer $l_2$", fontsize=LABEL_FONTSIZE)
    ax.set_title(r"$\rho(APM(l_1),APM(l_2))$", fontsize=TITLE_FONTSIZE)

    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close(fig)


def plot_all_heatmaps(out_dir: Path, layers: List[int], df_cat_typ_cos: pd.DataFrame, df_cat_typ_apm_by_layer: pd.DataFrame, df_cat_cos_apm_by_layer: pd.DataFrame, df_ll: pd.DataFrame, max_categories: int) -> None:
    vmin, vmax = -1.0, 1.0

    ordered_categories = make_ordered_categories_for_plots(
        df_cat_typ_apm_by_layer=df_cat_typ_apm_by_layer,
        layers=layers,
        max_categories=max_categories,
    )
    layer_cols_short = [f"layer_{l}" for l in layers]
    layer_labels = [str(l) for l in layers]

    # Heatmap 1: typicality vs cosim
    df1 = df_cat_typ_cos.set_index("category").reindex(ordered_categories)
    mat_typ_cos = df1["spearman_rho"].to_numpy(dtype=float).reshape(-1, 1)
    plot_heatmap(
        mat=mat_typ_cos,
        row_labels=ordered_categories,
        col_labels=["typ↔cosim"],
        save_path=out_dir / "heatmap_typicality_vs_cosim.png",
        title=r"$\rho$(typicality, APC)",
        figsize=FIGSIZE_SINGLE,
        xlabel="",
        vmin=vmin,
        vmax=vmax,
        max_xticks=1,
    )

    # Heatmap 2: typicality vs APM by layer
    df2 = df_cat_typ_apm_by_layer.set_index("category").reindex(ordered_categories)
    mat_typ_apm = df2[layer_cols_short].to_numpy(dtype=float)
    plot_heatmap(
        mat=mat_typ_apm,
        row_labels=ordered_categories,
        col_labels=layer_labels,
        save_path=out_dir / "heatmap_typicality_vs_apm_by_layer.png",
        title=r"$\rho$(typicality, APM)",
        figsize=FIGSIZE_LAYER,
        xlabel=r"Layer $l$",
        vmin=vmin,
        vmax=vmax,
        max_xticks=6,
    )

    # Heatmap 3: cosim vs APM by layer
    df3 = df_cat_cos_apm_by_layer.set_index("category").reindex(ordered_categories)
    mat_cos_apm = df3[layer_cols_short].to_numpy(dtype=float)
    plot_heatmap(
        mat=mat_cos_apm,
        row_labels=ordered_categories,
        col_labels=layer_labels,
        save_path=out_dir / "heatmap_cosim_vs_apm_by_layer.png",
        title=r"$\rho$(APC, APM)",
        figsize=FIGSIZE_LAYER,
        xlabel=r"Layer $l$",
        vmin=vmin,
        vmax=vmax,
        max_xticks=6,
    )

    # Layer×Layer similarity heatmap
    plot_layer_layer_heatmap(
        df_ll=df_ll,
        save_path=out_dir / "heatmap_layer_layer_spearman_apm_scores.png",
        vmin=vmin,
        vmax=vmax,
        max_xticks=6,
    )


def plot_all_heatmaps_paper(
    out_dir: Path,
    layers: List[int],
    emb_jaccard_layers: List[int],
    df_cat_typ_cos: pd.DataFrame,
    df_cat_typ_ctx_by_layer: pd.DataFrame,
    df_cat_typ_jac_by_layer: pd.DataFrame,
    df_cat_typ_emb_jac_by_layer: pd.DataFrame,
    df_cat_emb_jac_by_layer: pd.DataFrame,
    df_cat_emb_emb_jac_by_layer: pd.DataFrame,
    df_cat_ctx_jac_by_layer: pd.DataFrame,
    df_cat_ctx_emb_jac_by_layer: pd.DataFrame,
    df_ll_jac: pd.DataFrame,
    df_ll_emb_jac: pd.DataFrame,
    df_cat_typ_ctx: pd.DataFrame,
    df_cat_typ_jac: pd.DataFrame,
    df_cat_typ_emb_jac: pd.DataFrame,
    max_categories: int,
) -> None:
    """Plot heatmaps for the paper-aligned metrics (contextual cosim + Jaccard)."""
    vmin, vmax = -1.0, 1.0
    layer_cols_short = [f"layer_{l}" for l in layers]
    layer_labels = [str(l) for l in layers]
    emb_jac_cols_short = [f"layer_{l}" for l in emb_jaccard_layers]
    emb_jac_labels = [str(l) for l in emb_jaccard_layers]

    # Order categories by mean typicality vs Jaccard rho
    order_df = df_cat_typ_jac_by_layer.copy()
    order_df["mean_rho"] = order_df[layer_cols_short].mean(axis=1)
    ordered_categories = (
        order_df.sort_values("mean_rho", ascending=False)["category"]
        .tolist()[:max_categories]
    )

    # Static embedding cosim (single column, same as existing)
    df1 = df_cat_typ_cos.set_index("category").reindex(ordered_categories)
    mat = df1["spearman_rho"].to_numpy(dtype=float).reshape(-1, 1)
    plot_heatmap(
        mat=mat,
        row_labels=ordered_categories,
        col_labels=["typ↔emb"],
        save_path=out_dir / "paper_heatmap_typicality_vs_emb.png",
        title=r"$\rho$(typicality, emb cosim)",
        figsize=FIGSIZE_SINGLE,
        xlabel="",
        vmin=vmin,
        vmax=vmax,
        max_xticks=1,
    )

    # Aggregate: typicality vs mean ctx cosim (single column)
    df1b = df_cat_typ_ctx.set_index("category").reindex(ordered_categories)
    mat1b = df1b["spearman_rho"].to_numpy(dtype=float).reshape(-1, 1)
    plot_heatmap(
        mat=mat1b,
        row_labels=ordered_categories,
        col_labels=["typ↔ctx"],
        save_path=out_dir / "paper_heatmap_typicality_vs_ctx_mean.png",
        title=r"$\rho$(typicality, ctx cosim mean)",
        figsize=FIGSIZE_SINGLE,
        xlabel="",
        vmin=vmin,
        vmax=vmax,
        max_xticks=1,
    )

    # Aggregate: typicality vs mean Jaccard (single column)
    df1c = df_cat_typ_jac.set_index("category").reindex(ordered_categories)
    mat1c = df1c["spearman_rho"].to_numpy(dtype=float).reshape(-1, 1)
    plot_heatmap(
        mat=mat1c,
        row_labels=ordered_categories,
        col_labels=["typ↔jac"],
        save_path=out_dir / "paper_heatmap_typicality_vs_jac_mean.png",
        title=r"$\rho$(typicality, Jaccard mean)",
        figsize=FIGSIZE_SINGLE,
        xlabel="",
        vmin=vmin,
        vmax=vmax,
        max_xticks=1,
    )

    # Aggregate: typicality vs Emb Jaccard (lowest-layer SAE on static W_E)
    df1d = df_cat_typ_emb_jac.set_index("category").reindex(ordered_categories)
    mat1d = df1d["spearman_rho"].to_numpy(dtype=float).reshape(-1, 1)
    plot_heatmap(
        mat=mat1d,
        row_labels=ordered_categories,
        col_labels=["typ↔emb jac"],
        save_path=out_dir / "paper_heatmap_typicality_vs_emb_jac_mean.png",
        title=r"$\rho$(typicality, Emb Jaccard)",
        figsize=FIGSIZE_SINGLE,
        xlabel="",
        vmin=vmin,
        vmax=vmax,
        max_xticks=1,
    )

    # Typicality vs contextual cosim by layer
    df2 = df_cat_typ_ctx_by_layer.set_index("category").reindex(ordered_categories)
    mat2 = df2[layer_cols_short].to_numpy(dtype=float)
    plot_heatmap(
        mat=mat2,
        row_labels=ordered_categories,
        col_labels=layer_labels,
        save_path=out_dir / "paper_heatmap_typicality_vs_ctx_by_layer.png",
        title=r"$\rho$(typicality, ctx cosim)",
        figsize=FIGSIZE_LAYER,
        xlabel=r"Layer $l$",
        vmin=vmin,
        vmax=vmax,
        max_xticks=6,
    )

    # Typicality vs Jaccard by layer
    df3 = df_cat_typ_jac_by_layer.set_index("category").reindex(ordered_categories)
    mat3 = df3[layer_cols_short].to_numpy(dtype=float)
    plot_heatmap(
        mat=mat3,
        row_labels=ordered_categories,
        col_labels=layer_labels,
        save_path=out_dir / "paper_heatmap_typicality_vs_jac_by_layer.png",
        title=r"$\rho$(typicality, Jaccard)",
        figsize=FIGSIZE_LAYER,
        xlabel=r"Layer $l$",
        vmin=vmin,
        vmax=vmax,
        max_xticks=6,
    )

    # Embedding cosim vs Jaccard by layer
    df4 = df_cat_emb_jac_by_layer.set_index("category").reindex(ordered_categories)
    mat4 = df4[layer_cols_short].to_numpy(dtype=float)
    plot_heatmap(
        mat=mat4,
        row_labels=ordered_categories,
        col_labels=layer_labels,
        save_path=out_dir / "paper_heatmap_emb_vs_jac_by_layer.png",
        title=r"$\rho$(emb cosim, Jaccard)",
        figsize=FIGSIZE_LAYER,
        xlabel=r"Layer $l$",
        vmin=vmin,
        vmax=vmax,
        max_xticks=6,
    )

    # Contextual cosim vs Jaccard by layer
    df5 = df_cat_ctx_jac_by_layer.set_index("category").reindex(ordered_categories)
    mat5 = df5[layer_cols_short].to_numpy(dtype=float)
    plot_heatmap(
        mat=mat5,
        row_labels=ordered_categories,
        col_labels=layer_labels,
        save_path=out_dir / "paper_heatmap_ctx_vs_jac_by_layer.png",
        title=r"$\rho$(ctx cosim, Jaccard)",
        figsize=FIGSIZE_LAYER,
        xlabel=r"Layer $l$",
        vmin=vmin,
        vmax=vmax,
        max_xticks=6,
    )

    # Typicality vs Emb Jaccard (lowest layer)
    df3b = df_cat_typ_emb_jac_by_layer.set_index("category").reindex(ordered_categories)
    mat3b = df3b[emb_jac_cols_short].to_numpy(dtype=float)
    plot_heatmap(
        mat=mat3b,
        row_labels=ordered_categories,
        col_labels=emb_jac_labels,
        save_path=out_dir / "paper_heatmap_typicality_vs_emb_jac_by_layer.png",
        title=r"$\rho$(typicality, Emb Jaccard)",
        figsize=FIGSIZE_SINGLE if len(emb_jaccard_layers) == 1 else FIGSIZE_LAYER,
        xlabel=r"Layer $l$",
        vmin=vmin,
        vmax=vmax,
        max_xticks=6,
    )

    # Embedding cosim vs Emb Jaccard
    df4b = df_cat_emb_emb_jac_by_layer.set_index("category").reindex(ordered_categories)
    mat4b = df4b[emb_jac_cols_short].to_numpy(dtype=float)
    plot_heatmap(
        mat=mat4b,
        row_labels=ordered_categories,
        col_labels=emb_jac_labels,
        save_path=out_dir / "paper_heatmap_emb_vs_emb_jac_by_layer.png",
        title=r"$\rho$(emb cosim, Emb Jaccard)",
        figsize=FIGSIZE_SINGLE if len(emb_jaccard_layers) == 1 else FIGSIZE_LAYER,
        xlabel=r"Layer $l$",
        vmin=vmin,
        vmax=vmax,
        max_xticks=6,
    )

    # Contextual cosim vs Emb Jaccard
    df5b = df_cat_ctx_emb_jac_by_layer.set_index("category").reindex(ordered_categories)
    mat5b = df5b[emb_jac_cols_short].to_numpy(dtype=float)
    plot_heatmap(
        mat=mat5b,
        row_labels=ordered_categories,
        col_labels=emb_jac_labels,
        save_path=out_dir / "paper_heatmap_ctx_vs_emb_jac_by_layer.png",
        title=r"$\rho$(ctx cosim, Emb Jaccard)",
        figsize=FIGSIZE_SINGLE if len(emb_jaccard_layers) == 1 else FIGSIZE_LAYER,
        xlabel=r"Layer $l$",
        vmin=vmin,
        vmax=vmax,
        max_xticks=6,
    )

    # Layer×Layer Jaccard similarity heatmap
    plot_layer_layer_heatmap(
        df_ll=df_ll_jac,
        save_path=out_dir / "paper_heatmap_layer_layer_spearman_jac_scores.png",
        vmin=vmin,
        vmax=vmax,
        max_xticks=6,
    )

    if len(emb_jaccard_layers) > 1:
        plot_layer_layer_heatmap(
            df_ll=df_ll_emb_jac,
            save_path=out_dir / "paper_heatmap_layer_layer_spearman_emb_jac_scores.png",
            vmin=vmin,
            vmax=vmax,
            max_xticks=6,
        )


def plot_metric_heatmap_set(
    out_dir: Path,
    metric_key: str,
    metric_label: str,
    layers: List[int],
    emb_layers: List[int],
    ordered_categories: List[str],
    df_cat_typ_agg: pd.DataFrame,
    df_cat_typ_by_layer: pd.DataFrame,
    df_cat_emb_by_layer: pd.DataFrame,
    df_cat_ctx_by_layer: pd.DataFrame,
    df_ll: pd.DataFrame,
    df_cat_typ_emb_agg: pd.DataFrame,
    df_cat_typ_emb_by_layer: pd.DataFrame,
    df_cat_emb_emb_by_layer: pd.DataFrame,
    df_cat_ctx_emb_by_layer: pd.DataFrame,
) -> None:
    """
    Generic heatmap set for one amplitude-based SAE latent metric (L1, KL,
    cosine, dot product, ...), reproducing the same 9 plots the Jaccard /
    Emb-Jaccard block above makes, parameterized by metric name instead of
    hardcoded "jac".
    """
    vmin, vmax = -1.0, 1.0
    layer_cols_short = [f"layer_{l}" for l in layers]
    layer_labels = [str(l) for l in layers]
    emb_cols_short = [f"layer_{l}" for l in emb_layers]
    emb_labels = [str(l) for l in emb_layers]

    def _agg_heatmap(df: pd.DataFrame, save_name: str, title: str, col_label: str) -> None:
        d = df.set_index("category").reindex(ordered_categories)
        mat = d["spearman_rho"].to_numpy(dtype=float).reshape(-1, 1)
        plot_heatmap(
            mat=mat,
            row_labels=ordered_categories,
            col_labels=[col_label],
            save_path=out_dir / save_name,
            title=title,
            figsize=FIGSIZE_SINGLE,
            xlabel="",
            vmin=vmin,
            vmax=vmax,
            max_xticks=1,
        )

    def _by_layer_heatmap(df: pd.DataFrame, save_name: str, title: str, cols: List[str], labels: List[str]) -> None:
        d = df.set_index("category").reindex(ordered_categories)
        mat = d[cols].to_numpy(dtype=float)
        plot_heatmap(
            mat=mat,
            row_labels=ordered_categories,
            col_labels=labels,
            save_path=out_dir / save_name,
            title=title,
            figsize=FIGSIZE_SINGLE if len(cols) == 1 else FIGSIZE_LAYER,
            xlabel=r"Layer $l$",
            vmin=vmin,
            vmax=vmax,
            max_xticks=6,
        )

    # Contextual
    _agg_heatmap(
        df_cat_typ_agg, f"paper_heatmap_typicality_vs_{metric_key}_mean.png",
        rf"$\rho$(typicality, {metric_label} mean)", f"typ↔{metric_key}",
    )
    _by_layer_heatmap(
        df_cat_typ_by_layer, f"paper_heatmap_typicality_vs_{metric_key}_by_layer.png",
        rf"$\rho$(typicality, {metric_label})", layer_cols_short, layer_labels,
    )
    _by_layer_heatmap(
        df_cat_emb_by_layer, f"paper_heatmap_emb_vs_{metric_key}_by_layer.png",
        rf"$\rho$(emb cosim, {metric_label})", layer_cols_short, layer_labels,
    )
    _by_layer_heatmap(
        df_cat_ctx_by_layer, f"paper_heatmap_ctx_vs_{metric_key}_by_layer.png",
        rf"$\rho$(ctx cosim, {metric_label})", layer_cols_short, layer_labels,
    )
    if len(layers) > 1:
        plot_layer_layer_heatmap(
            df_ll=df_ll,
            save_path=out_dir / f"paper_heatmap_layer_layer_spearman_{metric_key}_scores.png",
            vmin=vmin,
            vmax=vmax,
            max_xticks=6,
        )

    # Emb (static, uncontextualized W_E) variant
    _agg_heatmap(
        df_cat_typ_emb_agg, f"paper_heatmap_typicality_vs_emb_{metric_key}_mean.png",
        rf"$\rho$(typicality, Emb {metric_label})", f"typ↔emb {metric_key}",
    )
    _by_layer_heatmap(
        df_cat_typ_emb_by_layer, f"paper_heatmap_typicality_vs_emb_{metric_key}_by_layer.png",
        rf"$\rho$(typicality, Emb {metric_label})", emb_cols_short, emb_labels,
    )
    _by_layer_heatmap(
        df_cat_emb_emb_by_layer, f"paper_heatmap_emb_vs_emb_{metric_key}_by_layer.png",
        rf"$\rho$(emb cosim, Emb {metric_label})", emb_cols_short, emb_labels,
    )
    _by_layer_heatmap(
        df_cat_ctx_emb_by_layer, f"paper_heatmap_ctx_vs_emb_{metric_key}_by_layer.png",
        rf"$\rho$(ctx cosim, Emb {metric_label})", emb_cols_short, emb_labels,
    )