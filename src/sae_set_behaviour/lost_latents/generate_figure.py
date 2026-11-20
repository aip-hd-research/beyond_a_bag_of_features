from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
from hydra.utils import to_absolute_path
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

LABEL_FONTSIZE = 20
LEGEND_FONTSIZE = 16
TITLE_FONTSIZE = 16
TICK_FONTSIZE = 14


def plot_layer_blocks_with_k_colors(
    selected_layers: List[int],
    mean_by_layer_by_k: Dict[int, Dict[int, float]],
    std_by_layer_by_k: Dict[int, Dict[int, float]],
    title: str,
    save_path: Path,
    k_values: List[int] = [1, 2, 3, 4, 5],
    gap: float = 1.5,
) -> None:
    k_to_color = {
        1: "tab:green",
        2: "tab:red",
        3: "tab:blue",
        4: "tab:orange",
        5: "tab:purple",
    }

    positions = {}
    x = 0.0
    layer_centers = []
    layer_labels = []
    for layer in selected_layers:
        start_x = x
        for k in k_values:
            positions[(layer, k)] = x
            x += 1.0
        end_x = x - 1.0
        layer_centers.append(0.5 * (start_x + end_x))
        layer_labels.append(f"layer {layer}")
        x += gap

    plt.figure(figsize=(max(8, 1.2 * len(selected_layers) * len(k_values)), 4))

    for k in k_values:
        xs = [positions[(layer, k)] for layer in selected_layers]
        ys = [mean_by_layer_by_k[layer].get(k, float("nan")) for layer in selected_layers]
        es = [std_by_layer_by_k[layer].get(k, 0.0) for layer in selected_layers]
        plt.bar(xs, ys, yerr=es, capsize=3, color=k_to_color.get(k, "gray"), label=f"k={k}")

    plt.xticks(layer_centers, layer_labels, rotation=0)
    plt.xlabel("Selected layers (blocks)")
    plt.ylabel("mean lost_rate (std error bars)")
    plt.title(title)
    plt.legend(title="Attributes added (k)", ncol=5, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_against_k_on_ax(
    ax: Axes,
    selected_layers: List[int],
    mean_by_layer_by_k: Dict[int, Dict[int, float]],
    std_by_layer_by_k: Optional[Dict[int, Dict[int, float]]] = None,
    title: str = "",
    ylabel: str = "",
    k_values: Optional[List[int]] = None,
    ylim: Optional[Tuple[float, float]] = (0.0, 1.0),
    show_legend: bool = True,
) -> None:
    layers = [int(x) for x in selected_layers]

    mean_int: Dict[int, Dict[int, float]] = {}
    for layer_k, d in mean_by_layer_by_k.items():
        layer = int(layer_k)
        mean_int[layer] = {int(k): float(v) for k, v in d.items()}

    std_int: Dict[int, Dict[int, float]] = {}
    if std_by_layer_by_k is not None:
        for layer_k, d in std_by_layer_by_k.items():
            layer = int(layer_k)
            std_int[layer] = {int(k): float(v) for k, v in d.items()}

    if k_values is None:
        ks = sorted({k for layer in layers for k in mean_int.get(layer, {}).keys()})
    else:
        ks = [int(k) for k in k_values]

    if not ks:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("step $k$", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
        return

    k_min, k_max = ks[0], ks[-1]
    xticks = [int(round(x)) for x in np.linspace(k_min, k_max, 5)]
    xticks = sorted(set(xticks))

    for layer in layers:
        x, y, s = [], [], []
        for k in ks:
            x.append(k)
            y.append(mean_int.get(layer, {}).get(k, float("nan")))
            if std_by_layer_by_k is not None:
                s.append(std_int.get(layer, {}).get(k, 0.0))

        line = ax.plot(x, y, linewidth=2.0, label=f"layer {layer}")
        line_color = line[0].get_color()

        if std_by_layer_by_k is not None and len(s) == len(y):
            lower = [yy - ss for yy, ss in zip(y, s)]
            upper = [yy + ss for yy, ss in zip(y, s)]
            ax.fill_between(x, lower, upper, color=line_color, alpha=0.18, linewidth=0)

    ax.set_xticks(xticks)
    ax.set_xlabel("step $k$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, linewidth=0.5, alpha=0.25)

    if ylim is not None:
        y_min, y_max = ylim
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(y_min, y_max, 11))

    if show_legend:
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")



def plot_against_k(
    selected_layers: List[int],
    mean_by_layer_by_k: Dict[int, Dict[int, float]],
    std_by_layer_by_k: Optional[Dict[int, Dict[int, float]]] = None,
    title: str = "",
    ylabel: str = "",
    save_path: Path = Path("k_vs_lost_rate.png"),
    k_values: Optional[List[int]] = None,
    show_error_bars: bool = False,
    ylim: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_against_k_on_ax(
        ax=ax,
        selected_layers=selected_layers,
        mean_by_layer_by_k=mean_by_layer_by_k,
        std_by_layer_by_k=std_by_layer_by_k,
        title=title,
        ylabel=ylabel,
        k_values=k_values,
        ylim=ylim,
        show_legend=True,
    )
    fig.tight_layout()
    fig.savefig(save_path)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_metric_against_layer_on_ax(
    ax: Axes,
    *,
    layers: List[int],
    mean_by_layer: Dict[int, float],
    std_by_layer: Dict[int, float],
    label: str = "",
    title: str = "",
    ylabel: str = "",
    show_error_bars: bool = True,
    color: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> None:
    x = [int(l) for l in layers]
    y = [float(mean_by_layer[l]) for l in x]
    s = [float(std_by_layer.get(l, 0.0)) for l in x]

    line = ax.plot(x, y, marker="o", label=label or None, color=color)
    line_color = line[0].get_color()

    if show_error_bars:
        lower = [yy - ss for yy, ss in zip(y, s)]
        upper = [yy + ss for yy, ss in zip(y, s)]
        ax.fill_between(x, lower, upper, color=line_color, alpha=0.18, linewidth=0)

    ax.set_xlabel("Layer $l$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)

    if ylim is not None:
        y_min, y_max = ylim
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.linspace(y_min, y_max, 11))

    if label:
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")


def plot_metric_against_layer(
    *,
    layers: List[int],
    mean_by_layer: Dict[int, float],
    std_by_layer: Dict[int, float],
    label: str = "",
    title: str = "",
    ylabel: str = "",
    save_path: Path,
    show_error_bars: bool = True,
    color: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_metric_against_layer_on_ax(
        ax=ax,
        layers=layers,
        mean_by_layer=mean_by_layer,
        std_by_layer=std_by_layer,
        label=label,
        title=title,
        ylabel=ylabel,
        show_error_bars=show_error_bars,
        color=color,
        ylim=ylim,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_amplitude_histogram(
    *,
    bin_edges: np.ndarray,
    general_counts: np.ndarray,
    lost_counts: np.ndarray,
    general_stats: Tuple[float, float],
    lost_stats: Tuple[float, float],
    title: str = "",
    save_path: Path,
) -> None:
    """
    Overlaid density histograms comparing SAE latent activation amplitude between
    two groups sharing the same bin edges: every active latent on the target token
    ("general", stats = (mean, std)) vs. the subset of those latents that go on to
    be lost ("lost", stats = (mean, std)).
    """
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)

    general_total = max(int(general_counts.sum()), 1)
    lost_total = max(int(lost_counts.sum()), 1)
    general_density = general_counts / general_total / widths
    lost_density = lost_counts / lost_total / widths

    general_mean, general_std = general_stats
    lost_mean, lost_std = lost_stats

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        centers, general_density, width=widths, align="center", alpha=0.5, color="tab:blue",
        label=f"general active (n={int(general_counts.sum())}, mean={general_mean:.3g}, std={general_std:.3g})",
    )
    ax.bar(
        centers, lost_density, width=widths, align="center", alpha=0.5, color="tab:red",
        label=f"lost (n={int(lost_counts.sum())}, mean={lost_mean:.3g}, std={lost_std:.3g})",
    )
    ax.set_xlabel("SAE latent activation amplitude", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("density", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=max(9, LEGEND_FONTSIZE - 3), loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_two_metrics_against_layer_on_ax(
    ax: Axes,
    *,
    layers: List[int],
    mean_metric1_by_layer: Dict[int, float],
    std_metric1_by_layer: Dict[int, float],
    mean_metric2_by_layer: Dict[int, float],
    std_metric2_by_layer: Dict[int, float],
    metric1_label: str,
    metric2_label: str,
    title: str,
    ylabel: str,
    show_error_bars: bool = True,
    metric1_color: Optional[str] = None,
    metric2_color: Optional[str] = None,
) -> None:
    x = [int(l) for l in layers]
    y1 = [float(mean_metric1_by_layer[l]) for l in x]
    y2 = [float(mean_metric2_by_layer[l]) for l in x]
    s1 = [float(std_metric1_by_layer[l]) for l in x]
    s2 = [float(std_metric2_by_layer[l]) for l in x]

    line1 = ax.plot(x, y1, marker="o", label=metric1_label, color=metric1_color)
    color1 = line1[0].get_color()

    if show_error_bars:
        lower1 = [yy - ss for yy, ss in zip(y1, s1)]
        upper1 = [yy + ss for yy, ss in zip(y1, s1)]
        ax.fill_between(x, lower1, upper1, color=color1, alpha=0.18, linewidth=0)

    line2 = ax.plot(x, y2, marker="o", label=metric2_label, color=metric2_color)
    color2 = line2[0].get_color()

    if show_error_bars:
        lower2 = [yy - ss for yy, ss in zip(y2, s2)]
        upper2 = [yy + ss for yy, ss in zip(y2, s2)]
        ax.fill_between(x, lower2, upper2, color=color2, alpha=0.18, linewidth=0)

    ax.set_xlabel("Layer $l$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(True, alpha=0.3)


def plot_two_metrics_against_layer(
    *,
    layers: List[int],
    mean_metric1_by_layer: Dict[int, float],
    std_metric1_by_layer: Dict[int, float],
    mean_metric2_by_layer: Dict[int, float],
    std_metric2_by_layer: Dict[int, float],
    metric1_label: str,
    metric2_label: str,
    title: str,
    ylabel: str,
    save_path: Path,
    show_error_bars: bool = True,
    metric1_color: Optional[str] = None,
    metric2_color: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_two_metrics_against_layer_on_ax(
        ax=ax,
        layers=layers,
        mean_metric1_by_layer=mean_metric1_by_layer,
        std_metric1_by_layer=std_metric1_by_layer,
        mean_metric2_by_layer=mean_metric2_by_layer,
        std_metric2_by_layer=std_metric2_by_layer,
        metric1_label=metric1_label,
        metric2_label=metric2_label,
        title=title,
        ylabel=ylabel,
        show_error_bars=show_error_bars,
        metric1_color=metric1_color,
        metric2_color=metric2_color,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt


def plot_main_metric_with_inlay_on_ax(
    ax: Axes,
    *,
    layers: List[int],
    mean_main_by_layer: Dict[int, float],
    std_main_by_layer: Dict[int, float],
    mean_inlay_metric1_by_layer: Dict[int, float],
    std_inlay_metric1_by_layer: Dict[int, float],
    mean_inlay_metric2_by_layer: Dict[int, float],
    std_inlay_metric2_by_layer: Dict[int, float],
    main_label: str,
    inlay_metric1_label: str,
    inlay_metric2_label: str,
    title: str,
    ylabel: str,
    show_error_bars: bool = True,
    main_color: Optional[str] = None,
    inlay_metric1_color: Optional[str] = None,
    inlay_metric2_color: Optional[str] = None,
    inlay_loc: str = "upper left",
    inlay_width: str = "34%",
    inlay_height: str = "34%",
    inlay_bbox_to_anchor: Optional[Tuple[float, float, float, float]] = (0.15, 0.02, 1.0, 1.0),
    legend_loc: str = "upper right",
) -> None:
    x = [int(l) for l in layers]

    if len(x) <= 5:
        main_xticks = x
    else:
        idxs = [round(i * (len(x) - 1) / 4) for i in range(5)]
        main_xticks = []
        seen = set()
        for idx in idxs:
            val = x[idx]
            if val not in seen:
                main_xticks.append(val)
                seen.add(val)

    y_main = [float(mean_main_by_layer[l]) for l in x]
    s_main = [float(std_main_by_layer[l]) for l in x]

    line_main = ax.plot(x, y_main, marker="o", markersize=5, label=main_label, color=main_color)
    color_main = line_main[0].get_color()

    if show_error_bars:
        lower_main = [yy - ss for yy, ss in zip(y_main, s_main)]
        upper_main = [yy + ss for yy, ss in zip(y_main, s_main)]
        ax.fill_between(x, lower_main, upper_main, color=color_main, alpha=0.18, linewidth=0)

    ax.set_xlabel("Layer $l$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.set_xticks(main_xticks)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    axins = inset_axes(
        ax,
        width=inlay_width,
        height=inlay_height,
        loc=inlay_loc,
        bbox_to_anchor=inlay_bbox_to_anchor,
        bbox_transform=ax.transAxes,
        borderpad=0.8,
    )

    axins.patch.set_facecolor("white")
    axins.patch.set_alpha(0.72)

    y1 = [float(mean_inlay_metric1_by_layer[l]) for l in x]
    y2 = [float(mean_inlay_metric2_by_layer[l]) for l in x]
    s1 = [float(std_inlay_metric1_by_layer[l]) for l in x]
    s2 = [float(std_inlay_metric2_by_layer[l]) for l in x]

    line1 = axins.plot(x, y1, marker="o", markersize=3, linewidth=1.6, label=inlay_metric1_label, color=inlay_metric1_color)
    color1 = line1[0].get_color()

    if show_error_bars:
        lower1 = [yy - ss for yy, ss in zip(y1, s1)]
        upper1 = [yy + ss for yy, ss in zip(y1, s1)]
        axins.fill_between(x, lower1, upper1, color=color1, alpha=0.18, linewidth=0)

    line2 = axins.plot(x, y2, marker="o", markersize=3, linewidth=1.6, label=inlay_metric2_label, color=inlay_metric2_color)
    color2 = line2[0].get_color()

    if show_error_bars:
        lower2 = [yy - ss for yy, ss in zip(y2, s2)]
        upper2 = [yy + ss for yy, ss in zip(y2, s2)]
        axins.fill_between(x, lower2, upper2, color=color2, alpha=0.18, linewidth=0)

    axins.set_xticks(main_xticks)
    axins.set_xlim(min(x) - 0.5, max(x) + 0.5)
    axins.set_yticks(np.linspace(0, 1, 6))
    axins.set_ylim(-0.02, 1.02)
    axins.tick_params(axis="both", labelsize=max(9, TICK_FONTSIZE - 3), pad=2)
    axins.grid(True, alpha=0.25)
    axins.margins(x=0.06, y=0.08)

    handles = [line_main[0], line1[0], line2[0]]
    labels = [main_label, inlay_metric1_label, inlay_metric2_label]
    ax.legend(handles, labels, fontsize=LEGEND_FONTSIZE, loc=legend_loc, ncol=1, framealpha=0.9)

def plot_main_metric_with_inlay(*, layers: List[int], mean_main_by_layer: Dict[int, float], std_main_by_layer: Dict[int, float], mean_inlay_metric1_by_layer: Dict[int, float],
    std_inlay_metric1_by_layer: Dict[int, float], mean_inlay_metric2_by_layer: Dict[int, float], std_inlay_metric2_by_layer: Dict[int, float], main_label: str, inlay_metric1_label: str,
    inlay_metric2_label: str, title: str, ylabel: str, save_path: Path, show_error_bars: bool = True, main_color: Optional[str] = None,
    inlay_metric1_color: Optional[str] = None, inlay_metric2_color: Optional[str] = None,inlay_loc: str = "lower left",inlay_width: str = "45%",inlay_height: str = "45%") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_main_metric_with_inlay_on_ax(
        ax=ax,
        layers=layers,
        mean_main_by_layer=mean_main_by_layer,
        std_main_by_layer=std_main_by_layer,
        mean_inlay_metric1_by_layer=mean_inlay_metric1_by_layer,
        std_inlay_metric1_by_layer=std_inlay_metric1_by_layer,
        mean_inlay_metric2_by_layer=mean_inlay_metric2_by_layer,
        std_inlay_metric2_by_layer=std_inlay_metric2_by_layer,
        main_label=main_label,
        inlay_metric1_label=inlay_metric1_label,
        inlay_metric2_label=inlay_metric2_label,
        title=title,
        ylabel=ylabel,
        show_error_bars=show_error_bars,
        main_color=main_color,
        inlay_metric1_color=inlay_metric1_color,
        inlay_metric2_color=inlay_metric2_color,
        inlay_loc=inlay_loc,
        inlay_width=inlay_width,
        inlay_height=inlay_height,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)