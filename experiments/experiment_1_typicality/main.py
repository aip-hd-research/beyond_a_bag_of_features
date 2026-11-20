from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import gc

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, MISSING

import torch
from transformer_lens import HookedTransformer
import pandas as pd
import wandb
from sae_lens import SAE
from sae_set_behaviour.typicality.utils import make_out_dir
from sae_set_behaviour.util import SetupConfig, WandbConfig, init_wandb, setup, build_dummy_dataset
from sae_set_behaviour.typicality.metrics import spearman, compute_cosim_table, embed_static_E
from sae_set_behaviour.typicality.clustering import (
    compute_clustering_ami_table,
    split_clustering_ami_tables,
)
from sae_set_behaviour.typicality.plots import plot_all_heatmaps_paper, plot_metric_heatmap_set, make_ordered_categories_for_plots
from sae_set_behaviour.typicality.prepare_dataset import load_and_group_dataset
from sae_set_behaviour.typicality.paper_metrics import (
    collect_prompted_word_activations_per_layer,
    collect_static_word_token_embeddings,
    compute_sae_union_per_layer,
    compute_jaccard_per_layer,
    compute_contextual_cosim_per_layer,
    compute_latent_amplitude_metrics_per_layer,
    PAPER_TEMPLATE,
)

NEW_METRICS = ["l1", "kl", "klsm", "saecos", "dot"]
NEW_METRIC_LABELS = {
    "l1": "L1", "kl": "KL (L1 norm)", "klsm": "KL (softmax)",
    "saecos": "SAE cosine", "dot": "SAE dot",
}

@dataclass
class Config(SetupConfig):
    dataset_name: str = "CShani/human-concepts"
    split: str = "train"
    min_items_per_category: int = 5
    lowercase: bool = False
    heatmap_max_categories: int = 60
    static_pooling: str = "avg"          # avg|first|last  (no apt_sym)
    paper_template: str = PAPER_TEMPLATE
    paper_batch_size: int = 8
    sae_encode_batch_size: int = 16
    sae_device: str = "cuda"
    wandb: WandbConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    init_wandb(cfg.wandb, cfg)
    run = wandb.run

    by_category, categories, items = load_and_group_dataset(cfg)
    layers = [int(x) for x in cfg.sae.layers]

    dummy_dataset = build_dummy_dataset()
    model, saes, _, _ = setup(cfg, dummy_dataset)
    model.eval()
    for sae in saes:
        sae.eval()
        sae.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sae_by_layer: Dict[int, SAE] = {int(l): sae for l, sae in zip(layers, saes)}
    hook_names_by_layer: Dict[int, str] = {
        int(l): str(sae_by_layer[int(l)].cfg.metadata.hook_name) for l in layers
    }
    out_dir = make_out_dir(cfg, layers)
    all_words = list(dict.fromkeys(categories + items))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # ── Static embedding baseline: avg-pooled W_E cosine (no template) ────────
    df_item = compute_cosim_table(
        model=model,
        by_category=by_category,
        categories=categories,
        items=items,
        static_pooling=cfg.static_pooling,
    )
    static_emb_all_words = embed_static_E(model, all_words, pooling=cfg.static_pooling)
    static_word_embs, static_word_mask = collect_static_word_token_embeddings(model, all_words)
    lowest_layer = min(layers)

    # ── Collect prompted residual stream activations for all words ─────────────
    avg_residual_by_layer, word_acts_by_layer, word_mask = collect_prompted_word_activations_per_layer(
        model=model,
        words=all_words,
        hook_names_by_layer=hook_names_by_layer,
        device=str(cfg.device),
        template=str(cfg.paper_template),
        batch_size=int(cfg.paper_batch_size),
    )

    del model
    del saes
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Contextual cosine similarity (residual stream avg over word tokens) ────
    ctx_cosim_by_layer = compute_contextual_cosim_per_layer(
        avg_residual_by_layer=avg_residual_by_layer,
        by_category=by_category,
        categories=categories,
        word_to_idx=word_to_idx,
        layers=layers,
    )

    # ── SAE Jaccard (union of active features over word tokens) ────────────────
    union_by_layer, mean_latent_by_layer = compute_sae_union_per_layer(
        sae_by_layer=sae_by_layer,
        word_acts_by_layer=word_acts_by_layer,
        word_mask=word_mask,
        layers=layers,
        sae_device=str(cfg.sae_device),
        encode_batch_size=int(cfg.sae_encode_batch_size),
    )
    del word_acts_by_layer, word_mask
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    jac_by_layer = compute_jaccard_per_layer(
        union_by_layer=union_by_layer,
        by_category=by_category,
        categories=categories,
        word_to_idx=word_to_idx,
        layers=layers,
    )
    latent_metrics_by_layer = compute_latent_amplitude_metrics_per_layer(
        mean_latent_by_layer=mean_latent_by_layer,
        by_category=by_category,
        categories=categories,
        word_to_idx=word_to_idx,
        layers=layers,
    )

    emb_jaccard_union_by_layer, emb_mean_latent_by_layer = compute_sae_union_per_layer(
        sae_by_layer=sae_by_layer,
        word_acts_by_layer={lowest_layer: static_word_embs},
        word_mask=static_word_mask,
        layers=[lowest_layer],
        sae_device=str(cfg.sae_device),
        encode_batch_size=int(cfg.sae_encode_batch_size),
    )
    del static_word_embs, static_word_mask
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    emb_jac_by_layer = compute_jaccard_per_layer(
        union_by_layer=emb_jaccard_union_by_layer,
        by_category=by_category,
        categories=categories,
        word_to_idx=word_to_idx,
        layers=[lowest_layer],
    )
    emb_latent_metrics_by_layer = compute_latent_amplitude_metrics_per_layer(
        mean_latent_by_layer=emb_mean_latent_by_layer,
        by_category=by_category,
        categories=categories,
        word_to_idx=word_to_idx,
        layers=[lowest_layer],
    )

    df_clustering_ami = compute_clustering_ami_table(
        words=all_words,
        categories=categories,
        by_category=by_category,
        layers=layers,
        avg_residual_by_layer=avg_residual_by_layer,
        static_emb=static_emb_all_words,
        union_by_layer=union_by_layer,
        emb_jaccard_union_by_layer=emb_jaccard_union_by_layer,
        mean_latent_by_layer=mean_latent_by_layer,
        emb_mean_latent_by_layer=emb_mean_latent_by_layer,
    )
    del (
        avg_residual_by_layer, static_emb_all_words, union_by_layer, emb_jaccard_union_by_layer,
        mean_latent_by_layer, emb_mean_latent_by_layer,
    )

    # ── Attach all metric columns to df_item ───────────────────────────────────
    emb_jaccard_layers = [lowest_layer]
    df_item, ctx_cols, jac_cols, emb_jac_cols, new_metric_cols, new_metric_emb_cols = attach_paper_columns(
        df_item,
        ctx_cosim_by_layer,
        jac_by_layer,
        emb_jac_by_layer,
        latent_metrics_by_layer,
        emb_latent_metrics_by_layer,
        layers,
        emb_jaccard_layers,
    )

    # ── Spearman tables ────────────────────────────────────────────────────────
    # Aggregate (mean over layers)
    df_cat_typ_emb = per_category_spearman(df_item, "typicality", "cosim_item_category")
    df_cat_typ_ctx = per_category_spearman(df_item, "typicality", "ctx_cosim_mean_layers")
    df_cat_typ_jac = per_category_spearman(df_item, "typicality", "jaccard_mean_layers")
    df_cat_typ_emb_jac = per_category_spearman(df_item, "typicality", "emb_jaccard_mean_layers")
    df_cat_emb_jac = per_category_spearman(df_item, "cosim_item_category", "jaccard_mean_layers")
    df_cat_emb_cosim_emb_jac = per_category_spearman(df_item, "cosim_item_category", "emb_jaccard_mean_layers")
    df_cat_ctx_jac = per_category_spearman(df_item, "ctx_cosim_mean_layers", "jaccard_mean_layers")
    df_cat_ctx_emb_jac = per_category_spearman(df_item, "ctx_cosim_mean_layers", "emb_jaccard_mean_layers")

    # Per-layer
    df_cat_typ_ctx_by_layer = per_category_by_layer_spearman_ycols(
        df_item, "typicality", "ctx_cosim_layer_{l}", layers
    )
    df_cat_typ_jac_by_layer = per_category_by_layer_spearman_ycols(
        df_item, "typicality", "jaccard_layer_{l}", layers
    )
    df_cat_typ_emb_jac_by_layer = per_category_by_layer_spearman_ycols(
        df_item, "typicality", "emb_jaccard_layer_{l}", emb_jaccard_layers
    )
    df_cat_emb_jac_by_layer = per_category_by_layer_spearman_ycols(
        df_item, "cosim_item_category", "jaccard_layer_{l}", layers
    )
    df_cat_emb_emb_jac_by_layer = per_category_by_layer_spearman_ycols(
        df_item, "cosim_item_category", "emb_jaccard_layer_{l}", emb_jaccard_layers
    )
    df_cat_ctx_jac_by_layer = per_category_by_layer_spearman_paired(
        df_item, "ctx_cosim_layer_{l}", "jaccard_layer_{l}", layers
    )
    df_cat_ctx_emb_jac_by_layer = per_category_by_layer_spearman_paired(
        df_item, "ctx_cosim_layer_{l}", "emb_jaccard_layer_{l}", emb_jaccard_layers
    )
    df_ll_jac = layer_layer_spearman(df_item, jac_cols, layers)
    df_ll_emb_jac = layer_layer_spearman(df_item, emb_jac_cols, emb_jaccard_layers)

    # ── Amplitude-based SAE latent metrics (L1, KL, cosine, dot product) ───────
    # Same three-level treatment as Jaccard, for both the contextual and Emb
    # (static, uncontextualized W_E) flavors.
    new_metric_tables: Dict[str, Dict[str, pd.DataFrame]] = {}
    for m in NEW_METRICS:
        t: Dict[str, pd.DataFrame] = {}
        t["typ_agg"] = per_category_spearman(df_item, "typicality", f"{m}_mean_layers")
        t["emb_agg"] = per_category_spearman(df_item, "cosim_item_category", f"{m}_mean_layers")
        t["ctx_agg"] = per_category_spearman(df_item, "ctx_cosim_mean_layers", f"{m}_mean_layers")
        t["typ_by_layer"] = per_category_by_layer_spearman_ycols(df_item, "typicality", f"{m}_layer_{{l}}", layers)
        t["emb_by_layer"] = per_category_by_layer_spearman_ycols(
            df_item, "cosim_item_category", f"{m}_layer_{{l}}", layers
        )
        t["ctx_by_layer"] = per_category_by_layer_spearman_paired(
            df_item, "ctx_cosim_layer_{l}", f"{m}_layer_{{l}}", layers
        )
        t["ll"] = layer_layer_spearman(df_item, new_metric_cols[m], layers)

        t["typ_emb_agg"] = per_category_spearman(df_item, "typicality", f"emb_{m}_mean_layers")
        t["emb_emb_agg"] = per_category_spearman(df_item, "cosim_item_category", f"emb_{m}_mean_layers")
        t["ctx_emb_agg"] = per_category_spearman(df_item, "ctx_cosim_mean_layers", f"emb_{m}_mean_layers")
        t["typ_emb_by_layer"] = per_category_by_layer_spearman_ycols(
            df_item, "typicality", f"emb_{m}_layer_{{l}}", emb_jaccard_layers
        )
        t["emb_emb_by_layer"] = per_category_by_layer_spearman_ycols(
            df_item, "cosim_item_category", f"emb_{m}_layer_{{l}}", emb_jaccard_layers
        )
        t["ctx_emb_by_layer"] = per_category_by_layer_spearman_paired(
            df_item, "ctx_cosim_layer_{l}", f"emb_{m}_layer_{{l}}", emb_jaccard_layers
        )
        t["ll_emb"] = layer_layer_spearman(df_item, new_metric_emb_cols[m], emb_jaccard_layers)

        # Metric on the continuous SAE activation vector vs Jaccard on the same
        # (binarized) SAE activation vector, both built over the target word.
        t["jac_agg"] = per_category_spearman(df_item, f"{m}_mean_layers", "jaccard_mean_layers")
        t["jac_by_layer"] = per_category_by_layer_spearman_paired(
            df_item, f"{m}_layer_{{l}}", "jaccard_layer_{l}", layers
        )

        new_metric_tables[m] = t

    # ── Print and save summaries ───────────────────────────────────────────────
    print_summaries(
        df_item, df_cat_typ_emb, df_cat_typ_ctx, df_cat_typ_jac, df_cat_typ_emb_jac,
        df_cat_emb_jac, df_cat_emb_cosim_emb_jac, df_cat_ctx_jac, df_cat_ctx_emb_jac,
        new_metric_tables,
    )
    print_clustering_ami(df_clustering_ami)
    write_summary_txt(
        out_dir, cfg, layers, lowest_layer, df_item,
        df_cat_typ_emb, df_cat_typ_ctx, df_cat_typ_jac, df_cat_typ_emb_jac,
        df_cat_emb_jac, df_cat_emb_cosim_emb_jac, df_cat_ctx_jac, df_cat_ctx_emb_jac,
        df_clustering_ami, new_metric_tables,
    )

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_all_heatmaps_paper(
        out_dir=out_dir,
        layers=layers,
        emb_jaccard_layers=emb_jaccard_layers,
        df_cat_typ_cos=df_cat_typ_emb,
        df_cat_typ_ctx_by_layer=df_cat_typ_ctx_by_layer,
        df_cat_typ_jac_by_layer=df_cat_typ_jac_by_layer,
        df_cat_typ_emb_jac_by_layer=df_cat_typ_emb_jac_by_layer,
        df_cat_emb_jac_by_layer=df_cat_emb_jac_by_layer,
        df_cat_emb_emb_jac_by_layer=df_cat_emb_emb_jac_by_layer,
        df_cat_ctx_jac_by_layer=df_cat_ctx_jac_by_layer,
        df_cat_ctx_emb_jac_by_layer=df_cat_ctx_emb_jac_by_layer,
        df_ll_jac=df_ll_jac,
        df_ll_emb_jac=df_ll_emb_jac,
        df_cat_typ_ctx=df_cat_typ_ctx,
        df_cat_typ_jac=df_cat_typ_jac,
        df_cat_typ_emb_jac=df_cat_typ_emb_jac,
        max_categories=int(cfg.heatmap_max_categories),
    )

    ordered_categories = make_ordered_categories_for_plots(
        df_cat_typ_apm_by_layer=df_cat_typ_jac_by_layer,
        layers=layers,
        max_categories=int(cfg.heatmap_max_categories),
    )
    for m in NEW_METRICS:
        t = new_metric_tables[m]
        plot_metric_heatmap_set(
            out_dir=out_dir,
            metric_key=m,
            metric_label=NEW_METRIC_LABELS[m],
            layers=layers,
            emb_layers=emb_jaccard_layers,
            ordered_categories=ordered_categories,
            df_cat_typ_agg=t["typ_agg"],
            df_cat_typ_by_layer=t["typ_by_layer"],
            df_cat_emb_by_layer=t["emb_by_layer"],
            df_cat_ctx_by_layer=t["ctx_by_layer"],
            df_ll=t["ll"],
            df_cat_typ_emb_agg=t["typ_emb_agg"],
            df_cat_typ_emb_by_layer=t["typ_emb_by_layer"],
            df_cat_emb_emb_by_layer=t["emb_emb_by_layer"],
            df_cat_ctx_emb_by_layer=t["ctx_emb_by_layer"],
        )

    # ── Log to WandB and save CSVs ─────────────────────────────────────────────
    _log_df_table(df_item,                csv_path=out_dir / "df_item.csv",                    wandb_key="tables/df_item")
    _log_df_table(df_cat_typ_emb,         csv_path=out_dir / "df_cat_typ_emb.csv",             wandb_key="tables/df_cat_typ_emb")
    _log_df_table(df_cat_typ_ctx,         csv_path=out_dir / "df_cat_typ_ctx.csv",             wandb_key="tables/df_cat_typ_ctx")
    _log_df_table(df_cat_typ_jac,         csv_path=out_dir / "df_cat_typ_jac.csv",             wandb_key="tables/df_cat_typ_jac")
    _log_df_table(df_cat_typ_emb_jac,     csv_path=out_dir / "df_cat_typ_emb_jac.csv",         wandb_key="tables/df_cat_typ_emb_jac")
    _log_df_table(df_cat_emb_jac,         csv_path=out_dir / "df_cat_emb_jac.csv",             wandb_key="tables/df_cat_emb_jac")
    _log_df_table(df_cat_emb_cosim_emb_jac, csv_path=out_dir / "df_cat_emb_cosim_emb_jac.csv", wandb_key="tables/df_cat_emb_cosim_emb_jac")
    _log_df_table(df_cat_ctx_jac,         csv_path=out_dir / "df_cat_ctx_jac.csv",             wandb_key="tables/df_cat_ctx_jac")
    _log_df_table(df_cat_ctx_emb_jac,     csv_path=out_dir / "df_cat_ctx_emb_jac.csv",         wandb_key="tables/df_cat_ctx_emb_jac")
    _log_df_table(df_cat_typ_ctx_by_layer, csv_path=out_dir / "df_cat_typ_ctx_by_layer.csv",   wandb_key="tables/df_cat_typ_ctx_by_layer")
    _log_df_table(df_cat_typ_jac_by_layer, csv_path=out_dir / "df_cat_typ_jac_by_layer.csv",   wandb_key="tables/df_cat_typ_jac_by_layer")
    _log_df_table(df_cat_typ_emb_jac_by_layer, csv_path=out_dir / "df_cat_typ_emb_jac_by_layer.csv", wandb_key="tables/df_cat_typ_emb_jac_by_layer")
    _log_df_table(df_cat_emb_jac_by_layer, csv_path=out_dir / "df_cat_emb_jac_by_layer.csv",   wandb_key="tables/df_cat_emb_jac_by_layer")
    _log_df_table(df_cat_emb_emb_jac_by_layer, csv_path=out_dir / "df_cat_emb_emb_jac_by_layer.csv", wandb_key="tables/df_cat_emb_emb_jac_by_layer")
    _log_df_table(df_cat_ctx_jac_by_layer, csv_path=out_dir / "df_cat_ctx_jac_by_layer.csv",   wandb_key="tables/df_cat_ctx_jac_by_layer")
    _log_df_table(df_cat_ctx_emb_jac_by_layer, csv_path=out_dir / "df_cat_ctx_emb_jac_by_layer.csv", wandb_key="tables/df_cat_ctx_emb_jac_by_layer")
    _log_df_table(df_ll_jac.reset_index(), csv_path=out_dir / "df_ll_jac.csv",                 wandb_key="tables/df_ll_jac")
    _log_df_table(df_ll_emb_jac.reset_index(), csv_path=out_dir / "df_ll_emb_jac.csv",         wandb_key="tables/df_ll_emb_jac")

    for m in NEW_METRICS:
        t = new_metric_tables[m]
        _log_df_table(t["typ_agg"], csv_path=out_dir / f"df_cat_typ_{m}.csv", wandb_key=f"tables/df_cat_typ_{m}")
        _log_df_table(t["emb_agg"], csv_path=out_dir / f"df_cat_emb_{m}.csv", wandb_key=f"tables/df_cat_emb_{m}")
        _log_df_table(t["ctx_agg"], csv_path=out_dir / f"df_cat_ctx_{m}.csv", wandb_key=f"tables/df_cat_ctx_{m}")
        _log_df_table(t["typ_by_layer"], csv_path=out_dir / f"df_cat_typ_{m}_by_layer.csv", wandb_key=f"tables/df_cat_typ_{m}_by_layer")
        _log_df_table(t["emb_by_layer"], csv_path=out_dir / f"df_cat_emb_{m}_by_layer.csv", wandb_key=f"tables/df_cat_emb_{m}_by_layer")
        _log_df_table(t["ctx_by_layer"], csv_path=out_dir / f"df_cat_ctx_{m}_by_layer.csv", wandb_key=f"tables/df_cat_ctx_{m}_by_layer")
        _log_df_table(t["ll"].reset_index(), csv_path=out_dir / f"df_ll_{m}.csv", wandb_key=f"tables/df_ll_{m}")

        _log_df_table(t["typ_emb_agg"], csv_path=out_dir / f"df_cat_typ_emb_{m}.csv", wandb_key=f"tables/df_cat_typ_emb_{m}")
        _log_df_table(t["emb_emb_agg"], csv_path=out_dir / f"df_cat_emb_emb_{m}.csv", wandb_key=f"tables/df_cat_emb_emb_{m}")
        _log_df_table(t["ctx_emb_agg"], csv_path=out_dir / f"df_cat_ctx_emb_{m}.csv", wandb_key=f"tables/df_cat_ctx_emb_{m}")
        _log_df_table(t["typ_emb_by_layer"], csv_path=out_dir / f"df_cat_typ_emb_{m}_by_layer.csv", wandb_key=f"tables/df_cat_typ_emb_{m}_by_layer")
        _log_df_table(t["emb_emb_by_layer"], csv_path=out_dir / f"df_cat_emb_emb_{m}_by_layer.csv", wandb_key=f"tables/df_cat_emb_emb_{m}_by_layer")
        _log_df_table(t["ctx_emb_by_layer"], csv_path=out_dir / f"df_cat_ctx_emb_{m}_by_layer.csv", wandb_key=f"tables/df_cat_ctx_emb_{m}_by_layer")
        _log_df_table(t["ll_emb"].reset_index(), csv_path=out_dir / f"df_ll_emb_{m}.csv", wandb_key=f"tables/df_ll_emb_{m}")

        _log_df_table(t["jac_agg"], csv_path=out_dir / f"df_cat_{m}_jac.csv", wandb_key=f"tables/df_cat_{m}_jac")
        _log_df_table(t["jac_by_layer"], csv_path=out_dir / f"df_cat_{m}_jac_by_layer.csv", wandb_key=f"tables/df_cat_{m}_jac_by_layer")

    _log_df_table(df_clustering_ami, csv_path=out_dir / "df_clustering_ami.csv", wandb_key="tables/df_clustering_ami")
    df_ami_static, df_ami_contextual, df_ami_jaccard, df_ami_emb_jaccard = (
        split_clustering_ami_tables(df_clustering_ami)
    )
    _log_df_table(df_ami_static, csv_path=out_dir / "df_clustering_ami_static.csv",
                  wandb_key="tables/clustering_ami_static_embedding")
    _log_df_table(df_ami_contextual, csv_path=out_dir / "df_clustering_ami_contextual.csv",
                  wandb_key="tables/clustering_ami_contextual")
    _log_df_table(df_ami_jaccard, csv_path=out_dir / "df_clustering_ami_jaccard.csv",
                  wandb_key="tables/clustering_ami_jaccard")
    _log_df_table(df_ami_emb_jaccard, csv_path=out_dir / "df_clustering_ami_emb_jaccard.csv",
                  wandb_key="tables/clustering_ami_emb_jaccard")

    for png_path in sorted(out_dir.glob("*.png")):
        wandb.log({f"plots/{png_path.stem}": wandb.Image(str(png_path))})

    run.finish()


def _log_df_table(df: pd.DataFrame, csv_path: Path, wandb_key: str | None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if wandb_key is not None:
        wandb.log({wandb_key: wandb.Table(dataframe=df)})


def _attach_layer_columns(
    df_item: pd.DataFrame,
    value_by_layer: Dict[int, Dict[Tuple[str, str], float]],
    name: str,
    layers: List[int],
) -> List[str]:
    """Add per-layer columns f"{name}_layer_{l}" plus f"{name}_mean_layers"."""
    for layer in layers:
        df_item[f"{name}_layer_{layer}"] = df_item.apply(
            lambda r, l=layer: value_by_layer[l].get((r["category"], r["item"]), float("nan")),
            axis=1,
        )
    cols = [f"{name}_layer_{l}" for l in layers]
    df_item[f"{name}_mean_layers"] = df_item[cols].mean(axis=1)
    return cols


def attach_paper_columns(
    df_item: pd.DataFrame,
    ctx_cosim_by_layer: Dict[int, Dict[Tuple[str, str], float]],
    jac_by_layer: Dict[int, Dict[Tuple[str, str], float]],
    emb_jac_by_layer: Dict[int, Dict[Tuple[str, str], float]],
    latent_metrics_by_layer: Dict[str, Dict[int, Dict[Tuple[str, str], float]]],
    emb_latent_metrics_by_layer: Dict[str, Dict[int, Dict[Tuple[str, str], float]]],
    layers: List[int],
    emb_jaccard_layers: List[int],
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """Add per-layer contextual cosim, Jaccard, Emb Jaccard, and amplitude-metric columns + means."""
    df_item = df_item.copy()
    ctx_cols = _attach_layer_columns(df_item, ctx_cosim_by_layer, "ctx_cosim", layers)
    jac_cols = _attach_layer_columns(df_item, jac_by_layer, "jaccard", layers)
    emb_jac_cols = _attach_layer_columns(df_item, emb_jac_by_layer, "emb_jaccard", emb_jaccard_layers)

    new_metric_cols: Dict[str, List[str]] = {}
    new_metric_emb_cols: Dict[str, List[str]] = {}
    for m in NEW_METRICS:
        new_metric_cols[m] = _attach_layer_columns(df_item, latent_metrics_by_layer[m], m, layers)
        new_metric_emb_cols[m] = _attach_layer_columns(
            df_item, emb_latent_metrics_by_layer[m], f"emb_{m}", emb_jaccard_layers
        )

    return df_item, ctx_cols, jac_cols, emb_jac_cols, new_metric_cols, new_metric_emb_cols


def per_category_spearman(df_item: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    return (
        df_item.groupby("category", as_index=False)
        .apply(lambda g: pd.Series({
            "n_items": len(g),
            "spearman_rho": spearman(g[x_col].tolist(), g[y_col].tolist()),
        }))
        .reset_index()
        .drop(columns=["index"], errors="ignore"))


def per_category_by_layer_spearman_ycols(
    df_item: pd.DataFrame,
    x_col: str,
    y_col_template: str,
    layers: List[int],
) -> pd.DataFrame:
    """x is a fixed column; y = y_col_template.format(l=l) varies by layer."""
    def _agg(g: pd.DataFrame) -> pd.Series:
        out = {"n_items": len(g)}
        x = g[x_col].tolist()
        for l in layers:
            out[f"layer_{l}"] = spearman(x, g[y_col_template.format(l=l)].tolist())
        return pd.Series(out)
    return df_item.groupby("category", as_index=False).apply(_agg).reset_index().drop(columns=["index"], errors="ignore")


def per_category_by_layer_spearman_paired(
    df_item: pd.DataFrame,
    x_col_template: str,
    y_col_template: str,
    layers: List[int],
) -> pd.DataFrame:
    """Both x and y vary by layer (same layer l paired)."""
    def _agg(g: pd.DataFrame) -> pd.Series:
        out = {"n_items": len(g)}
        for l in layers:
            out[f"layer_{l}"] = spearman(
                g[x_col_template.format(l=l)].tolist(),
                g[y_col_template.format(l=l)].tolist(),
            )
        return pd.Series(out)
    return df_item.groupby("category", as_index=False).apply(_agg).reset_index().drop(columns=["index"], errors="ignore")


def layer_layer_spearman(df_item: pd.DataFrame, layer_cols: List[str], layers: List[int]) -> pd.DataFrame:
    """Layer×layer Spearman matrix pooled over all (category, item) pairs."""
    return pd.DataFrame(
        df_item[layer_cols].to_numpy(dtype=float),
        columns=[str(l) for l in layers],
    ).corr(method="spearman")


def print_summaries(
    df_item: pd.DataFrame,
    df_cat_typ_emb: pd.DataFrame,
    df_cat_typ_ctx: pd.DataFrame,
    df_cat_typ_jac: pd.DataFrame,
    df_cat_typ_emb_jac: pd.DataFrame,
    df_cat_emb_jac: pd.DataFrame,
    df_cat_emb_cosim_emb_jac: pd.DataFrame,
    df_cat_ctx_jac: pd.DataFrame,
    df_cat_ctx_emb_jac: pd.DataFrame,
    new_metric_tables: Dict[str, Dict[str, pd.DataFrame]],
) -> None:
    def macro(df: pd.DataFrame) -> float:
        return float(df["spearman_rho"].mean())

    def micro(x_col: str, y_col: str) -> float:
        return spearman(df_item[x_col].tolist(), df_item[y_col].tolist())

    print("Metric summaries  (macro = mean per-category ρ,  micro = pooled ρ)\n")
    print(f"macro ρ(typicality, emb cosim):      {macro(df_cat_typ_emb):.4f}   "
          f"micro: {micro('typicality', 'cosim_item_category'):.4f}")
    print(f"macro ρ(typicality, ctx cosim):      {macro(df_cat_typ_ctx):.4f}   "
          f"micro: {micro('typicality', 'ctx_cosim_mean_layers'):.4f}")
    print(f"macro ρ(typicality, Jaccard):        {macro(df_cat_typ_jac):.4f}   "
          f"micro: {micro('typicality', 'jaccard_mean_layers'):.4f}")
    print(f"macro ρ(typicality, Emb Jaccard):    {macro(df_cat_typ_emb_jac):.4f}   "
          f"micro: {micro('typicality', 'emb_jaccard_mean_layers'):.4f}")
    print(f"macro ρ(emb cosim,  Jaccard):        {macro(df_cat_emb_jac):.4f}   "
          f"micro: {micro('cosim_item_category', 'jaccard_mean_layers'):.4f}")
    print(f"macro ρ(emb cosim,  Emb Jaccard):    {macro(df_cat_emb_cosim_emb_jac):.4f}   "
          f"micro: {micro('cosim_item_category', 'emb_jaccard_mean_layers'):.4f}")
    print(f"macro ρ(ctx cosim,  Jaccard):        {macro(df_cat_ctx_jac):.4f}   "
          f"micro: {micro('ctx_cosim_mean_layers', 'jaccard_mean_layers'):.4f}")
    print(f"macro ρ(ctx cosim,  Emb Jaccard):    {macro(df_cat_ctx_emb_jac):.4f}   "
          f"micro: {micro('ctx_cosim_mean_layers', 'emb_jaccard_mean_layers'):.4f}")

    for m in NEW_METRICS:
        label = NEW_METRIC_LABELS[m]
        t = new_metric_tables[m]
        print(f"macro ρ(typicality, {label}):        {macro(t['typ_agg']):.4f}   "
              f"micro: {micro('typicality', f'{m}_mean_layers'):.4f}")
        print(f"macro ρ(emb cosim,  {label}):        {macro(t['emb_agg']):.4f}   "
              f"micro: {micro('cosim_item_category', f'{m}_mean_layers'):.4f}")
        print(f"macro ρ(ctx cosim,  {label}):        {macro(t['ctx_agg']):.4f}   "
              f"micro: {micro('ctx_cosim_mean_layers', f'{m}_mean_layers'):.4f}")
        print(f"macro ρ(typicality, Emb {label}):    {macro(t['typ_emb_agg']):.4f}   "
              f"micro: {micro('typicality', f'emb_{m}_mean_layers'):.4f}")
        print(f"macro ρ(emb cosim,  Emb {label}):    {macro(t['emb_emb_agg']):.4f}   "
              f"micro: {micro('cosim_item_category', f'emb_{m}_mean_layers'):.4f}")
        print(f"macro ρ(ctx cosim,  Emb {label}):    {macro(t['ctx_emb_agg']):.4f}   "
              f"micro: {micro('ctx_cosim_mean_layers', f'emb_{m}_mean_layers'):.4f}")
        print(f"macro ρ({label}, Jaccard):           {macro(t['jac_agg']):.4f}   "
              f"micro: {micro(f'{m}_mean_layers', 'jaccard_mean_layers'):.4f}")


def print_clustering_ami(df_clustering_ami: pd.DataFrame) -> None:
    print("\nK-means clustering AMI vs dataset categories (k = #categories)\n")
    for _, row in df_clustering_ami.iterrows():
        print(f"  {row['representation']}: AMI = {row['ami']:.4f}")


def write_summary_txt(
    out_dir: Path,
    cfg: Any,
    layers: List[int],
    lowest_layer: int,
    df_item: pd.DataFrame,
    df_cat_typ_emb: pd.DataFrame,
    df_cat_typ_ctx: pd.DataFrame,
    df_cat_typ_jac: pd.DataFrame,
    df_cat_typ_emb_jac: pd.DataFrame,
    df_cat_emb_jac: pd.DataFrame,
    df_cat_emb_cosim_emb_jac: pd.DataFrame,
    df_cat_ctx_jac: pd.DataFrame,
    df_cat_ctx_emb_jac: pd.DataFrame,
    df_clustering_ami: pd.DataFrame,
    new_metric_tables: Dict[str, Dict[str, pd.DataFrame]],
) -> None:
    def macro(df: pd.DataFrame) -> float:
        return float(df["spearman_rho"].mean())

    def micro(x_col: str, y_col: str) -> float:
        return spearman(df_item[x_col].tolist(), df_item[y_col].tolist())

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"dataset: {cfg.dataset_name} ({cfg.split})\n")
        f.write(f"model: {cfg.sae.model.name}\n")
        f.write(f"static_pooling: {cfg.static_pooling}\n")
        f.write(f"paper_template: {cfg.paper_template!r}\n")
        f.write(f"lowercase: {cfg.lowercase}\n")
        f.write(f"min_items_per_category: {cfg.min_items_per_category}\n")
        f.write(f"SAE release: {cfg.sae.release}\n")
        f.write(f"SAE id template: {cfg.sae.id_template}\n")
        f.write(f"SAE layers: {layers}\n")
        f.write(f"Emb Jaccard SAE layer: {lowest_layer}\n\n")
        f.write(f"macro ρ(typicality, emb cosim):   {macro(df_cat_typ_emb):.4f}  "
                f"micro: {micro('typicality', 'cosim_item_category'):.4f}\n")
        f.write(f"macro ρ(typicality, ctx cosim):   {macro(df_cat_typ_ctx):.4f}  "
                f"micro: {micro('typicality', 'ctx_cosim_mean_layers'):.4f}\n")
        f.write(f"macro ρ(typicality, Jaccard):     {macro(df_cat_typ_jac):.4f}  "
                f"micro: {micro('typicality', 'jaccard_mean_layers'):.4f}\n")
        f.write(f"macro ρ(typicality, Emb Jaccard): {macro(df_cat_typ_emb_jac):.4f}  "
                f"micro: {micro('typicality', 'emb_jaccard_mean_layers'):.4f}\n")
        f.write(f"macro ρ(emb cosim,  Jaccard):     {macro(df_cat_emb_jac):.4f}  "
                f"micro: {micro('cosim_item_category', 'jaccard_mean_layers'):.4f}\n")
        f.write(f"macro ρ(emb cosim,  Emb Jaccard): {macro(df_cat_emb_cosim_emb_jac):.4f}  "
                f"micro: {micro('cosim_item_category', 'emb_jaccard_mean_layers'):.4f}\n")
        f.write(f"macro ρ(ctx cosim,  Jaccard):     {macro(df_cat_ctx_jac):.4f}  "
                f"micro: {micro('ctx_cosim_mean_layers', 'jaccard_mean_layers'):.4f}\n")
        f.write(f"macro ρ(ctx cosim,  Emb Jaccard): {macro(df_cat_ctx_emb_jac):.4f}  "
                f"micro: {micro('ctx_cosim_mean_layers', 'emb_jaccard_mean_layers'):.4f}\n")

        for m in NEW_METRICS:
            label = NEW_METRIC_LABELS[m]
            t = new_metric_tables[m]
            f.write(f"macro ρ(typicality, {label}):     {macro(t['typ_agg']):.4f}  "
                    f"micro: {micro('typicality', f'{m}_mean_layers'):.4f}\n")
            f.write(f"macro ρ(emb cosim,  {label}):     {macro(t['emb_agg']):.4f}  "
                    f"micro: {micro('cosim_item_category', f'{m}_mean_layers'):.4f}\n")
            f.write(f"macro ρ(ctx cosim,  {label}):     {macro(t['ctx_agg']):.4f}  "
                    f"micro: {micro('ctx_cosim_mean_layers', f'{m}_mean_layers'):.4f}\n")
            f.write(f"macro ρ(typicality, Emb {label}): {macro(t['typ_emb_agg']):.4f}  "
                    f"micro: {micro('typicality', f'emb_{m}_mean_layers'):.4f}\n")
            f.write(f"macro ρ(emb cosim,  Emb {label}): {macro(t['emb_emb_agg']):.4f}  "
                    f"micro: {micro('cosim_item_category', f'emb_{m}_mean_layers'):.4f}\n")
            f.write(f"macro ρ(ctx cosim,  Emb {label}): {macro(t['ctx_emb_agg']):.4f}  "
                    f"micro: {micro('ctx_cosim_mean_layers', f'emb_{m}_mean_layers'):.4f}\n")
            f.write(f"macro ρ({label}, Jaccard):        {macro(t['jac_agg']):.4f}  "
                    f"micro: {micro(f'{m}_mean_layers', 'jaccard_mean_layers'):.4f}\n")

        f.write("\nK-means clustering AMI vs dataset categories (k = #categories)\n")
        for _, row in df_clustering_ami.iterrows():
            f.write(f"  {row['representation']}: AMI = {row['ami']:.4f}\n")


if __name__ == "__main__":
    main()
