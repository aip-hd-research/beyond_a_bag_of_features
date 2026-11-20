from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Any

from omegaconf import DictConfig
import torch
import pandas as pd

from transformer_lens import HookedTransformer
from sae_lens import SAE

from sae_set_behaviour.lost_latents.construct_embedding import tokenize_texts
from sae_set_behaviour.typicality.metrics import spearman  

def normalize_text(x: str, lowercase: bool) -> str:
    x = x.strip()
    return x.lower() if lowercase else x


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def make_out_dir(cfg: DictConfig, layers: List[int]) -> Path:
    out_dir = (
        Path(cfg.base_save_dir).expanduser().resolve()
        / str(cfg.sae.model.name)
        / str(cfg.sae.release)
        / f"static_pooling={cfg.static_pooling}"
        / f"layers={','.join(str(l) for l in layers)}"
        / f"lowercase={bool(cfg.lowercase)}"
        / f"min_items={int(cfg.min_items_per_category)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@torch.no_grad()
def collect_sae_token_signatures(model: HookedTransformer, sae: SAE, texts: List[str], device: str, max_length: int, batch_size: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    sig_batches: List[torch.Tensor] = []
    mask_batches: List[torch.Tensor] = []

    hook_name = sae.cfg.metadata.hook_name
    sae_device = sae.W_enc.device
    sae_dtype = sae.W_enc.dtype
    target_len = int(max_length)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        input_ids, attention_mask = tokenize_texts(model, batch_texts, device=device, max_length=max_length)
        _, cache = model.run_with_cache(input_ids, names_filter=lambda n: n == hook_name)

        acts = cache[hook_name]
        del cache, input_ids

        acts = acts.to(device=sae_device, dtype=sae_dtype)

        sae_acts = sae.encode(acts)
        sig = (sae_acts > 0).to(torch.int8).cpu()
        mask = attention_mask.to(torch.int8).cpu()

        cur_len = sig.shape[1]
        if cur_len < target_len:
            pad_len = target_len - cur_len
            sig = torch.nn.functional.pad(sig, (0, 0, 0, pad_len, 0, 0), value=0)
            mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
        elif cur_len > target_len:
            sig = sig[:, :target_len, :]
            mask = mask[:, :target_len]

        del acts, sae_acts
        sig_batches.append(sig)
        mask_batches.append(mask)

    return torch.cat(sig_batches, dim=0), torch.cat(mask_batches, dim=0)


def print_summaries(df_item: pd.DataFrame, df_cat_typ_cos: pd.DataFrame, df_cat_typ_apm: pd.DataFrame, df_cat_cos_apm: pd.DataFrame) -> None:
    """
    Macro = mean over categories of per-category Spearman
    Micro = Spearman computed over all items pooled
    """
    macro_typ_cos = float(df_cat_typ_cos["spearman_rho"].mean())
    macro_typ_apm = float(df_cat_typ_apm["spearman_rho"].mean())
    macro_cos_apm = float(df_cat_cos_apm["spearman_rho"].mean())

    micro_typ_cos = spearman(df_item["typicality"].tolist(), df_item["cosim_item_category"].tolist())
    micro_typ_apm = spearman(df_item["typicality"].tolist(), df_item["apm_sym_mean_layers"].tolist())
    micro_cos_apm = spearman(df_item["cosim_item_category"].tolist(), df_item["apm_sym_mean_layers"].tolist())

    print("Metric summaries")
    print("Macro = mean (per-category ρ), Micro = pooled ρ over all items\n")
    print(f"Macro Spearman(typicality, cosim): {macro_typ_cos:.4f}")
    print(f"Micro Spearman(typicality, cosim): {micro_typ_cos:.4f}")
    print(f"Macro Spearman(typicality, APM):   {macro_typ_apm:.4f}")
    print(f"Micro Spearman(typicality, APM):   {micro_typ_apm:.4f}")
    print(f"Macro Spearman(cosim, APM):        {macro_cos_apm:.4f}")
    print(f"Micro Spearman(cosim, APM):        {micro_cos_apm:.4f}")


def save_tables(out_dir: Path, df_item: pd.DataFrame, df_cat_typ_cos: pd.DataFrame, df_cat_typ_apm: pd.DataFrame, df_cat_cos_apm: pd.DataFrame, df_cat_typ_apm_by_layer: pd.DataFrame, df_cat_cos_apm_by_layer: pd.DataFrame, df_ll: pd.DataFrame) -> None:
    """Write all CSV outputs."""
    df_cat_typ_cos.sort_values("spearman_rho", ascending=False).to_csv(out_dir / "per_category_spearman_typ_vs_cosim.csv", index=False)
    df_cat_typ_apm.sort_values("spearman_rho", ascending=False).to_csv(out_dir / "per_category_spearman_typ_vs_apm_mean.csv", index=False)
    df_cat_cos_apm.sort_values("spearman_rho", ascending=False).to_csv(out_dir / "per_category_spearman_cosim_vs_apm_mean.csv", index=False)

    df_cat_typ_apm_by_layer.to_csv(out_dir / "per_category_spearman_typ_vs_apm_by_layer.csv", index=False)
    df_cat_cos_apm_by_layer.to_csv(out_dir / "per_category_spearman_cosim_vs_apm_by_layer.csv", index=False)

    df_item.to_csv(out_dir / "per_item_scores.csv", index=False)
    df_ll.to_csv(out_dir / "layer_layer_spearman_apm_scores.csv")




def write_summary_txt(out_dir: Path, cfg: Any, layers: List[int], apm_max_len: int,
    df_cat_typ_cos: pd.DataFrame,
    df_cat_typ_apm: pd.DataFrame,
    df_cat_cos_apm: pd.DataFrame,
    df_item: pd.DataFrame) -> None:
    macro_typ_cos = float(df_cat_typ_cos["spearman_rho"].mean())
    macro_typ_apm = float(df_cat_typ_apm["spearman_rho"].mean())
    macro_cos_apm = float(df_cat_cos_apm["spearman_rho"].mean())

    micro_typ_cos = spearman(df_item["typicality"].tolist(), df_item["cosim_item_category"].tolist())
    micro_typ_apm = spearman(df_item["typicality"].tolist(), df_item["apm_sym_mean_layers"].tolist())
    micro_cos_apm = spearman(df_item["cosim_item_category"].tolist(), df_item["apm_sym_mean_layers"].tolist())

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"dataset: {cfg.dataset_name} ({cfg.split})\n")
        f.write(f"model: {cfg.sae.model.name}\n")
        f.write(f"static_pooling: {cfg.static_pooling}\n")
        f.write(f"lowercase: {cfg.lowercase}\n")
        f.write(f"min_items_per_category: {cfg.min_items_per_category}\n")
        f.write(f"SAE release: {cfg.sae.release}\n")
        f.write(f"SAE id template: {cfg.sae.id_template}\n")
        f.write(f"SAE layers: {layers}\n")
        f.write(f"APM max_length: {apm_max_len}\n")
        f.write(f"macro_typ_cos: {macro_typ_cos}\n")
        f.write(f"micro_typ_cos: {micro_typ_cos}\n")
        f.write(f"macro_typ_apm: {macro_typ_apm}\n")
        f.write(f"micro_typ_apm: {micro_typ_apm}\n")
        f.write(f"macro_cos_apm: {macro_cos_apm}\n")
        f.write(f"micro_cos_apm: {micro_cos_apm}\n")