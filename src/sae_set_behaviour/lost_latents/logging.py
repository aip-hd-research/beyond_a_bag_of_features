from pathlib import Path
from typing import Dict
import pandas as pd
import wandb
import torch

def _metric_dicts_to_df(
    mean_by_layer_by_k: Dict[int, Dict[int, float]],
    std_by_layer_by_k: Dict[int, Dict[int, float]],
    metric_name: str,
    mix_mode: str,
    setup_name: str,
) -> pd.DataFrame:
    rows = []
    for layer, mean_by_k in mean_by_layer_by_k.items():
        for k, mean_val in mean_by_k.items():
            rows.append({
                "layer": int(layer),
                "k": int(k),
                "mean": float(mean_val),
                "std": float(std_by_layer_by_k.get(layer, {}).get(k, float("nan"))),
                "metric": metric_name,
                "mix_mode": mix_mode,
                "setup_name": setup_name,
            })
    return pd.DataFrame(rows)

def _log_plot_and_data(
    plot_path: Path,
    mean_by_layer_by_k: Dict[int, Dict[int, float]],
    std_by_layer_by_k: Dict[int, Dict[int, float]],
    metric_name: str,
    mix_mode: str,
    setup_name: str,
) -> None:
    df = _metric_dicts_to_df(
        mean_by_layer_by_k=mean_by_layer_by_k,
        std_by_layer_by_k=std_by_layer_by_k,
        metric_name=metric_name,
        mix_mode=mix_mode,
        setup_name=setup_name,
    )
    csv_path = plot_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    wandb.log({
        f"{mix_mode}/{setup_name}/{metric_name}_plot": wandb.Image(str(plot_path)),
        f"{mix_mode}/{setup_name}/{metric_name}_data": wandb.Table(dataframe=df),
    })


def _tensor_metric_dicts_to_df(
    metric_by_layer_by_k: Dict[int, Dict[int, torch.Tensor]],
    metric_name: str,
    setup_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for layer, by_k in metric_by_layer_by_k.items():
        for k, vec in by_k.items():
            vals = vec.detach().cpu().tolist()
            for idx, val in enumerate(vals):
                rows.append({
                    "layer": int(layer),
                    "k": int(k),
                    "trial_idx": int(idx),
                    "value": float(val),
                    "metric": metric_name,
                    "setup_name": setup_name,
                })
    return pd.DataFrame(rows)