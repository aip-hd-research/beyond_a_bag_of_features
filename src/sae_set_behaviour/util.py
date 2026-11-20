from collections import defaultdict
from dataclasses import dataclass
import re
import time
from typing import Dict, Generator, Optional, Tuple, List, Callable
from omegaconf import II, MISSING, DictConfig
import omegaconf
from transformer_lens import HookedTransformer
from sae_lens import SAE
from torch.utils.data import DataLoader
from functools import partial, wraps
from pathlib import Path
from os import makedirs
from tqdm import tqdm
from datasets import Dataset
from jaxtyping import Float
import torch
import wandb


@dataclass
class ModelConfig:
    name: str = MISSING
    device: str = "cuda"
    dtype: str = "float32"
    batch_size: int = 1 # set largest batch size so that running the model on batch_size * max_length tokens will fit in memory
    n_devices: int = 1
    fold_ln: bool = True
    center_writing_weights: bool = False
    center_unembed: bool = False
    max_length: Optional[int] = None

@dataclass
class SAEConfig:
    model: ModelConfig = MISSING
    release: Optional[str] = None
    hook_name_template: Optional[str] = None
    id_template: Optional[str] = None
    id_template_neuronpedia: Optional[str] = None
    interpretability_layers: Optional[List[int]] = None
    layers: List[int] = MISSING
    device: str = II(".model.device")
    dtype: str = II(".model.dtype")


@dataclass
class SetupConfig:
    sae: SAEConfig = MISSING
    base_save_dir: str = MISSING
    num_workers: int = 4
    device: str = II(".sae.device") # syntactic sugar for using the sae.device
    dtype: str = II(".sae.dtype") # syntactic sugar for using the sae.dtype


def setup(args: SetupConfig, dataset: Dataset) -> Tuple[HookedTransformer, List[SAE], DataLoader, Path]:
    device = args.device
    model = HookedTransformer.from_pretrained(
        model_name=args.sae.model.name,
        device=device,
        fold_ln=args.sae.model.fold_ln,
        center_writing_weights=args.sae.model.center_writing_weights,
        center_unembed=args.sae.model.center_unembed,
        dtype=args.sae.model.dtype,
        n_devices=args.sae.model.n_devices,
    )
    saes = [
        SAE.from_pretrained(
            release=args.sae.release,
            sae_id=args.sae.id_template.format(i),
            device="cpu",
            dtype=args.dtype,
        )
        for i in tqdm(args.sae.layers, desc="Loading SAEs")
    ]
    for sae in saes:
        sae.eval()

    dataloader = DataLoader(
        dataset, 
        batch_size=args.sae.model.batch_size, 
        num_workers=args.num_workers,
        collate_fn=lambda batch: model.tokenizer(
            [sample['text'] for sample in batch], 
            padding=True, 
            truncation=True,
            return_tensors="pt", 
            max_length=args.sae.model.max_length if args.sae.model.max_length is not None else saes[0].cfg.metadata.context_size
        )
    )

    save_dir = Path(args.base_save_dir) / Path(args.sae.model.name) / Path(args.sae.release)
    makedirs(save_dir, exist_ok=True)

    return model, saes, dataloader, save_dir


def build_dummy_dataset() -> Dataset:
    return Dataset.from_dict({"text": ["dummy"]}) 


def collect_activations(model: HookedTransformer, dataloader: DataLoader, hook_points: List[str], n_tokens: int) -> Dict[str, List[torch.Tensor]]:
    activations = defaultdict(list)
    with torch.no_grad(), tqdm(total=n_tokens, desc="Collecting activations") as pbar:
        n = 0   
        for batch in dataloader:
            if n > n_tokens:
                break

            _, cache = model.run_with_cache(batch["input_ids"])
            for hook_point in hook_points:
                mask = batch["attention_mask"].to(cache[hook_point].device)
                activations[hook_point].append(cache[hook_point][mask > 0].detach().cpu())
                del mask
            num_tokens_in_batch = batch["attention_mask"].sum().cpu().item()
            n += num_tokens_in_batch
            pbar.update(num_tokens_in_batch)
            del batch, cache

    return {k: torch.concatenate(activation)[:n_tokens] for k, activation in activations.items()}


def divide_into_batches(data: Float[torch.Tensor, "n ..."], batch_size: int, drop_last: bool = False) -> Generator[Float[torch.Tensor, "batch_size ..."], None, None]:
    idx = 0
    while idx < len(data):
        if len(data) - idx > batch_size or not drop_last:
            yield data[idx: min(idx + batch_size, len(data))]
            idx += batch_size


@dataclass
class WandbConfig:
    entity: Optional[str] = None  # None -> your default W&B entity
    project: str = MISSING
    name: str = MISSING
    group: str = MISSING
    offline: bool = False


def init_wandb(wandb_config: WandbConfig, global_config: DictConfig, job_type: str = None) -> None:
    wandb.init(
        entity=wandb_config.entity,
        project=wandb_config.project,
        name=wandb_config.name,
        group=wandb_config.group,
        job_type=job_type,
        mode="offline" if wandb_config.offline else None,  # None -> respect WANDB_MODE (default: online)
        config=omegaconf.OmegaConf.to_container(
            global_config, resolve=True, throw_on_missing=True
        ),
        reinit="finish_previous",
    )

def infer_layer_int(*, sae_hook_name: str, sae_obj: object) -> int:
    try:
        layer = getattr(getattr(getattr(sae_obj, "cfg", None), "metadata", None), "layer")
        if layer is not None:
            return int(layer)
    except Exception:
        pass

    hook = str(sae_hook_name)
    m = re.search(r"(?:^|\b)blocks\.(\d+)\.", hook)
    if m:
        return int(m.group(1))
    if hook.startswith("blocks."):
        try:
            return int(hook.split(".", 2)[1])
        except Exception:
            pass

    raise ValueError(
        "Could not infer SAE layer. Expected `sae.cfg.metadata.layer` or a hook name like `blocks.<L>.…` "
        f"but got hook_name={sae_hook_name!r}."
    )