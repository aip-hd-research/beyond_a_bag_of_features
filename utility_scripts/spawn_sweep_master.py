"""
Create a wandb grid sweep of one experiment script over all SAE configs in configs/sae.

Example (from the repo root):
    python utility_scripts/spawn_sweep_master.py \
        experiment_script_path=experiments/experiment_2_lost_latents/main_by_input_data.py
then start agents with the printed `wandb agent ...` command.
"""

import os
from pathlib import Path
import subprocess
import hydra
from omegaconf import DictConfig
from dataclasses import dataclass
from typing import List, Optional, Tuple
from omegaconf import MISSING
import wandb
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from termcolor import colored

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"

@dataclass
class HydraConfig:
    experiment_script_path: str = "experiments/experiment_2_lost_latents/main_by_input_data.py"  # relative to the repo root (or absolute)
    experiment_config_group: Optional[str] = None
    experiment_config_include_glob: str = "base.yaml"
    experiment_config_exclude_glob: str = ""
    sae_include_glob: str = "**/*.yaml"
    sae_exclude_glob: str = "**/base.yaml"  # comma-separated list of globs
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None

WandbConfig = DictConfig

ConfigStore.instance().store(name="config", node=HydraConfig)

@hydra.main(version_base=None, config_name="config", config_path="../configs")
def main(args: HydraConfig) -> None:
    script = Path(args.experiment_script_path)
    args.experiment_script_path = str(script if script.is_absolute() else REPO_ROOT / script)
    if args.experiment_config_group is None:
        args.experiment_config_group = get_experiment_config_group_from_script_path(args.experiment_script_path)
    if args.wandb_entity is None:
        args.wandb_entity, args.wandb_project = extract_wandb_entity_and_project_from_experimemt_configs(args)
    sweep_config = build_config(args)
    launch_sweep(sweep_config)

def build_config(args: HydraConfig) -> WandbConfig:
    experiment_configs = collect_experiment_configs(args)
    sae_configs = collect_sae_configs(args)
    sweep_configuration = {
        "program": args.experiment_script_path,
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "method": "grid",
        "parameters": {
            f"+{args.experiment_config_group}": {"values": experiment_configs},
            "sae": {"values": sae_configs},
            },
        "command": ["${interpreter}", "${program}", "${args_no_hyphens}"],
    }
    return sweep_configuration

def extract_wandb_entity_and_project_from_experimemt_configs(args: HydraConfig) -> Tuple[str, str]:
    """Assumes all experiments have the same wandb entity and project."""
    experiment_config_path = collect_experiment_configs(args)[0]
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base=None, config_path="../configs"):
        config = hydra.compose(config_name=f"{args.experiment_config_group}/{experiment_config_path}")
    return config.wandb.entity, config.wandb.project

def get_experiment_config_group_from_script_path(experiment_script_path: str) -> str:
    return Path(experiment_script_path).parent.name

def collect_experiment_configs(args: HydraConfig) -> List[str]:
    base_path = CONFIG_ROOT / args.experiment_config_group
    return [str(remove_file_extension(strip_path_prefix(path, base_path))) for path in collect_paths(base_path, args.experiment_config_include_glob, args.experiment_config_exclude_glob)]

def collect_sae_configs(args: HydraConfig) -> List[str]:
    base_path = CONFIG_ROOT / "sae"
    return [str(remove_file_extension(strip_path_prefix(path, base_path))) for path in collect_paths(base_path, args.sae_include_glob, args.sae_exclude_glob)]

def remove_file_extension(path: Path) -> Path:
    return path.with_suffix("")

def strip_path_prefix(path: Path, prefix: Path) -> Path:
    return path.relative_to(prefix)

def collect_paths(base_path: Path, include_glob: str, exclude_glob: str) -> List[Path]:
    """Both globs may be comma-separated lists of patterns; an empty exclude excludes nothing."""
    def expand(globs: str) -> set:
        return {path for pattern in globs.split(",") if pattern.strip() for path in base_path.rglob(pattern.strip())}
    return sorted(expand(include_glob) - expand(exclude_glob))

def launch_sweep(sweep_config: WandbConfig) -> None:
    sweep_id = wandb.sweep(sweep=sweep_config, entity=sweep_config["entity"], project=sweep_config["project"])
    print("To start an agent on this sweep, run: " + colored(f"wandb agent {sweep_config['entity']}/{sweep_config['project']}/{sweep_id}", "green", attrs=["bold"]))


if __name__ == "__main__":
    main()