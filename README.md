# Beyond a Bag of Features: Set-Level Instability in Sparse Autoencoders

Code for the paper *Beyond a Bag of Features: Set-Level Instability in Sparse Autoencoders*.

We treat the set of active SAE latents (the *SAE signature*) as the unit of analysis and test whether
signatures behave like a bag of features: whether they recover human category structure and typicality
(Experiment 1), and whether they stay consistent when a concept is refined compositionally,
e.g. *shovel* → *yellow shovel* (Experiment 2).

## Repository layout

| Folder | Paper | What it does |
|---|---|---|
| `experiments/preliminary_toy_model/` | Sec. 3, App. B.1 | Union consistency of SAE signatures in an LRH-aligned toy model (orthogonal, scaling and Gemma-decoder-sampled setups) |
| `experiments/preliminary_signature_clusters/` | Sec. 3, App. B.2 | Interactive latent-set cluster browser on The Pile (APM similarity) |
| `experiments/experiment_1_typicality/` | Sec. 4, App. C–E | Category recovery (AMI) and human typicality rankings from SAE signatures; non-binary measures |
| `experiments/experiment_2_lost_latents/` | Sec. 5, App. F | Lost latents under adjective–noun refinement: lost rates, upstream presence, pre-activation regimes, generalization (German, category–instance), threshold choice, activation magnitudes |
| `experiments/appendix_interference/` | App. G | Steering with SAE decoder directions as a probe of interference |
| `experiments/appendix_autointerp/` | App. G.2 | Auto-interpretability of lost vs. surviving vs. control latents (SAEBench) |
| `src/sae_set_behaviour/` | – | Shared code: model/SAE loading (`util.py`), lost-latent utilities (`lost_latents/`), typicality utilities (`typicality/`) |
| `configs/` | – | Hydra configs: one group per experiment folder, SAE definitions (`configs/sae/`, App. A), W&B settings |
| `data/` | – | Adjective–noun chains (`data.csv`) and their German translations (`data_german.csv`) |
| `third_party/` | – | Vendored feature-hedging toy-model code (see *Acknowledgements*) and a SAEBench patch |
| `utility_scripts/spawn_sweep_master.py` | – | Creates a W&B grid sweep of one experiment over all SAEs in `configs/sae/` |

The human typicality / category–instance data (`CShani/human-concepts`) and The Pile are loaded from the Hugging Face Hub.

## Setup

We use [pixi](https://pixi.sh). There are three environments:

```bash
pixi install                 # default: all experiments except the two below
pixi install -e autointerp   # + SAEBench and OpenAI, for experiments/appendix_autointerp
pixi install -e toy          # separate environment for experiments/preliminary_toy_model
```

The toy-model code builds on the feature-hedging codebase, which targets SAELens 5.x, so it gets its own environment.
Some models (Gemma 3, Llama 3.1) are gated on the Hugging Face Hub, so log in with `huggingface-cli login` first.
Results are logged to [Weights & Biases](https://wandb.ai). Set `WANDB_ENTITY` (or pass `wandb.entity=...`), or run with `WANDB_MODE=offline`.

Run all commands from the repository root. Outputs go to `results/<experiment>/`.

## Running the experiments

All language-model experiments are Hydra scripts. Select the experiment config with `+<experiment_folder>=base`
and the SAE with `sae=<model>/<sae>`, choosing from `configs/sae/`:

```bash
# Experiment 1: typicality and category recovery
pixi run python experiments/experiment_1_typicality/main.py +experiment_1_typicality=base sae=gemma-3-270m/16k_big

# Experiment 2: lost latents
#   (i) lost rates, fixed-base and stepwise
pixi run python experiments/experiment_2_lost_latents/main_by_input_data.py +experiment_2_lost_latents=base sae=gemma-3-270m/16k_big
#   (ii)+(iii) upstream presence and pre-activation regimes: full sequence (main lines) and noun signature (insets)
pixi run python experiments/experiment_2_lost_latents/what_happened_to_lost_features_full_sequence.py +experiment_2_lost_latents=base sae=gemma-3-270m/16k_big
pixi run python experiments/experiment_2_lost_latents/what_happened_to_lost_features_noun.py +experiment_2_lost_latents=base sae=gemma-3-270m/16k_big
#   activation magnitudes of lost latents (App. F.4)
pixi run python experiments/experiment_2_lost_latents/amplitude_lost_vs_general.py +experiment_2_lost_latents=amplitude_lost_vs_general sae=gemma-3-270m/16k_big

# Interference / steering (App. G)
pixi run python experiments/appendix_interference/main_by_sae_features.py +appendix_interference=base sae=gemma-3-270m/16k_big

# Auto-interpretability (App. G.2); needs OPENAI_API_KEY
pixi run -e autointerp python experiments/appendix_autointerp/run_multilayer_standard_interp.py +appendix_autointerp=base sae=gemma-3-270m/16k_medium

# Cluster browser (Sec. 3 / App. B.2); interactive, run in a terminal
pixi run python experiments/preliminary_signature_clusters/main.py +preliminary_signature_clusters=base
```

By default, Experiment 2 runs all three datasets (`english`, `german`, `human_typicality`) in one invocation. Restrict this with e.g. `data_cases=[english]`.

### Sweeping over all models and SAEs

```bash
pixi run python utility_scripts/spawn_sweep_master.py \
    experiment_script_path=experiments/experiment_2_lost_latents/main_by_input_data.py
# for the magnitude experiment, also pass experiment_config_include_glob=amplitude_lost_vs_general.yaml
pixi run wandb agent <entity>/<project>/<sweep_id>    # command is printed by the script
```

### Figures

Each experiment folder has the notebook(s) that produce the paper figures from the logged W&B tables:

| Notebook | Figures |
|---|---|
| `experiment_1_typicality/paper_plots_with_ami.ipynb` | Fig. 2, AMI overview (App. C), binary typicality rankings (App. D) |
| `experiment_1_typicality/paper_plots_additional_metrics.ipynb` | Non-binary signals (App. E) |
| `experiment_2_lost_latents/plots_lost_features.ipynb` | Lost-latent Sankey diagram, main lost-latent figure, k-dependence and stepwise lost rates (App. F) |
| `experiment_2_lost_latents/plots_lost_features_all_data.ipynb` | Per-model sweep grids incl. German and category–instance rows (App. F) |
| `experiment_2_lost_latents/threshold_choice.ipynb` | Threshold θ for non-JumpReLU SAEs (App. F.1) |
| `experiment_2_lost_latents/plots_amplitude.ipynb` | Activation magnitudes of lost latents (App. F.4; figure `amplitude_hist__layerQ3__k1.png`) and the overlap/quartile statistics quoted there |
| `appendix_interference/plots_sae_features.ipynb` | Steering figures (App. G) |
| `appendix_autointerp/plot_autointerp_grid.ipynb` | Auto-interpretability grid (App. G.2) |

The notebooks read from the W&B projects named in `configs/` (e.g. `experiment_2_lost_latents`) under `$WANDB_ENTITY`, falling back to your default entity. If you logged to a different project, change the project constant at the top of the notebook.

### Toy-model studies (Sec. 3, App. B.1)

These scripts are argparse-based and run in the `toy` environment. Trained SAEs and plots go to `results/preliminary_toy_model/`.

```bash
# Orthogonal setup (D=50): trains all N/widths/seeds, then plots per N (N=70 -> Fig. 1a)
pixi run -e toy python experiments/preliminary_toy_model/union_property_experiment.py --p 0.2
pixi run -e toy python experiments/preliminary_toy_model/make_combined_plot.py --p 0.2 --ns 10 50 100

# Scaling setup (D=1000)
pixi run -e toy python experiments/preliminary_toy_model/union_property_experiment_scaling.py --d 1000 --p 0.2
pixi run -e toy python experiments/preliminary_toy_model/make_combined_plot.py --p 0.2 --d 1000 --ns 500 1000 5000 --log-x \
    --sae-root results/preliminary_toy_model/saes/union_property_scaling_D1000_p020 \
    --plot-root results/preliminary_toy_model/plots/union_property_scaling_D1000_p020

# Gemma-decoder-sampled setup (D=640). First export the decoder of the Gemma-3-270M layer-9 16k SAE (default env):
pixi run python -c "import torch; from sae_lens import SAE; \
sae = SAE.from_pretrained('gemma-scope-2-270m-pt-res-all', 'layer_9_width_16k_l0_big'); \
torch.save(sae.W_dec.detach().cpu(), 'results/preliminary_toy_model/gemma3_270m_layer9_16k_W_dec.pt')"
pixi run -e toy python experiments/preliminary_toy_model/union_property_experiment_gemma.py --layer 9 --p 0.2 --n-max 2500
pixi run -e toy python experiments/preliminary_toy_model/make_combined_plot.py --p 0.2 --d 640 --ns 500 1500 2500 --log-x \
    --out-prefix sampling_combined \
    --sae-root results/preliminary_toy_model/saes/union_property_gemma_layer9_thr020_p020 \
    --plot-root results/preliminary_toy_model/plots/union_property_gemma_layer9_thr020_p020
```

Create `results/preliminary_toy_model/` before the export step if it doesn't exist yet.

## Acknowledgements

The toy-model studies build on the code of
[*Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders*](https://arxiv.org/abs/2505.11756)
(Chanin, Dulka & Garriga-Alonso, 2025; [chanind/feature-hedging-paper](https://github.com/chanind/feature-hedging-paper), MIT license).
The subset we use is vendored, with our modifications, in `third_party/feature_hedging/` (see its `NOTICE.md`). If you use the toy-model code, please also cite:

```bibtex
@article{chanin2025feature,
  title   = {Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders},
  author  = {Chanin, David and Dulka, Tom{\'a}{\v{s}} and Garriga-Alonso, Adri{\`a}},
  journal = {arXiv preprint arXiv:2505.11756},
  year    = {2025}
}
```

Auto-interpretability uses [SAEBench](https://github.com/adamkarvonen/SAEBench) 0.6.0.
`third_party/sae_bench_0.6.0.patch` contains our memory/device fixes (chunked SAE encoding for 262k-wide SAEs, multi-GPU models) and OpenAI rate-limit retries. They don't change results.
Apply the patch inside the installed package if you hit out-of-memory or rate-limit errors.
Models and SAEs are loaded through [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [SAELens](https://github.com/decoderesearch/SAELens).
