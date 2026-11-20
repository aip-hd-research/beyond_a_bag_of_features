# Vendored: feature-hedging-paper

This folder has the subset of
[chanind/feature-hedging-paper](https://github.com/chanind/feature-hedging-paper)
(Copyright (c) 2025 David Chanin, MIT license, see `LICENSE`; citation in `CITATION.cff`) that the
toy-model studies in `experiments/preliminary_toy_model/` need. The base is upstream commit `6b6d7d6`.

Modifications compared to upstream:
- `hedging_paper/evals/union_property_eval.py`: new. Union-consistency evaluation (Jaccard / recall / precision of
  the composed-input signature against the union of single-feature signatures).
- `hedging_paper/toy_models/gemma_feature_sampler.py`: new. Greedy selection of near-orthogonal ground-truth
  directions from a pretrained SAE decoder.
- `hedging_paper/toy_models/toy_model.py`, `hedging_paper/toy_models/train_toy_sae.py`: small changes for the
  union-property experiments.
- All other upstream files (LLM SAE training, Matryoshka SAEs, hedging/absorption evals, notebooks) were removed.

The experiment scripts in `experiments/preliminary_toy_model/` were written for this paper on top of this code.
