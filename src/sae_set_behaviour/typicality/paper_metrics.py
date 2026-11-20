from __future__ import annotations

from typing import Any, Dict, List, Tuple
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE


PAPER_TEMPLATE = "This is a {word}. "


def make_prompted_text(word: str, template: str = PAPER_TEMPLATE) -> str:
    return template.format(word=word)


def find_word_token_span(tokenizer, prompt: str, word: str) -> Tuple[int, int]:
    """
    Return (start, end) exclusive token index span for `word` inside `prompt`.
    Uses offset_mapping for robust subword alignment including BOS tokens.
    BOS/special tokens have (0,0) offsets and are naturally excluded.
    """
    char_start = prompt.find(word)
    if char_start == -1:
        raise ValueError(f"{word!r} not found in prompt {prompt!r}")
    char_end = char_start + len(word)

    enc = tokenizer(prompt, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    indices = [
        i for i, (ts, te) in enumerate(offsets)
        if ts < char_end and te > char_start
    ]
    if not indices:
        raise ValueError(f"No tokens span {word!r} in {prompt!r}")
    return indices[0], indices[-1] + 1


@torch.no_grad()
def collect_static_word_token_embeddings(
    model: HookedTransformer,
    words: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token static W_E embeddings for each word (no prompt template).

    Returns:
      word_embs: [N, W, d_model] padded, CPU float32
      word_mask: [N, W] int8, 1 for real word tokens
    """
    W_E = model.W_E
    d_model = model.cfg.d_model

    token_rows: List[torch.Tensor] = []
    for word in words:
        ids = model.tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(ids) == 0:
            token_rows.append(torch.zeros(0, d_model))
            continue
        tok_ids = torch.tensor(ids, device=W_E.device, dtype=torch.long)
        token_rows.append(W_E[tok_ids].detach().float().cpu())

    max_word_toks = max((t.shape[0] for t in token_rows), default=1)
    max_word_toks = max(max_word_toks, 1)

    N = len(words)
    word_embs = torch.zeros(N, max_word_toks, d_model, dtype=torch.float32)
    word_mask = torch.zeros(N, max_word_toks, dtype=torch.int8)

    for i, tok_embs in enumerate(token_rows):
        n = tok_embs.shape[0]
        if n == 0:
            continue
        word_embs[i, :n] = tok_embs
        word_mask[i, :n] = 1

    return word_embs, word_mask


@torch.no_grad()
def collect_prompted_word_activations_per_layer(
    model: HookedTransformer,
    words: List[str],
    hook_names_by_layer: Dict[int, str],
    device: str,
    template: str = PAPER_TEMPLATE,
    batch_size: int = 8,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], torch.Tensor]:
    """
    Run each word through the model under `template` and extract residual stream
    activations at the target word tokens only, for each layer.

    Returns:
      avg_residual_by_layer[layer]: [N, d_model]     mean over word tokens, CPU float32
      word_acts_by_layer[layer]:    [N, W, d_model]   per-token acts, padded, CPU float32
      word_mask:                    [N, W] int8        1 for real word tokens
    where W = max word token count across all words.
    """
    prompts = [make_prompted_text(w, template) for w in words]
    hook_name_set = set(hook_names_by_layer.values())

    # Precompute token spans using same default tokenizer settings as the model run
    spans: List[Tuple[int, int]] = [
        find_word_token_span(model.tokenizer, p, w)
        for p, w in zip(prompts, words)
    ]
    max_word_toks = max(e - s for s, e in spans)

    avg_batches: Dict[int, List[torch.Tensor]] = {l: [] for l in hook_names_by_layer}
    word_batches: Dict[int, List[torch.Tensor]] = {l: [] for l in hook_names_by_layer}
    mask_batches: List[torch.Tensor] = []

    for i in range(0, len(words), batch_size):
        batch_words = words[i:i + batch_size]
        batch_prompts = prompts[i:i + batch_size]
        batch_spans = spans[i:i + batch_size]

        enc = model.tokenizer(batch_prompts, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        _, cache = model.run_with_cache(input_ids, names_filter=lambda n: n in hook_name_set)

        B = len(batch_words)
        batch_mask = torch.zeros(B, max_word_toks, dtype=torch.int8)
        for b_idx, (s, e) in enumerate(batch_spans):
            n = min(e - s, max_word_toks)
            batch_mask[b_idx, :n] = 1
        mask_batches.append(batch_mask)

        for layer, hook_name in hook_names_by_layer.items():
            acts = cache[hook_name].float().cpu()  # [B, T, d_model]
            d = acts.shape[-1]

            avg_list: List[torch.Tensor] = []
            word_list: List[torch.Tensor] = []

            for b_idx, (s, e) in enumerate(batch_spans):
                word_acts = acts[b_idx, s:e]  # [n_word_toks, d_model]
                avg_list.append(word_acts.mean(dim=0))

                n = min(word_acts.shape[0], max_word_toks)
                padded = torch.zeros(max_word_toks, d)
                padded[:n] = word_acts[:n]
                word_list.append(padded)

            avg_batches[layer].append(torch.stack(avg_list, dim=0))
            word_batches[layer].append(torch.stack(word_list, dim=0))

        del cache, input_ids, enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_residual_by_layer = {l: torch.cat(b, dim=0) for l, b in avg_batches.items()}
    word_acts_by_layer = {l: torch.cat(b, dim=0) for l, b in word_batches.items()}
    word_mask = torch.cat(mask_batches, dim=0)

    return avg_residual_by_layer, word_acts_by_layer, word_mask


@torch.no_grad()
def compute_sae_union_per_layer(
    sae_by_layer: Dict[int, SAE],
    word_acts_by_layer: Dict[int, torch.Tensor],
    word_mask: torch.Tensor,
    layers: List[int],
    sae_device: str = "cuda",
    encode_batch_size: int = 16,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    For each layer, encode word-token activations through the SAE and take
    the union of active features over the word tokens. Also mean-pools the
    continuous (pre-binarization) latent activations over the word tokens,
    as the amplitude-preserving counterpart to the binary union.

    word_acts_by_layer[layer]: [N, W, d_model]
    word_mask:                 [N, W] int8

    Returns:
      union_by_layer[layer]:       [N, d_sae] int8   — 1 if feature active in ANY word token
      mean_latent_by_layer[layer]: [N, d_sae] float32 — masked mean of latent activations over word tokens
    """
    union_by_layer: Dict[int, torch.Tensor] = {}
    mean_latent_by_layer: Dict[int, torch.Tensor] = {}

    for layer in layers:
        sae = sae_by_layer[layer].to(sae_device)
        sae.eval()
        sae_dtype = sae.W_enc.dtype

        word_acts = word_acts_by_layer[layer]  # [N, W, d_model]
        N, W, d_model = word_acts.shape

        union_batches: List[torch.Tensor] = []
        mean_latent_batches: List[torch.Tensor] = []
        for i in range(0, N, encode_batch_size):
            acts_b = word_acts[i:i + encode_batch_size]  # [B, W, d_model]
            mask_b = word_mask[i:i + encode_batch_size]  # [B, W]
            B = acts_b.shape[0]

            acts_flat = acts_b.reshape(B * W, d_model).to(sae_device, dtype=sae_dtype)
            sae_acts = sae.encode(acts_flat)                      # [B*W, d_sae]
            binary_flat = (sae_acts > 0).to(torch.int8).cpu()    # [B*W, d_sae]
            binary = binary_flat.view(B, W, -1)                  # [B, W, d_sae]

            mask_exp = mask_b.unsqueeze(-1).expand_as(binary)
            binary = binary * mask_exp

            union_b = (binary.sum(dim=1) > 0).to(torch.int8)    # [B, d_sae]
            union_batches.append(union_b)

            sae_acts_cpu = sae_acts.cpu().float().view(B, W, -1)  # transfer in native dtype, upcast on CPU
            mask_f = mask_b.unsqueeze(-1).float()                 # [B, W, 1]
            # `* mask_f` alone assumes masked-out positions are finite; SAEs with
            # constant_norm_rescale produce NaN for exact-zero padding rows (0/0 in
            # the norm rescale), and NaN * 0 stays NaN, silently poisoning the mean.
            # torch.where zeroes those positions outright, matching the masking's
            # original intent without altering any already-finite value.
            masked = torch.where(mask_b.unsqueeze(-1).bool(), sae_acts_cpu, torch.zeros((), dtype=sae_acts_cpu.dtype))
            mean_latent_b = masked.sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
            mean_latent_batches.append(mean_latent_b)

            del acts_flat, sae_acts, binary_flat, binary, acts_b, mask_b, sae_acts_cpu, mask_f

        union_by_layer[layer] = torch.cat(union_batches, dim=0)  # [N, d_sae]
        mean_latent_by_layer[layer] = torch.cat(mean_latent_batches, dim=0)  # [N, d_sae]

        sae_by_layer[layer].to("cpu")
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return union_by_layer, mean_latent_by_layer


@torch.no_grad()
def compute_latent_amplitude_metrics_per_layer(
    mean_latent_by_layer: Dict[int, torch.Tensor],
    by_category: Dict[str, List[Dict[str, Any]]],
    categories: List[str],
    word_to_idx: Dict[str, int],
    layers: List[int],
    eps: float = 1e-8,
) -> Dict[str, Dict[int, Dict[Tuple[str, str], float]]]:
    """
    Amplitude-preserving counterparts to Jaccard: for each (category, item)
    pair and layer, compute pairwise metrics on the mean-pooled continuous
    SAE latent activation vectors.

    "kl" L1-normalizes each (non-negative) latent vector into a distribution
    (p_i = x_i / sum(x)) before computing symmetric KL. "klsm" instead
    softmax-normalizes (p = softmax(x)) — an alternative way to turn the
    latent vector into a distribution, kept alongside "kl" for comparison
    rather than replacing it.

    Returns {"l1": ..., "kl": ..., "klsm": ..., "saecos": ..., "dot": ...},
    each shaped like compute_jaccard_per_layer's return value:
      metrics[name][layer][(category, item)] -> float
    """
    metrics: Dict[str, Dict[int, Dict[Tuple[str, str], float]]] = {
        name: {l: {} for l in layers} for name in ("l1", "kl", "klsm", "saecos", "dot")
    }

    # Flatten all (category, item) pairs once — same pair list for every layer —
    # so each layer is computed as a handful of batched [P, d_sae] tensor ops
    # instead of a Python loop with ~20 individual tensor ops per pair.
    pair_cats: List[str] = []
    pair_items: List[str] = []
    for cat in categories:
        for row in by_category[cat]:
            pair_cats.append(cat)
            pair_items.append(row["item"])
    pair_keys = list(zip(pair_cats, pair_items))
    cat_idx = torch.tensor([word_to_idx[c] for c in pair_cats], dtype=torch.long)
    item_idx = torch.tensor([word_to_idx[i] for i in pair_items], dtype=torch.long)

    for layer in layers:
        vecs = mean_latent_by_layer[layer].float().clamp_min(0.0)  # [N, d_sae]
        cat_vecs = vecs[cat_idx]    # [P, d_sae]
        item_vecs = vecs[item_idx]  # [P, d_sae]

        l1 = (item_vecs - cat_vecs).abs().sum(dim=1)  # [P]

        p = item_vecs + eps
        q = cat_vecs + eps
        p = p / p.sum(dim=1, keepdim=True)
        q = q / q.sum(dim=1, keepdim=True)
        kl_pq = (p * (p / q).log()).sum(dim=1)
        kl_qp = (q * (q / p).log()).sum(dim=1)
        kl = 0.5 * (kl_pq + kl_qp)  # [P]

        # Softmax-normalized alternative distribution (torch.softmax subtracts
        # the row max internally, so this is safe from overflow). Softmax
        # outputs are strictly positive in exact arithmetic, but the smallest
        # components can still underflow to exactly 0.0 in floating point for
        # widely-spread inputs, which would make the subsequent .log() -inf —
        # clamp to a tiny floor (negligible vs. any real probability mass) to
        # guard against that without needing eps-style renormalization.
        p_sm = torch.softmax(item_vecs, dim=1).clamp_min(1e-30)
        q_sm = torch.softmax(cat_vecs, dim=1).clamp_min(1e-30)
        klsm_pq = (p_sm * (p_sm / q_sm).log()).sum(dim=1)
        klsm_qp = (q_sm * (q_sm / p_sm).log()).sum(dim=1)
        klsm = 0.5 * (klsm_pq + klsm_qp)  # [P]

        dot = (item_vecs * cat_vecs).sum(dim=1)  # [P]
        denom = (item_vecs.norm(dim=1) * cat_vecs.norm(dim=1)).clamp_min(eps)
        saecos = dot / denom  # [P]

        metrics["l1"][layer] = dict(zip(pair_keys, l1.tolist()))
        metrics["kl"][layer] = dict(zip(pair_keys, kl.tolist()))
        metrics["klsm"][layer] = dict(zip(pair_keys, klsm.tolist()))
        metrics["saecos"][layer] = dict(zip(pair_keys, saecos.tolist()))
        metrics["dot"][layer] = dict(zip(pair_keys, dot.tolist()))

    return metrics


def compute_jaccard_per_layer(
    union_by_layer: Dict[int, torch.Tensor],
    by_category: Dict[str, List[Dict[str, Any]]],
    categories: List[str],
    word_to_idx: Dict[str, int],
    layers: List[int],
) -> Dict[int, Dict[Tuple[str, str], float]]:
    """
    Jaccard(union_item, union_category) for each (category, item) pair per layer.
    jaccard(A, B) = |A ∩ B| / |A ∪ B|
    """
    jac_by_layer: Dict[int, Dict[Tuple[str, str], float]] = {l: {} for l in layers}

    for layer in layers:
        vecs = union_by_layer[layer].float()  # [N, d_sae]

        for cat in categories:
            cat_vec = vecs[word_to_idx[cat]]  # [d_sae]

            for row in by_category[cat]:
                item = row["item"]
                item_vec = vecs[word_to_idx[item]]

                inter = (cat_vec * item_vec).sum()
                union_count = ((cat_vec + item_vec) > 0).float().sum().clamp_min(1.0)
                jac_by_layer[layer][(cat, item)] = float((inter / union_count).item())

    return jac_by_layer


def compute_contextual_cosim_per_layer(
    avg_residual_by_layer: Dict[int, torch.Tensor],
    by_category: Dict[str, List[Dict[str, Any]]],
    categories: List[str],
    word_to_idx: Dict[str, int],
    layers: List[int],
    eps: float = 1e-8,
) -> Dict[int, Dict[Tuple[str, str], float]]:
    """
    Cosine similarity between prompted residual stream vectors of items and
    categories at each layer, averaged over target word tokens.
    """
    ctx_cosim_by_layer: Dict[int, Dict[Tuple[str, str], float]] = {l: {} for l in layers}

    for layer in layers:
        vecs = avg_residual_by_layer[layer]  # [N, d_model]
        norms = vecs.norm(dim=-1, keepdim=True).clamp_min(eps)
        norm_vecs = (vecs / norms).float()

        for cat in categories:
            cat_nvec = norm_vecs[word_to_idx[cat]]  # [d_model]

            for row in by_category[cat]:
                item = row["item"]
                item_nvec = norm_vecs[word_to_idx[item]]
                cos = float((cat_nvec * item_nvec).sum().item())
                ctx_cosim_by_layer[layer][(cat, item)] = cos

    return ctx_cosim_by_layer
