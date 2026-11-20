from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from transformer_lens import HookedTransformer
from sae_set_behaviour.lost_latents.construct_text_sequence import last_non_padding_token_index, apply_prefix_to_pairs
from sae_set_behaviour.lost_latents.metrics import lost_features_rate


def find_noun_token_spans(
    tokenizer, texts: List[str], nouns: List[str], max_length: int = None,
) -> List[Tuple[int, int]]:
    """
    Return (start, end) exclusive token span for the last occurrence of each noun in its text.
    Uses rfind so the terminal noun is found even when it also appears in attributes.
    Safe under right-padding: individual token positions are preserved.

    max_length must match whatever truncation the caller will actually apply when tokenizing
    for the model (tokenize_texts's max_length) -- otherwise a span computed against the full,
    untruncated text can point past the truncated sequence the model actually sees, causing a
    silent shape-mismatch crash downstream. Passing None (the default, unless a caller opts in)
    preserves the old untruncated behaviour for any caller that doesn't pass max_length.
    """
    spans = []
    for text, noun in zip(texts, nouns):
        char_start = text.rfind(noun)
        if char_start == -1:
            raise ValueError(f"{noun!r} not found in {text!r}")
        char_end = char_start + len(noun)
        if max_length is not None:
            enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length)
        else:
            enc = tokenizer(text, return_offsets_mapping=True)
        noun_toks = [
            i for i, (ts, te) in enumerate(enc["offset_mapping"])
            if ts < char_end and te > char_start
        ]
        if not noun_toks:
            raise ValueError(f"No tokens span {noun!r} in {text!r}")
        spans.append((noun_toks[0], noun_toks[-1] + 1))
    return spans


def find_noun_last_token_indices(tokenizer, texts: List[str], nouns: List[str], max_length: int = None) -> List[int]:
    """Return the index of the LAST token spanning each noun (convenience wrapper)."""
    return [e - 1 for _, e in find_noun_token_spans(tokenizer, texts, nouns, max_length=max_length)]


def extract_noun_word_acts(
    acts: torch.Tensor,
    noun_spans: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract noun-token activations from full sequence activations.

    acts: [N, T, d_model] (any device)
    Returns:
      word_acts: [N, W, d_model] CPU float32, padded to max noun token count
      word_mask: [N, W] int8, 1 for real noun tokens
    """
    N, T, d = acts.shape
    max_w = max(e - s for s, e in noun_spans)
    word_acts = torch.zeros(N, max_w, d, dtype=torch.float32)
    word_mask = torch.zeros(N, max_w, dtype=torch.int8)
    for i, (s, e) in enumerate(noun_spans):
        n = min(e - s, max_w)
        word_acts[i, :n] = acts[i, s:s + n].float().cpu()
        word_mask[i, :n] = 1
    return word_acts, word_mask


@torch.no_grad()
def collect_noun_word_activations(
    model: HookedTransformer,
    texts: List[str],
    nouns: List[str],
    hook_name: str,
    device: str,
    max_length: int,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect residual-stream activations at ALL tokens spanning each noun.
    Returns:
      word_acts: [N, W, d_model] CPU float32, padded to max noun token count
      word_mask: [N, W] int8, 1 for real noun tokens
    """
    spans = find_noun_token_spans(model.tokenizer, texts, nouns, max_length=max_length)
    max_w = max(e - s for s, e in spans)

    word_batches: List[torch.Tensor] = []
    mask_batches: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_spans = spans[start:start + batch_size]
        B = len(batch_texts)

        input_ids, attention_mask = tokenize_texts(
            model=model, texts=batch_texts, device=device, max_length=max_length
        )
        _, cache = model.run_with_cache(input_ids, names_filter=lambda n: n == hook_name)
        acts = cache[hook_name].float().cpu()  # [B, T, d_model]
        d = acts.shape[-1]

        batch_word = torch.zeros(B, max_w, d)
        batch_mask = torch.zeros(B, max_w, dtype=torch.int8)
        for b_idx, (s, e) in enumerate(batch_spans):
            n = min(e - s, max_w)
            batch_word[b_idx, :n] = acts[b_idx, s:s + n]
            batch_mask[b_idx, :n] = 1

        word_batches.append(batch_word)
        mask_batches.append(batch_mask)
        del input_ids, attention_mask, cache, acts

    return torch.cat(word_batches, dim=0), torch.cat(mask_batches, dim=0)


@torch.no_grad()
def collect_noun_last_token_activations(
    model: HookedTransformer,
    texts: List[str],
    nouns: List[str],
    hook_name: str,
    device: str,
    max_length: int,
    batch_size: int = 64,
) -> torch.Tensor:
    """Collect residual-stream activations at the last token of noun in each text."""
    outs = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_nouns = nouns[start:start + batch_size]
        input_ids, attention_mask = tokenize_texts(
            model=model, texts=batch_texts, device=device, max_length=max_length
        )
        _, cache = model.run_with_cache(input_ids, names_filter=lambda n: n == hook_name)
        acts = cache[hook_name]
        noun_indices = find_noun_last_token_indices(model.tokenizer, batch_texts, batch_nouns, max_length=max_length)
        bidx = torch.arange(acts.shape[0], device=acts.device)
        tidx = torch.tensor(noun_indices, dtype=torch.long, device=acts.device)
        selected = acts[bidx, tidx, :].detach().to(device)
        outs.append(selected)
        del input_ids, attention_mask, cache, acts, selected
    return torch.cat(outs, dim=0)


@torch.no_grad()
def collect_noun_word_activations_multi_cpu(
    *,
    model: HookedTransformer,
    texts: List[str],
    nouns: List[str],
    hook_names_by_layer: Dict[int, str],
    device: str,
    max_length: int,
    batch_size: int,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """
    Multi-layer version of collect_noun_word_activations.
    Returns:
      word_acts_by_layer[layer]: [N, W, d_model] CPU float32
      word_mask: [N, W] int8
    """
    spans = find_noun_token_spans(model.tokenizer, texts, nouns, max_length=max_length)
    max_w = max(e - s for s, e in spans)
    hook_name_set = set(hook_names_by_layer.values())

    word_outs: Dict[int, List[torch.Tensor]] = {layer: [] for layer in hook_names_by_layer}
    mask_batches: List[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_spans = spans[start:start + batch_size]
        B = len(batch_texts)

        input_ids, attention_mask = tokenize_texts(
            model=model, texts=batch_texts, device=device, max_length=max_length
        )
        _, cache = model.run_with_cache(input_ids, names_filter=lambda n: n in hook_name_set)

        batch_mask = torch.zeros(B, max_w, dtype=torch.int8)
        for b_idx, (s, e) in enumerate(batch_spans):
            n = min(e - s, max_w)
            batch_mask[b_idx, :n] = 1
        mask_batches.append(batch_mask)

        for layer, hook_name in hook_names_by_layer.items():
            acts = cache[hook_name].float().cpu()  # [B, T, d_model]
            d = acts.shape[-1]
            batch_word = torch.zeros(B, max_w, d)
            for b_idx, (s, e) in enumerate(batch_spans):
                n = min(e - s, max_w)
                batch_word[b_idx, :n] = acts[b_idx, s:s + n]
            word_outs[layer].append(batch_word)
            del acts

        del input_ids, attention_mask, cache

    word_acts_by_layer = {layer: torch.cat(chunks, dim=0) for layer, chunks in word_outs.items()}
    word_mask = torch.cat(mask_batches, dim=0)
    return word_acts_by_layer, word_mask


def tokenize_texts(model: HookedTransformer, texts: List[str], device: str, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenized = model.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    return input_ids, attention_mask

@torch.no_grad()
def collect_last_token_activations(model: HookedTransformer, texts: List[str], hook_name: str, device: str, max_length: int, batch_size: int = 64) -> torch.Tensor:
    outs = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        input_ids, attention_mask = tokenize_texts(model=model, texts=batch_texts, device=device, max_length=max_length)
        _, cache = model.run_with_cache(input_ids, names_filter=lambda n: n == hook_name)
        acts = cache[hook_name]
        last_idx = last_non_padding_token_index(attention_mask).to(acts.device)
        bidx = torch.arange(acts.shape[0], device=acts.device)
        selected = acts[bidx, last_idx, :].detach().to(device)
        outs.append(selected)
        del input_ids, attention_mask, cache, acts, last_idx, bidx, selected
    return torch.cat(outs, dim=0)

@torch.no_grad()
def collect_mean_pooled_activations(model: HookedTransformer, texts: List[str], hook_name: str, device: str, max_length: int, batch_size: int = 64) -> torch.Tensor:
    """
    Run the model on `input` and return mean-pooled activations at `hook_name`.
    Create an embedding like:
      v = (sum_{t where attention_mask==1} h_t) / (number of non-padding tokens)
    Returns: Tensor[batch, d_model] (on `device`)
    """
    outs = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        input_ids, attention_mask = tokenize_texts(model=model, texts=batch_texts, device=device, max_length=max_length)
        _, cache = model.run_with_cache(input_ids, names_filter=lambda n: n == hook_name)

        activations_at_hook = cache[hook_name]

        attention_mask_hook = attention_mask.to(device=activations_at_hook.device)
        mask = attention_mask_hook.to(dtype=activations_at_hook.dtype).unsqueeze(-1)
        summed = (activations_at_hook * mask).sum(dim=1)

        lengths = attention_mask_hook.sum(dim=1).clamp_min(1).to(dtype=activations_at_hook.dtype).unsqueeze(-1)
        mean_pooled = (summed / lengths).detach().to(device)

        outs.append(mean_pooled)
        del input_ids, attention_mask, cache, activations_at_hook, attention_mask_hook, mask, summed, lengths, mean_pooled
    return torch.cat(outs, dim=0)


@torch.no_grad()
def collect_sequence_activations(*, model: HookedTransformer, texts: List[str], hook_name: str, device: str, max_length: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns: acts: [B,S,d_in], attention_mask: [B,S] as bool"""
    input_ids, attention_mask = tokenize_texts(model=model, texts=texts, device=device, max_length=max_length)
    _, cache = model.run_with_cache(input_ids, names_filter=lambda n: n == hook_name)
    acts = cache[hook_name].detach().to(device)
    attention_mask = attention_mask.to(device=device, dtype=torch.bool)
    return acts, attention_mask

def get_activation_collector(pooling: str):
    """
    Returns the activation collection function based on the pooling mode.
    """
    if pooling == "last_token":
        return collect_last_token_activations
    if pooling == "mean_pool":
        return collect_mean_pooled_activations
    raise ValueError(f"Unknown pooling mode: {pooling}. Use 'last_token' or 'mean_pool'.")