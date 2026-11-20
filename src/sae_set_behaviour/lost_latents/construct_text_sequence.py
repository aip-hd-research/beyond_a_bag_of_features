from __future__ import annotations
from typing import Dict, List
import torch
import csv
from dataclasses import dataclass
from pathlib import Path

PAPER_TEMPLATE = "This is a {sequence}. "


@dataclass(frozen=True)
class TextPair:
    """Represents a compositional pair: X and X+Y."""
    base_text: str       # X (may be templated)
    composed_text: str   # X + Y (may be templated)
    noun: str = ""       # bare noun of base_text (for last-token extraction)
    composed_noun: str = ""  # bare noun of composed_text; if empty, falls back to noun

    @property
    def noun_for_base(self) -> str:
        return self.noun

    @property
    def noun_for_composed(self) -> str:
        return self.composed_noun if self.composed_noun else self.noun


def apply_prefix_to_pairs(pairs: List[TextPair], prefix: str) -> List[TextPair]:
    if not prefix:
        return pairs
    return [
        TextPair(
            base_text=f"{prefix}{p.base_text}",
            composed_text=f"{prefix}{p.composed_text}",
            noun=p.noun,
            composed_noun=p.composed_noun,
        )
        for p in pairs
    ]


def apply_template_to_pairs(pairs: List[TextPair], template: str) -> List[TextPair]:
    """Wrap the sequence of each pair in a full template, e.g. 'This is a {sequence}.  '."""
    if not template or "{sequence}" not in template:
        return pairs
    return [
        TextPair(
            base_text=template.format(sequence=p.base_text),
            composed_text=template.format(sequence=p.composed_text),
            noun=p.noun,
            composed_noun=p.composed_noun,
        )
        for p in pairs
    ]


def last_non_padding_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute the last non-padding token index for each sequence in a batch."""
    return attention_mask.sum(dim=1) - 1


def select_last_token_embedding_from_sequence(
    sequence_activations: torch.Tensor,
    last_token_index: torch.Tensor,
) -> torch.Tensor:
    batch_size = sequence_activations.size(0)
    batch_indices = torch.arange(batch_size, device=sequence_activations.device)
    return sequence_activations[batch_indices, last_token_index]


def read_noun_attribute_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def build_chain_texts(noun: str, attrs: List[str]) -> List[str]:
    """
    Returns [t0..tK] where:
      t0 = noun
      t1 = attr1 noun
      t2 = attr1 attr2 noun
      ...
    """
    noun = noun.strip()
    chain = [noun]
    prefix: List[str] = []
    for a in attrs:
        a = (a or "").strip()
        if not a:
            continue
        prefix.append(a)
        chain.append(" ".join(prefix + [noun]))
    return chain


def build_pairs_setup_A(rows: List[Dict[str, str]], k: int) -> List[TextPair]:
    """Fixed base: t0 compared to tk for each noun row."""
    pairs: List[TextPair] = []
    for row in rows:
        noun = row["noun"]
        attrs = [row.get(f"attr{i}", "") for i in range(1, 6)]
        chain = build_chain_texts(noun, attrs)
        if len(chain) > k:
            pairs.append(TextPair(base_text=chain[0], composed_text=chain[k], noun=noun))
    return pairs


def build_pairs_setup_B(rows: List[Dict[str, str]], k: int) -> List[TextPair]:
    """Stepwise: t(k-1) -> tk for each noun row."""
    pairs: List[TextPair] = []
    for row in rows:
        noun = row["noun"]
        attrs = [row.get(f"attr{i}", "") for i in range(1, 6)]
        chain = build_chain_texts(noun, attrs)
        if len(chain) > k:
            pairs.append(TextPair(base_text=chain[k - 1], composed_text=chain[k], noun=noun))
    return pairs


def build_pairs(
    rows: List[Dict[str, str]],
    setup_name: str,
    k: int,
    text_prefix: str = "",
    text_template: str = "",
) -> List[TextPair]:
    if setup_name == "A_fixed_base":
        pairs = build_pairs_setup_A(rows, k)
    elif setup_name == "B_stepwise":
        pairs = build_pairs_setup_B(rows, k)
    else:
        raise ValueError(f"Unknown setup_name: {setup_name}")

    if text_template and "{sequence}" in text_template:
        pairs = apply_template_to_pairs(pairs, text_template)
    elif text_prefix:
        pairs = apply_prefix_to_pairs(pairs, text_prefix)

    return pairs
