from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from sae_set_behaviour.lost_latents.construct_text_sequence import (
    PAPER_TEMPLATE,
    TextPair,
    read_noun_attribute_rows,
)

# Anchor default CSV paths on this file's location (not the launching shell's cwd) --
# Hydra's to_absolute_path resolves relative paths against the original working
# directory the process was invoked from, which silently breaks when a script is
# launched from somewhere other than the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROMPT_CONSTRUCTIONS_DIR = _REPO_ROOT / "data"

# German indefinite-article prefixes keyed by the CSV's "gender" column (m/f/n).
# The attribute words in data_german.csv are already declined for nominative
# indefinite-article agreement (e.g. "gelbe Schaufel", "schwerer Hammer"), so
# only the article itself needs to vary by gender. Nominative "ein" is shared
# by masculine and neuter (e.g. "ein Hammer", "ein Seil"); only feminine differs.
GERMAN_TEMPLATE_BY_GENDER: Dict[str, str] = {
    "m": "Dies ist ein {sequence}. ",
    "n": "Dies ist ein {sequence}. ",
    "f": "Dies ist eine {sequence}. ",
}


@dataclass(frozen=True)
class ChainStep:
    """One step of a per-row text chain.

    text: bare (untemplated) text for this step, e.g. "big yellow shovel".
    content_words: words whose token spans matter at this step, ordered so
        the last entry is always the "primary"/head word of the step (the
        noun for noun-attribute chains; the category or item word for the
        human-typicality case).
    """

    text: str
    content_words: List[str]


@dataclass(frozen=True)
class DataCase:
    """Encapsulates everything that differs between data sources (English
    noun-attribute chains, German noun-attribute chains, human-typicality
    category/instance pairs) so experiment_1 scripts can stay data-agnostic.
    """

    name: str
    k_max: int
    default_csv: Optional[str]
    load_rows: Callable[[Optional[Path]], List[Dict[str, str]]]
    build_chain: Callable[[Dict[str, str], int], List[ChainStep]]
    render_step_text: Callable[[Dict[str, str], str, str, str], str]


def _noun_attr_build_chain(row: Dict[str, str], k_max: int) -> List[ChainStep]:
    """t0 = noun, t1 = attr1 noun, t2 = attr1 attr2 noun, ... (skips empty attrs)."""
    noun = row["noun"].strip()
    steps = [ChainStep(text=noun, content_words=[noun])]
    prefix_words: List[str] = []
    for i in range(1, k_max + 1):
        a = (row.get(f"attr{i}", "") or "").strip()
        if not a:
            continue
        prefix_words.append(a)
        steps.append(ChainStep(text=" ".join(prefix_words + [noun]), content_words=prefix_words + [noun]))
    return steps


def _human_typicality_build_chain(row: Dict[str, str], k_max: int) -> List[ChainStep]:
    """t0 = category word, t1 = instance word."""
    category = row["category"].strip()
    item = row["item"].strip()
    return [
        ChainStep(text=category, content_words=[category]),
        ChainStep(text=item, content_words=[item]),
    ][: k_max + 1]


def _fixed_render(row: Dict[str, str], text: str, base_template: str, base_prefix: str) -> str:
    if base_template and "{sequence}" in base_template:
        return base_template.format(sequence=text)
    if base_prefix:
        return f"{base_prefix}{text}"
    return text


def _german_gendered_render(row: Dict[str, str], text: str, base_template: str, base_prefix: str) -> str:
    # Templating is opt-in the same way as the fixed-template cases: if the
    # caller passed neither a template nor a prefix, leave the text bare.
    if not base_template and not base_prefix:
        return text
    gender = str(row.get("gender", "")).strip().lower()
    template = GERMAN_TEMPLATE_BY_GENDER.get(gender)
    if template is None:
        raise ValueError(f"Row has missing/invalid gender {row.get('gender')!r} (expected 'm' or 'f'): {row!r}")
    return template.format(sequence=text)


def _load_csv_rows(csv_path: Optional[Path]) -> List[Dict[str, str]]:
    if csv_path is None:
        raise ValueError("csv_path is required for this data case")
    return read_noun_attribute_rows(csv_path)


def _load_human_typicality_rows(_csv_path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Loads every (category, item) pair from the CShani/human-concepts dataset live from HF."""
    from datasets import load_dataset

    from sae_set_behaviour.typicality.prepare_dataset import preprocess_human_concepts_rows

    ds = load_dataset("CShani/human-concepts")
    raw_rows = [
        {"category": str(r["category"]), "item": str(r["item"]), "typicality": float(r["typicality"])}
        for r in ds["train"]
    ]
    cleaned_rows, _merge_report = preprocess_human_concepts_rows(raw_rows)
    return cleaned_rows


_CASES: Dict[str, DataCase] = {
    "english": DataCase(
        name="english",
        k_max=5,
        default_csv=str(_PROMPT_CONSTRUCTIONS_DIR / "data.csv"),
        load_rows=_load_csv_rows,
        build_chain=_noun_attr_build_chain,
        render_step_text=_fixed_render,
    ),
    "german": DataCase(
        name="german",
        k_max=5,
        default_csv=str(_PROMPT_CONSTRUCTIONS_DIR / "data_german.csv"),
        load_rows=_load_csv_rows,
        build_chain=_noun_attr_build_chain,
        render_step_text=_german_gendered_render,
    ),
    "human_typicality": DataCase(
        name="human_typicality",
        k_max=1,
        default_csv=None,
        load_rows=_load_human_typicality_rows,
        build_chain=_human_typicality_build_chain,
        render_step_text=_fixed_render,
    ),
}


def get_data_case(name: str) -> DataCase:
    key = str(name).strip().lower()
    if key not in _CASES:
        raise ValueError(f"Unknown data_case {name!r}; expected one of {sorted(_CASES)}")
    return _CASES[key]


def build_chain_texts_and_words_for_rows(
    rows: List[Dict[str, str]],
    case: DataCase,
    base_template: str = "",
    base_prefix: str = "",
) -> Tuple[List[List[str]], List[List[str]], List[List[List[str]]]]:
    """Returns (chains_by_row, primary_words_by_row, content_words_by_row).

    chains_by_row[r]        -> rendered chain texts for row r, length case.k_max+1
    primary_words_by_row[r] -> the head/primary word at each step (same length)
    content_words_by_row[r] -> the full content-word list at each step (same length)
    """
    chains_by_row: List[List[str]] = []
    primary_words_by_row: List[List[str]] = []
    content_words_by_row: List[List[List[str]]] = []
    for row in rows:
        steps = case.build_chain(row, case.k_max)
        chains_by_row.append([case.render_step_text(row, s.text, base_template, base_prefix) for s in steps])
        primary_words_by_row.append([s.content_words[-1] for s in steps])
        content_words_by_row.append([s.content_words for s in steps])
    return chains_by_row, primary_words_by_row, content_words_by_row


def build_pairs_for_case(
    rows: List[Dict[str, str]],
    case: DataCase,
    setup_name: str,
    k: int,
    base_template: str = "",
    base_prefix: str = "",
) -> List[TextPair]:
    """Generalizes build_pairs_setup_A/B over an arbitrary DataCase."""
    pairs: List[TextPair] = []
    for row in rows:
        steps = case.build_chain(row, case.k_max)
        if setup_name == "A_fixed_base":
            base_idx, comp_idx = 0, k
        elif setup_name == "B_stepwise":
            base_idx, comp_idx = k - 1, k
        else:
            raise ValueError(f"Unknown setup_name: {setup_name}")
        if base_idx < 0 or comp_idx >= len(steps):
            continue
        base_step, comp_step = steps[base_idx], steps[comp_idx]
        pairs.append(
            TextPair(
                base_text=case.render_step_text(row, base_step.text, base_template, base_prefix),
                composed_text=case.render_step_text(row, comp_step.text, base_template, base_prefix),
                noun=base_step.content_words[-1],
                composed_noun=comp_step.content_words[-1],
            )
        )
    return pairs


def build_random_mismatch_pairs_for_case(
    rows: List[Dict[str, str]],
    case: DataCase,
    k: int,
    setup_name: str,
    seed: int,
    base_template: str = "",
    base_prefix: str = "",
) -> List[TextPair]:
    """Generalizes build_random_mismatch_pairs over an arbitrary DataCase.

    Pairs each row's base_text with another (shuffled) row's composed_text.
    Both sides are rendered per their own originating row before shuffling,
    so e.g. German gender agreement stays correct on the mismatched side too.
    """
    rng = random.Random(seed)

    left_pairs = build_pairs_for_case(rows, case, setup_name, k, base_template, base_prefix)
    right_pairs = build_pairs_for_case(rows, case, setup_name, k, base_template, base_prefix)
    if not left_pairs or not right_pairs:
        return []

    right_data = [(p.composed_text, p.composed_noun) for p in right_pairs]
    rng.shuffle(right_data)

    return [
        TextPair(base_text=lp.base_text, composed_text=right_text, noun=lp.noun, composed_noun=right_noun)
        for lp, (right_text, right_noun) in zip(left_pairs, right_data)
    ]


def resolve_csv_path(case: DataCase, data_csv: Optional[str], to_absolute_path: Callable[[str], str]) -> Optional[Path]:
    """Resolves the CSV path to use: explicit data_csv override wins, else the case default."""
    if data_csv:
        return Path(to_absolute_path(str(data_csv)))
    if case.default_csv:
        return Path(to_absolute_path(case.default_csv))
    return None
