from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import pandas as pd
from datasets import load_dataset

def _norm_text(s: str) -> str:
    return str(s).lower()

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import load_dataset


CATEGORY_ALIASES = {
    "birds": "bird",
}


def _canonicalize_category(x: str) -> str:
    x = _norm_text(x)
    return CATEGORY_ALIASES.get(x, x)


def preprocess_human_concepts_rows(
    raw_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    1) normalize category/item to lowercase + strip
    2) canonicalize category aliases (e.g. birds -> bird)
    3) if an item occurs multiple times in the same canonicalized category -> average typicality, keep one row

    Returns:
      - cleaned_rows: list of dicts with keys: category, item, typicality
      - merge_report: dataframe listing (category,item) that were merged and their stats
    """
    buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    source_categories: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for r in raw_rows:
        cat_raw = str(r["category"])
        cat = _canonicalize_category(cat_raw)
        item = _norm_text(r["item"])
        typ = float(r["typicality"])

        buckets[(cat, item)].append(typ)
        source_categories[(cat, item)].append(_norm_text(cat_raw))

    cleaned_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    for (cat, item), typs in buckets.items():
        mean_typ = float(sum(typs) / len(typs))
        cleaned_rows.append({
            "category": cat,
            "item": item,
            "typicality": mean_typ,
        })

        src_cats = sorted(set(source_categories[(cat, item)]))
        if len(typs) > 1 or len(src_cats) > 1:
            report_rows.append({
                "category": cat,
                "item": item,
                "n_merged": len(typs),
                "source_categories": src_cats,
                "typicality_values": typs,
                "mean_typicality": mean_typ,
                "min_typicality": float(min(typs)),
                "max_typicality": float(max(typs)),
            })

    merge_report = pd.DataFrame(report_rows)
    if not merge_report.empty:
        merge_report = merge_report.sort_values(
            ["n_merged", "category", "item"],
            ascending=[False, True, True],
        )

    return cleaned_rows, merge_report


def load_and_group_dataset(cfg: TypicalityStaticCosimConfig) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str], List[str]]:
    ds = load_dataset(cfg.dataset_name)
    data = ds[cfg.split]

    raw_rows: List[Dict[str, Any]] = []
    for row in data:
        raw_rows.append({
            "category": str(row["category"]),
            "item": str(row["item"]),
            "typicality": float(row["typicality"]),
            "subdataset": str(row.get("subdataset", "")),
        })

    cleaned_rows, merge_report = preprocess_human_concepts_rows(raw_rows)

    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for r in cleaned_rows:
        c = r["category"]
        it = r["item"]
        typ = float(r["typicality"])

        by_category.setdefault(c, []).append({
            "item": it,
            "category": c,
            "typicality": typ,
            "subdataset": "",
        })

    # by_category = {c: rows for c, rows in by_category.items() if len(rows) >= cfg.min_items_per_category}
    # if not by_category:
    #     raise ValueError("No categories left after filtering. Lower min_items_per_category?")

    categories = sorted(by_category.keys())
    items = sorted({r["item"] for rows in by_category.values() for r in rows})
    return by_category, categories, items