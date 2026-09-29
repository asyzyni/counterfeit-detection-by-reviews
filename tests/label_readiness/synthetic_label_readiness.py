"""Audit harness only: do not implement the label pipeline here.

It builds the required synthetic corpus, then attempts to import the two
documented Jalur-B modules.  When either module is unavailable the six
behavioural checks are reported as BLOCKED rather than simulated.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def make_corpus() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(160):
        rows.append({
            "product_id": "produk_counterfeit",
            "text_lex": "bukan_ori jahitan_lepas luntur kirim",
        })
        rows.append({
            "product_id": "produk_genuine",
            "text_lex": "ori awet sesuai_deskripsi kirim",
        })
    return rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    corpus = make_corpus()
    print(f"synthetic_rows={len(corpus)}")
    print("synthetic_products=2")
    print("boundary_probe=['kandidat', 'seed_counterfeit']")
    unavailable = []
    for module in ("src.features.lexicon", "src.features.tfidf"):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as error:
            unavailable.append(module)
            print(f"IMPORT_ERROR[{module}]={type(error).__name__}: {error}")
    if unavailable:
        for number, name in enumerate((
            "SO_and_CI",
            "review_score_ranking",
            "mean_and_max_product_ranking",
            "swapped_seed_ranking",
            "review_boundary",
            "determinism",
        ), start=1):
            print(f"test_{number}_{name}=BLOCKED: Jalur B module unavailable")
        return
    print("IMPORT_OK: harness needs binding to the real public API")


if __name__ == "__main__":
    main()
