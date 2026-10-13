"""Daftar id produk Train/Test.

Daftar sungguhan seharusnya datang dari luar modul HMM (distratifikasi menurut label dari tahap
Labeling). Fungsi ``temporary_product_split`` hanya penampung sementara dan tidak untuk dilaporkan.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def load_split_ids(path: str | Path, product_col: str = "product_id", split_col: str = "split") -> tuple[list[Any], list[Any]]:
    """Baca CSV/parquet berkolom ``product_id`` dan ``split`` (nilai train/test).

    Hanya dua kolom itu yang dibaca; kolom label (bila ada di file) diabaikan.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Daftar split tidak ditemukan: {path}")
    if path.suffix.lower() == ".parquet":
        table = pd.read_parquet(path, columns=[product_col, split_col])
    else:
        table = pd.read_csv(path, usecols=[product_col, split_col], encoding="utf-8-sig")
    table[split_col] = table[split_col].astype(str).str.strip().str.lower()
    unknown = sorted(set(table[split_col]) - {"train", "test"})
    if unknown:
        raise ValueError(f"Nilai kolom {split_col!r} harus 'train' atau 'test'; ada {unknown}")
    if table[product_col].duplicated().any():
        raise ValueError("Ada product_id ganda di daftar split.")
    train = table.loc[table[split_col] == "train", product_col].tolist()
    test = table.loc[table[split_col] == "test", product_col].tolist()
    return train, test


def temporary_product_split(
    product_ids: Sequence[Any], test_size: float = 0.20, seed: int = 42
) -> tuple[list[Any], list[Any]]:
    """TEMPORARY, bukan untuk dilaporkan: split acak tanpa stratifikasi. Ganti dengan daftar sungguhan."""
    warnings.warn(
        "TEMPORARY SPLIT: split acak tanpa stratifikasi label, bukan untuk dilaporkan. "
        "Berikan daftar id Train/Test sungguhan (--split-ids).",
        stacklevel=2,
    )
    ids = sorted(set(product_ids), key=str)
    if len(ids) < 3:
        raise ValueError("Butuh minimal 3 produk untuk split.")
    rng = np.random.default_rng(seed)
    perm = [ids[i] for i in rng.permutation(len(ids))]
    n_test = max(1, int(round(len(ids) * test_size)))
    return perm[n_test:], perm[:n_test]
