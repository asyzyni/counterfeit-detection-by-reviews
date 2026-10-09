from __future__ import annotations

import string

import pandas as pd


# ============================================================
# EKSPOR TABEL CONTOH (format Tabel 3.1 skripsi)
# ============================================================

def format_timestamp(
    timestamp: pd.Timestamp,
    has_time: bool,
) -> str:
    """'2026-03-07' atau '2025-09-08 11:53'; NaT -> ''."""

    if pd.isna(timestamp):
        return ""

    return timestamp.strftime("%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d")


def _store_label(index: int) -> str:
    # 0 -> A, 25 -> Z, 26 -> AA, ...
    letters = string.ascii_uppercase
    label = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        label = letters[remainder] + label
    return f"Toko {label}"


def export_tabel_contoh(
    df: pd.DataFrame,
    n: int = 3,
    anonimkan: bool = False,
    random_state: int = 42,
    hanya_include: bool = True,
) -> pd.DataFrame:
    """
    Kolom: Nama Toko, Nama Produk, Review, Timestamp.
    Baris dipilih acak (deterministik lewat random_state). Anonimisasi
    (Toko A, Produk 1, ...) HANYA pada keluaran ini; data kerja tetap asli.
    """

    pool = df[df["include_for_inference"]] if hanya_include else df
    sample = pool.sample(n=min(n, len(pool)), random_state=random_state)
    sample = sample.sort_values(["nama_toko", "nama_produk", "seq_in_product"], kind="mergesort")

    table = pd.DataFrame(
        {
            "Nama Toko": sample["nama_toko"].to_numpy(),
            "Nama Produk": sample["nama_produk"].to_numpy(),
            "Review": sample["review"].to_numpy(),
            "Timestamp": [
                format_timestamp(ts, has_time)
                for ts, has_time in zip(sample["timestamp"], sample["timestamp_has_time"])
            ],
        }
    )

    if anonimkan:
        stores = {name: _store_label(i) for i, name in enumerate(pd.unique(table["Nama Toko"]))}
        products = {
            key: f"Produk {i + 1}"
            for i, key in enumerate(dict.fromkeys(zip(table["Nama Toko"], table["Nama Produk"])))
        }
        table["Nama Produk"] = [
            products[(store, product)]
            for store, product in zip(table["Nama Toko"], table["Nama Produk"])
        ]
        table["Nama Toko"] = table["Nama Toko"].map(stores)

    return table.reset_index(drop=True)
