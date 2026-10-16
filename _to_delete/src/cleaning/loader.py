"""Pintu masuk pembacaan tabel ulasan hasil pipeline cleaning baru."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .config import CleaningConfig


_INFERENCE_COLUMNS = ("review_id", "product_id", "text_light", "timestamp")


def load_clean_reviews(
    config: CleaningConfig | None = None,
    path: str | Path | None = None,
    only_included: bool = True,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Baca satu parquet, filter kelayakan inferensi, lalu urutkan per produk.

    ``path`` menunjuk file dan mengalahkan ``config.reviews_path``; tanpa
    keduanya dipakai lokasi bawaan CleaningConfig. Pembacaan tidak memerlukan
    data mentah, validasi konfigurasi, atau pembuatan direktori.
    ``only_included=True`` memilih include_for_inference == True.
    ``columns=None`` mengembalikan semua kolom; pilihan kolom selalu ditambah
    review_id, product_id, text_light, dan timestamp agar siap untuk inferensi.
    Kedua mode diurutkan menurut product_id lalu seq_in_product dengan indeks
    baru. Tipe kolom, nilai timestamp NaT, dan nomor urut asli dipertahankan.
    """

    reviews_path = Path(path) if path is not None else (config or CleaningConfig()).reviews_path
    if not reviews_path.is_file():
        raise FileNotFoundError(
            f"Hasil cleaning tidak ditemukan: {reviews_path}. "
            "Jalankan python -m src.cleaning.run untuk membuat reviews_clean.parquet."
        )

    reviews = pd.read_parquet(reviews_path)
    required = [*_INFERENCE_COLUMNS, "seq_in_product"]
    if only_included:
        required.append("include_for_inference")
    missing = [column for column in required if column not in reviews.columns]
    if missing:
        raise ValueError(f"Kolom wajib hasil cleaning tidak tersedia di {reviews_path}: {missing}")

    selected = None
    if columns is not None:
        selected = list(dict.fromkeys([*columns, *_INFERENCE_COLUMNS]))
        missing = [column for column in selected if column not in reviews.columns]
        if missing:
            raise ValueError(f"Kolom yang diminta tidak tersedia di {reviews_path}: {missing}")

    if only_included:
        reviews = reviews.loc[reviews["include_for_inference"].eq(True)]
    reviews = reviews.sort_values(["product_id", "seq_in_product"], kind="mergesort")
    if selected is not None:
        reviews = reviews.loc[:, selected]
    return reviews.reset_index(drop=True)
