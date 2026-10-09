from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import CleaningConfig


# ============================================================
# CONSTANTS
# ============================================================

MAPPING_COLUMNS = [
    "file",
    "nama_toko",
    "nama_produk",
    "product_key",
    "status",
    "sumber_nama",
    "catatan",
    "ukuran_byte",
]

CSV_ENCODINGS = (
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin1",
)


class MappingError(RuntimeError):
    """Pemetaan tidak sesuai harapan -> run harus berhenti."""


@dataclass
class FileMap:
    # Baris berstatus "ok" yang akan diproses.
    ok: pd.DataFrame
    # Baris yang dilewati beserta kolom `alasan`.
    excluded: pd.DataFrame
    # Jumlah baris mentah di file pemetaan.
    n_mapping_rows: int


# ============================================================
# PEMETAAN
# ============================================================

def load_file_map(
    config: CleaningConfig,
) -> FileMap:

    mapping = pd.read_csv(
        config.mapping_path,
        dtype=str,
        keep_default_na=False,
        encoding=config.mapping_encoding,
    )

    missing_columns = [
        col for col in MAPPING_COLUMNS
        if col not in mapping.columns
    ]

    if missing_columns:
        raise MappingError(
            f"Kolom pemetaan hilang: {missing_columns}"
        )

    mapping = mapping[MAPPING_COLUMNS].copy()

    for col in MAPPING_COLUMNS:
        mapping[col] = mapping[col].astype(str)

    n_mapping_rows = len(mapping)

    # --------------------------------------------------------
    # Baris yang merujuk file pemetaan itu sendiri
    # (ditemukan pada pemetaan 2026-10-09: status "ok").
    # --------------------------------------------------------

    is_self = mapping["file"] == config.mapping_filename
    is_ok = mapping["status"].str.strip() == "ok"

    excluded = mapping[is_self | ~is_ok].copy()
    excluded["alasan"] = [
        "baris pemetaan merujuk file pemetaan itu sendiri"
        if self_row
        else f"status pemetaan: {status}"
        for self_row, status in zip(
            is_self[is_self | ~is_ok],
            excluded["status"],
        )
    ]

    ok = mapping[~is_self & is_ok].copy()

    # --------------------------------------------------------
    # Validasi jumlah
    # --------------------------------------------------------

    n_rows_without_self = n_mapping_rows - int(is_self.sum())

    if (
        config.expected_mapping_rows is not None
        and n_rows_without_self != config.expected_mapping_rows
    ):
        raise MappingError(
            f"Jumlah baris pemetaan (tanpa baris dirinya sendiri) = "
            f"{n_rows_without_self}, diharapkan "
            f"{config.expected_mapping_rows}."
        )

    if (
        config.expected_ok_files is not None
        and len(ok) != config.expected_ok_files
    ):
        raise MappingError(
            f"Jumlah file status ok = {len(ok)}, diharapkan "
            f"{config.expected_ok_files}."
        )

    # --------------------------------------------------------
    # Validasi isi
    # --------------------------------------------------------

    duplicated_files = ok["file"][ok["file"].duplicated()].tolist()
    if duplicated_files:
        raise MappingError(
            f"File muncul lebih dari sekali di pemetaan: {duplicated_files}"
        )

    empty_keys = ok["file"][ok["product_key"].str.strip() == ""].tolist()
    if empty_keys:
        raise MappingError(
            f"product_key kosong untuk file berstatus ok: {empty_keys}"
        )

    missing_files = [
        name for name in ok["file"]
        if not (Path(config.raw_dir) / name).is_file()
    ]
    if missing_files:
        raise MappingError(
            f"File berstatus ok tidak ditemukan: {missing_files}"
        )

    ok = ok.sort_values("file", kind="mergesort").reset_index(drop=True)
    excluded = excluded.sort_values("file", kind="mergesort").reset_index(drop=True)

    return FileMap(
        ok=ok,
        excluded=excluded,
        n_mapping_rows=n_mapping_rows,
    )


# ============================================================
# HASH & PEMBACAAN CSV
# ============================================================

def file_sha256(
    path: Path,
) -> str:

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_raw_csv(
    path: Path,
) -> tuple[pd.DataFrame, str]:
    """Baca CSV mentah apa adanya: semua string, sel kosong = ""."""

    last_error: Exception | None = None

    for encoding in CSV_ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
            )
            return df, encoding

        except UnicodeDecodeError as error:
            last_error = error

    raise RuntimeError(
        f"Gagal membaca {path}: {last_error}"
    )


def detect_family(
    columns: list[str],
) -> str:
    """Keluarga skema berdasarkan header (hanya informasi, bukan peran)."""

    names = {str(col).strip() for col in columns}

    if "web_scraper_order" in names:
        return "A_webscraper"

    if any(name.startswith("item-content") for name in names) or "time" in names:
        return "B_lazada"

    return "C_acak"


def build_duplicate_file_groups(
    hashes: pd.DataFrame,
) -> pd.DataFrame:
    """
    hashes: kolom file, product_key, sha256, ukuran_byte.
    Output: hanya file yang hash-nya dimiliki >1 file, dengan grup "FDnn".
    """

    counts = hashes["sha256"].value_counts()
    dup_hashes = sorted(counts[counts > 1].index)

    group_ids = {
        sha: f"FD{index + 1:02d}"
        for index, sha in enumerate(
            sorted(
                dup_hashes,
                key=lambda sha: hashes.loc[hashes["sha256"] == sha, "file"].min(),
            )
        )
    }

    duplicates = hashes[hashes["sha256"].isin(group_ids)].copy()
    duplicates.insert(0, "file_dup_group", duplicates["sha256"].map(group_ids))

    duplicates["product_key_sama_dalam_grup"] = (
        duplicates.groupby("file_dup_group")["product_key"].transform("nunique") == 1
    )

    return duplicates.sort_values(
        ["file_dup_group", "file"],
        kind="mergesort",
    ).reset_index(drop=True)
