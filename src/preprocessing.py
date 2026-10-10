from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import re
import shutil

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import ExperimentConfig


# ============================================================
# CONSTANTS
# ============================================================

METADATA_COLUMNS = {
    "web_scraper_order",
    "web_scraper_start_url",
    "pagination",
    "product_id",
    "timestamp",
    "phone",
    "data",
}


REVIEW_PATTERNS = [
    "review_text",
    "review_content",
    "review",
    "content",
    "text",
    "comment",
    "ulasan",
    "body",
    "description",
]


TIMESTAMP_PATTERNS = [
    "timestamp",
    "datetime",
    "created_at",
    "created",
    "date",
    "tanggal",
    "time",
]


NULL_TEXT_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "<na>",
}


# ============================================================
# HELPERS
# ============================================================

def make_safe_filename(
    value: str,
) -> str:
    value = str(value).strip()

    value = re.sub(
        r"[^\w\-.]+",
        "_",
        value,
    )

    value = value.strip("-")

    if not value:
        value = "unknown_product"

    return value


# ============================================================
# CSV READER
# ============================================================

def read_csv_sample(
    filepath: Path,
    nrows: int = 500,
) -> tuple[pd.DataFrame, str]:

    encodings = [
        "utf-8",
        "latin1",
        "iso-8859-1",
        "cp1252",
        "utf-16",
    ]

    last_error: Exception | None = None

    for encoding in encodings:
        try:
            df = pd.read_csv(
                filepath,
                dtype=str,
                keep_default_na=False,
                nrows=nrows,
                encoding=encoding,
            )

            return df, encoding

        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"Tidak bisa membaca {filepath.name} "
        f"dengan encoding yang diuji coba. "
        f"Error terakhir: {last_error}"
    )


# ============================================================
# REVIEW COLUMN DETECTOR
# ============================================================

def detect_review_columns(
    df: pd.DataFrame,
) -> list[str]:

    columns = list(df.columns)

    normalized_columns = {
        str(col).lower().strip(): col
        for col in columns
    }

    # --------------------------------------------------------
    # Prioritas utama: kolom bernama "review"
    # --------------------------------------------------------

    if "review" in normalized_columns:
        return [
            normalized_columns["review"]
        ]

    review_columns: list[str] = []

    # --------------------------------------------------------
    # Deteksi berdasarkan nama kolom
    # --------------------------------------------------------

    for col in columns:

        col_lower = (
            str(col)
            .lower()
            .strip()
        )

        if col_lower in METADATA_COLUMNS:
            continue

        # Contoh:
        # data1
        # data2
        # data3
        if re.fullmatch(
            r"data\d+",
            col_lower,
        ):
            review_columns.append(col)
            continue

        if any(
            pattern in col_lower
            for pattern in REVIEW_PATTERNS
        ):
            review_columns.append(col)

    if review_columns:
        return review_columns

    # --------------------------------------------------------
    # Fallback heuristic:
    # cari kolom dengan teks relatif panjang
    # --------------------------------------------------------

    for col in columns:

        col_lower = (
            str(col)
            .lower()
            .strip()
        )

        if col_lower in METADATA_COLUMNS:
            continue

        sample = (
            df[col]
            .astype("string")
            .replace("", pd.NA)
            .dropna()
            .head(30)
        )

        if sample.empty:
            continue

        average_length = (
            sample
            .str
            .len()
            .mean()
        )

        if (
            pd.notna(average_length)
            and average_length > 20
        ):
            review_columns.append(col)

    return review_columns


# ============================================================
# TIMESTAMP COLUMN DETECTOR
# ============================================================

def detect_timestamp_column(
    df: pd.DataFrame,
    review_columns: list[str],
) -> Optional[str]:

    normalized_columns = {
        str(col).lower().strip(): col
        for col in df.columns
    }

    # --------------------------------------------------------
    # Prioritas utama: kolom timestamp
    # --------------------------------------------------------

    if "timestamp" in normalized_columns:
        return normalized_columns[
            "timestamp"
        ]

    review_set = set(review_columns)

    # --------------------------------------------------------
    # Fallback berdasarkan pola nama kolom
    # --------------------------------------------------------

    for col in df.columns:

        if col in review_set:
            continue

        col_lower = (
            str(col)
            .lower()
            .strip()
        )

        if any(
            pattern in col_lower
            for pattern in TIMESTAMP_PATTERNS
        ):
            return col

    return None


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def normalize_text_series(
    series: pd.Series,
) -> pd.Series:

    result = (
        series
        .astype("string")
        .fillna("")
        .str.strip()
    )

    # --------------------------------------------------------
    # NULL-LIKE TEXT
    # --------------------------------------------------------

    null_mask = (
        result
        .str.lower()
        .isin(NULL_TEXT_VALUES)
    )

    result = result.mask(
        null_mask,
        "",
    )

    # --------------------------------------------------------
    # REMOVE HTML
    # --------------------------------------------------------

    result = result.str.replace(
        r"<[^>]+>",
        " ",
        regex=True,
    )

    # --------------------------------------------------------
    # REMOVE URL
    # --------------------------------------------------------

    result = result.str.replace(
        r"(?:https?://|www\.)\S+",
        " ",
        regex=True,
    )

    # --------------------------------------------------------
    # NORMALIZE REPEATED CHARACTERS
    #
    # contoh:
    # baguuuusss -> baguuss
    # --------------------------------------------------------

    result = result.str.replace(
        r"(.)\1{2,}",
        r"\1\1",
        regex=True,
    )

    # --------------------------------------------------------
    # NORMALIZE WHITESPACE
    # --------------------------------------------------------

    result = (
        result
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    return result


# ============================================================
# COMBINE REVIEW COLUMNS
# ============================================================

def combine_review_columns(
    df: pd.DataFrame,
    review_columns: list[str],
) -> pd.Series:

    if not review_columns:
        raise ValueError(
            "review_columns kosong"
        )

    combined = pd.Series(
        "",
        index=df.index,
        dtype="string",
    )

    valid_column_found = False

    for col in review_columns:

        if col not in df.columns:
            continue

        valid_column_found = True

        text = normalize_text_series(
            df[col]
        )

        combined = combined.str.cat(
            text,
            sep=" ",
        )

    if not valid_column_found:
        raise ValueError(
            "Kolom review tidak ditemukan pada chunk"
        )

    combined = (
        combined
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    return combined


# ============================================================
# TIMESTAMP CLEANING
# ============================================================

def clean_timestamp(
    series: pd.Series,
) -> pd.Series:

    cleaned = (
        series
        .astype("string")
        .str.split(
            "|",
            regex=False,
        )
        .str[0]
        .str.strip()
    )

    result = pd.to_datetime(
        cleaned,
        errors="coerce",
    )

    return result


# ============================================================
# CLEAN ONE CHUNK
# ============================================================

def clean_chunk(
    df: pd.DataFrame,
    product_id: str,
    source_file: str,
    review_columns: list[str],
    timestamp_column: Optional[str],
    row_offset: int = 0,
    min_review_length: int = 2,
) -> pd.DataFrame:

    output_columns = [
        "review_id",
        "product_id",
        "review",
        "text_light",
        "timestamp",
        "row_order",
        "source_file",
    ]

    if df.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    df = df.copy()

    # --------------------------------------------------------
    # ROW ORDER
    # --------------------------------------------------------

    df["row_order"] = (
        np.arange(
            len(df),
            dtype=np.int64,
        )
        + row_offset
    )

    # --------------------------------------------------------
    # PRODUCT ID
    # --------------------------------------------------------

    df["product_id"] = (
        product_id
    )

    # --------------------------------------------------------
    # REVIEW
    # --------------------------------------------------------

    df["review"] = (
        combine_review_columns(
            df=df,
            review_columns=review_columns,
        )
    )

    # --------------------------------------------------------
    # TEXT LIGHT
    #
    # Jalur A untuk IndoBERT.
    # Pada tahap ini sama dengan hasil basic cleaning.
    # --------------------------------------------------------

    df["text_light"] = (
        df["review"]
    )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    if (
        timestamp_column is not None
        and timestamp_column in df.columns
    ):

        df["timestamp"] = (
            clean_timestamp(
                df[timestamp_column]
            )
        )

    else:

        df["timestamp"] = pd.Series(
            pd.NaT,
            index=df.index,
            dtype="datetime64[ns]",
        )

    # --------------------------------------------------------
    # MINIMUM REVIEW LENGTH
    # --------------------------------------------------------

    review_length = (
        df["review"]
        .str.len()
        .fillna(0)
    )

    valid_review_mask = (
        review_length
        >= min_review_length
    )

    df = (
        df[
            valid_review_mask
        ]
        .copy()
    )

    # --------------------------------------------------------
    # EXACT DUPLICATE REVIEW
    #
    # Duplicate hanya dibuang dalam product yang sama.
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=[
                "product_id",
                "review",
            ],
            keep="first",
        )
        .copy()
    )

    # --------------------------------------------------------
    # SOURCE FILE
    # --------------------------------------------------------

    df["source_file"] = (
        source_file
    )

    # --------------------------------------------------------
    # REVIEW ID
    # --------------------------------------------------------

    df["review_id"] = (
        df["product_id"].astype(str)
        + "__"
        + df["row_order"].astype(str)
    )

    # --------------------------------------------------------
    # OUTPUT SCHEMA
    # --------------------------------------------------------

    return (
        df[
            output_columns
        ]
        .reset_index(
            drop=True
        )
    )


# ============================================================
# DATA PREPROCESSOR
# ============================================================

class DataPreprocessor:
    """Pipeline lama per produk; sudah digantikan oleh modul src.cleaning."""

    def __init__(
        self,
        config: ExperimentConfig,
    ) -> None:

        self.config = config

        self.config.validate()
        self.config.create_directories()

    # ========================================================
    # DISCOVER FILES
    # ========================================================

    def discover_files(
        self,
    ) -> list[Path]:

        if not self.config.data_dir.exists():
            raise FileNotFoundError(
                f"Tidak ada data di "
                f"{self.config.data_dir}"
            )

        files = sorted(
            self.config.data_dir.glob(
                "*.csv"
            )
        )

        if not files:
            raise FileNotFoundError(
                f"Tidak ada file CSV di "
                f"{self.config.data_dir}"
            )

        return files

    # ========================================================
    # PRODUCT OUTPUT DIRECTORY
    # ========================================================

    def get_product_output_dir(
        self,
        product_id: str,
    ) -> Path:

        safe_product_id = (
            make_safe_filename(
                product_id
            )
        )

        return (
            self.config.cleaned_dir
            / safe_product_id
        )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    def get_done_marker(
        self,
        product_id: str,
    ) -> Path:

        safe_product_id = (
            make_safe_filename(
                product_id
            )
        )

        return (
            self.config.cleaning_done_dir
            / f"{safe_product_id}.done"
        )

    # ========================================================
    # CLEAN SINGLE FILE
    # ========================================================

    def clean_single_file(
        self,
        filepath: Path,
    ) -> dict:
        """Cleaning lama; pengaman ini juga berlaku saat dipanggil clean_all."""

        if (self.config.cleaned_dir / "reviews_clean.parquet").exists():
            raise RuntimeError(
                "format baru terdeteksi; pakai python -m src.cleaning.run"
            )

        filepath = Path(
            filepath
        )

        filename = (
            filepath.name
        )

        product_id = (
            filepath
            .stem
            .strip()
        )

        product_output_dir = (
            self.get_product_output_dir(
                product_id
            )
        )

        done_marker = (
            self.get_done_marker(
                product_id
            )
        )

        meta_path = (
            product_output_dir
            / "_meta.json"
        )

        # ----------------------------------------------------
        # RESUME / SKIP FILE YANG SUDAH SELESAI
        # ----------------------------------------------------

        if (
            done_marker.exists()
            and meta_path.exists()
            and not self.config.overwrite
        ):

            try:
                with open(
                    meta_path,
                    "r",
                    encoding="utf-8",
                ) as file:

                    metadata = (
                        json.load(file)
                    )

            except Exception:
                metadata = {}

            return {
                "file": filename,
                "product_id": product_id,
                "status": "skipped",
                "encoding": metadata.get(
                    "encoding"
                ),
                "review_columns": metadata.get(
                    "review_columns"
                ),
                "timestamp_column": metadata.get(
                    "timestamp_column"
                ),
                "rows_input": metadata.get(
                    "rows_input"
                ),
                "rows_output": metadata.get(
                    "rows_output"
                ),
                "rows_removed": metadata.get(
                    "rows_removed"
                ),
                "parts": metadata.get(
                    "parts"
                ),
                "error": None,
            }

        # ----------------------------------------------------
        # HAPUS OUTPUT LAMA
        # ----------------------------------------------------

        if product_output_dir.exists():
            shutil.rmtree(
                product_output_dir
            )

        product_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if done_marker.exists():
            done_marker.unlink()

        # ----------------------------------------------------
        # PROCESSING
        # ----------------------------------------------------

        try:

            # ------------------------------------------------
            # SAMPLE FILE
            # ------------------------------------------------

            sample_df, encoding = (
                read_csv_sample(
                    filepath=filepath,
                    nrows=self.config.sample_rows,
                )
            )

            # ------------------------------------------------
            # DETECT REVIEW COLUMNS
            # ------------------------------------------------

            review_columns = (
                detect_review_columns(
                    sample_df
                )
            )

            if not review_columns:
                raise ValueError(
                    "Tidak menemukan kolom review"
                )

            # ------------------------------------------------
            # DETECT TIMESTAMP COLUMN
            # ------------------------------------------------

            timestamp_column = (
                detect_timestamp_column(
                    df=sample_df,
                    review_columns=review_columns,
                )
            )

            # ------------------------------------------------
            # COUNTERS
            # ------------------------------------------------

            total_input = 0
            total_output = 0

            row_offset = 0
            part_number = 0

            # ------------------------------------------------
            # CHUNK READER
            # ------------------------------------------------

            reader = pd.read_csv(
                filepath,
                dtype=str,
                keep_default_na=False,
                chunksize=self.config.chunk_size,
                encoding=encoding,
            )

            # ------------------------------------------------
            # PROCESS CHUNKS
            # ------------------------------------------------

            for chunk in reader:

                input_rows = len(chunk)

                total_input += (
                    input_rows
                )

                cleaned = clean_chunk(
                    df=chunk,
                    product_id=product_id,
                    source_file=filename,
                    review_columns=review_columns,
                    timestamp_column=timestamp_column,
                    row_offset=row_offset,
                    min_review_length=(
                        self.config.min_review_length
                    ),
                )

                # --------------------------------------------
                # Update offset berdasarkan raw chunk
                # --------------------------------------------

                row_offset += (
                    input_rows
                )

                if cleaned.empty:
                    continue

                output_rows = (
                    len(cleaned)
                )

                total_output += (
                    output_rows
                )

                # --------------------------------------------
                # OUTPUT PART
                # --------------------------------------------

                output_path = (
                    product_output_dir
                    / f"part_{part_number:06d}.parquet"
                )

                cleaned.to_parquet(
                    output_path,
                    index=False,
                    compression=(
                        self.config.parquet_compression
                    ),
                )

                part_number += 1

                del cleaned

            # =================================================
            # SEMUA CHUNK SUDAH SELESAI
            #
            # Penting:
            # bagian ini di luar FOR tetapi masih di dalam TRY.
            # =================================================

            rows_removed = (
                total_input
                - total_output
            )

            # ------------------------------------------------
            # METADATA
            # ------------------------------------------------

            metadata = {
                "file": filename,
                "product_id": product_id,
                "encoding": encoding,
                "review_columns": review_columns,
                "timestamp_column": timestamp_column,
                "rows_input": int(
                    total_input
                ),
                "rows_output": int(
                    total_output
                ),
                "rows_removed": int(
                    rows_removed
                ),
                "parts": int(
                    part_number
                ),
            }

            with open(
                meta_path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    metadata,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            # ------------------------------------------------
            # DONE MARKER
            #
            # Baru dibuat setelah seluruh chunk dan metadata
            # berhasil disimpan.
            # ------------------------------------------------

            done_marker.touch()

            # ------------------------------------------------
            # SUCCESS RESULT
            # ------------------------------------------------

            return {
                "file": filename,
                "product_id": product_id,
                "status": "success",
                "encoding": encoding,
                "review_columns": "|".join(
                    review_columns
                ),
                "timestamp_column": timestamp_column,
                "rows_input": int(
                    total_input
                ),
                "rows_output": int(
                    total_output
                ),
                "rows_removed": int(
                    rows_removed
                ),
                "parts": int(
                    part_number
                ),
                "error": None,
            }

        # ====================================================
        # ERROR HANDLING
        #
        # `except` HARUS sejajar dengan `try`.
        # ====================================================

        except Exception as error:

            # ------------------------------------------------
            # Hapus output parsial supaya file gagal tidak
            # dianggap sebagai hasil preprocessing valid.
            # ------------------------------------------------

            if product_output_dir.exists():
                shutil.rmtree(
                    product_output_dir
                )

            if done_marker.exists():
                done_marker.unlink()

            return {
                "file": filename,
                "product_id": product_id,
                "status": "failed",
                "encoding": None,
                "review_columns": None,
                "timestamp_column": None,
                "rows_input": None,
                "rows_output": None,
                "rows_removed": None,
                "parts": None,
                "error": str(error),
            }

    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    def save_manifest(
        self,
        results: list[dict],
    ) -> pd.DataFrame:

        manifest = pd.DataFrame(
            results
        )

        manifest.to_csv(
            self.config.cleaning_manifest_path,
            index=False,
        )

        return manifest

    # ========================================================
    # CLEAN ALL
    # ========================================================

    def clean_all(
        self,
    ) -> pd.DataFrame:

        files = (
            self.discover_files()
        )

        results: list[dict] = []

        for filepath in tqdm(
            files,
            desc="Cleaning Files",
        ):

            result = (
                self.clean_single_file(
                    filepath
                )
            )

            results.append(
                result
            )

        # ----------------------------------------------------
        # FINAL MANIFEST
        # ----------------------------------------------------

        manifest = (
            self.save_manifest(
                results
            )
        )

        self.print_summary(
            manifest
        )

        return manifest

    # ========================================================
    # SUMMARY
    # ========================================================

    def print_summary(
        self,
        manifest: pd.DataFrame,
    ) -> None:

        if manifest.empty:
            print(
                "Manifest kosong"
            )
            return

        success_count = int(
            manifest[
                "status"
            ]
            .eq("success")
            .sum()
        )

        skipped_count = int(
            manifest[
                "status"
            ]
            .eq("skipped")
            .sum()
        )

        failed_count = int(
            manifest[
                "status"
            ]
            .eq("failed")
            .sum()
        )

        processed = (
            manifest[
                manifest[
                    "status"
                ]
                .isin(
                    [
                        "success",
                        "skipped",
                    ]
                )
            ]
            .copy()
        )

        # ----------------------------------------------------
        # ROW COUNTS
        # ----------------------------------------------------

        rows_input = pd.to_numeric(
            processed["rows_input"],
            errors="coerce",
        )

        rows_output = pd.to_numeric(
            processed["rows_output"],
            errors="coerce",
        )

        total_input = int(
            rows_input
            .fillna(0)
            .sum()
        )

        total_output = int(
            rows_output
            .fillna(0)
            .sum()
        )

        total_removed = (
            total_input
            - total_output
        )

        # ----------------------------------------------------
        # PRINT SUMMARY
        # ----------------------------------------------------

        print("Summary:")
        print(
            f"Success: {success_count}"
        )
        print(
            f"Skipped: {skipped_count}"
        )
        print(
            f"Failed: {failed_count}"
        )
        print(
            f"Total input: {total_input}"
        )
        print(
            f"Total output: {total_output}"
        )
        print(
            f"Total removed: {total_removed}"
        )

        if total_input > 0:

            removed_percentage = (
                total_removed
                / total_input
                * 100
            )

            print(
                "Removed percentage: "
                f"{removed_percentage:.2f}%"
            )

        print()
        print("Manifest:")
        print(
            self.config.cleaning_manifest_path
        )

        # ----------------------------------------------------
        # FAILED FILES
        # ----------------------------------------------------

        if failed_count > 0:

            print()
            print(
                f"Failed Files: "
                f"{failed_count}"
            )

            failed = (
                manifest[
                    manifest[
                        "status"
                    ]
                    == "failed"
                ]
            )

            for _, row in (
                failed
                .head(10)
                .iterrows()
            ):

                print(
                    f"- {row['file']}"
                )

                print(
                    row["error"]
                )

            if failed_count > 10:
                print(
                    f"{failed_count - 10:,} "
                    "more failed files..."
                )

    # ========================================================
    # GET CLEANED FILES
    # ========================================================

    def get_cleaned_files(
        self,
    ) -> pd.DataFrame:
        """Baca manifest lama; tabel baru dibaca lewat src.cleaning.load_clean_reviews."""

        if (self.config.cleaned_dir / "reviews_clean.parquet").exists():
            raise RuntimeError(
                "format baru terdeteksi; gunakan src.cleaning.load_clean_reviews() "
                "untuk membaca reviews_clean.parquet"
            )

        path = (
            self.config.cleaning_manifest_path
        )

        if not path.exists():
            raise FileNotFoundError(
                "Cleaning manifest belum tersedia"
            )

        manifest = (
            pd.read_csv(
                path
            )
        )

        return (
            manifest[
                manifest[
                    "status"
                ]
                .isin(
                    [
                        "success",
                        "skipped",
                    ]
                )
            ]
            .reset_index(
                drop=True
            )
        )


# ============================================================
# QUALITY CHECK
# ============================================================

def inspect_cleaned_data(
    cleaned_dir: Path,
    sample_files: int = 10,
    random_state: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """Inspeksi format lama; format baru memakai src.cleaning.load_clean_reviews."""

    cleaned_dir = Path(
        cleaned_dir
    )

    if (cleaned_dir / "reviews_clean.parquet").exists():
        raise RuntimeError(
            "format baru terdeteksi; gunakan src.cleaning.load_clean_reviews() "
            "untuk membaca reviews_clean.parquet"
        )

    parquet_files = sorted(
        cleaned_dir.rglob(
            "*.parquet"
        )
    )

    if not parquet_files:
        raise FileNotFoundError(
            f"Tidak ada file parquet di "
            f"{cleaned_dir}"
        )

    # --------------------------------------------------------
    # SAMPLING FILE
    # --------------------------------------------------------

    rng = (
        np.random.default_rng(
            random_state
        )
    )

    sample_size = min(
        sample_files,
        len(parquet_files),
    )

    indices = rng.choice(
        len(parquet_files),
        size=sample_size,
        replace=False,
    )

    selected_files = [
        parquet_files[i]
        for i in indices
    ]

    samples: list[pd.DataFrame] = []

    for filepath in selected_files:

        df = (
            pd.read_parquet(
                filepath
            )
        )

        if not df.empty:
            samples.append(
                df
            )

    if not samples:
        raise ValueError(
            "Tidak ada data di file parquet yang disampling"
        )

    sample = pd.concat(
        samples,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # QUALITY REPORT
    # --------------------------------------------------------

    report = {
        "total_parquet_files": (
            len(parquet_files)
        ),

        "sampled_parquet_files": (
            len(selected_files)
        ),

        "sampled_rows": (
            len(sample)
        ),

        "unique_products": (
            sample[
                "product_id"
            ]
            .nunique()
            if "product_id" in sample.columns
            else None
        ),

        "missing_review": (
            int(
                sample[
                    "review"
                ]
                .isna()
                .sum()
            )
            if "review" in sample.columns
            else None
        ),

        "missing_text_light": (
            int(
                sample[
                    "text_light"
                ]
                .isna()
                .sum()
            )
            if "text_light" in sample.columns
            else None
        ),

        "missing_timestamp": (
            int(
                sample[
                    "timestamp"
                ]
                .isna()
                .sum()
            )
            if "timestamp" in sample.columns
            else None
        ),

        "valid_timestamp_ratio": (
            float(
                sample[
                    "timestamp"
                ]
                .notna()
                .mean()
            )
            if "timestamp" in sample.columns
            else None
        ),

        "duplicate_review_id": (
            int(
                sample[
                    "review_id"
                ]
                .duplicated()
                .sum()
            )
            if "review_id" in sample.columns
            else None
        ),

        "duplicate_review_within_product": (
            int(
                sample
                .duplicated(
                    subset=[
                        "product_id",
                        "review",
                    ]
                )
                .sum()
            )
            if (
                "product_id" in sample.columns
                and "review" in sample.columns
            )
            else None
        ),
    }

    return report, sample
