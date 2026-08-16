from numpy import sort
from posixpath import sep
from numpy import average
from typing_extensions import runtime
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

# Helper 
def make_safe_filename(
    value:str, 
) -> str: 
    value = str(value).strip() 
    value = re.sub(
        r"[^\w\-.]+",
        "_", value, 
    )
    value = value.strip("-")

    if not value: 
        value = "unknown_product" 

    return value

## CSV READDER 

def read_csv_sample(
    filepath: Path, 
    nrows: int=500, 
) -> tuple[pd.DataFrame, str]:
    encodings = [
        "utf-8", 
        "latin1", 
        "iso-8859-1", 
        "cp1252", 
        "utf-16", 
    ]

    for encoding in encodings: 
        try: 
            df = pd.read_csv(
                filepath, dtype=str, keep_default_na=False, 
                nrows=nrows, encoding=encoding
            )

            return df, encoding 
        except Exception as error: 
            last_error = error 

    raise RuntimeError(
        f"Tidak bisa membaca {filepath.name} dengan encoding yang diujicoba" 
        f"error terakhir : {last_error}"
    ) 

## column detector 

def detect_review_columns(
    df: pd.DataFrame, 
) -> list[str]: 
    
    columns = list(df.columns)

    normalized_columns = {
        str(col).lower().strip(); col 
        for col in columns
    }

    if "review" in normalized_columns: 
        return [
            normalized_columns['review']
        ]
    
    review_columns = []

    for col in columns:
        col_lower = (
            str(col).lower().strip()
        )

        if col_lower in METADATA_COLUMNS:
            continue

        if re.fullmatch(
            r"data\d+", col_lower
        ):
            review_columns.append(col)
            continue

        if any(
            pattern in col_lower
            for pattern in REVIEW_PATTERNS
        ):

            review_columns.append(
                col
            )

    if review_columns:
        review_columns

    for col in columns:
        col_lower = (
            str(col).lower().strip()
        )

        if col_lower in METADATA_COLUMNS: 
            continue

        ## pelaajri lagi !!!!!!
        sample = (
            df[col].astype("string").replace("", pd.NA).dropna().head(30)
        )

        if sample.empty: 
            continue

        average_length = (
            sample.str.len().mean()
        )

        if (
            pd.notna(average_length) and average_length > 20
        ): 
            review_columns.append(
                col
            )

    return review_columns 


## timestamp detection 

def detect_timestamp_column(
    df: pd.DataFrame,
    review_column: list[str],
) -> Optional[str]:

    normalized_columns = {
        str(col).lower().strip():col
        for col in df.columns
    }

    if "timestamp" in normalized_columns: 
        return normalized_columns[
            "timestamp"
        ]

    review_set = set(review_column)

    for col in df.columns:
        if col in review_set:
            continue

        col_lower = (
            str(col).lower().strip()
        )

        if any(
            pattern in col_lower
            for pattern in TIMESTAMP_PATTERNS
        ):

            return col
        
    return None

## TEXT NORMALISASI 

def normalize_text_series(
    series: pd.Series,
) -> pd.Series:
    result = (
        series.astype("string").fillna("").str.strip()
    )

    null_mask = (
        result.str.lower().isin(NULL_TEXT_VALUES)
    )

    result = result.mask(
        null_mask, ""
    )

    ## WHITESPACE 

    result = (
        result.str.replace(
            r"\s+", " ". regex=True
        ).str.strip()
    )

    return result

## column combination 

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
        dtype="string"
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
            text, sep=" ", 
        )

    if not valid_column_found: 
        raise ValueError(
            "kolom review tidak dtemukan pada chunk"
        )

    combined = (
        combined.str.replace(
            r"\s+", " ", regex=True
        ).str.strip()
    )

    return combined

# Timestamp cleaning 

def clean_timestamp(
    series: pd.Series,
) -> pd.Series:

    cleaned = (
        series.astype("string").str.split(
            "|", regex=False  
        ).str[0].str.strip()
    )

    result = pd.to_datetime(
        cleaned, errors="coerce"
    )

    return result 

## cleane one chukn 

def clean_chunk(
    df: pd.DataFrame, 
    product_id: str, 
    source_file: str, 
    review_columns: list[str], 
    timestamp_columns: Optional[str]. 
    row_offset: int = 0, 
    min_review_length: int = 2,
) -> pd.DataFrame: 
    
    if df.empty:
        return pd.DataFrame(
            columns=[
                "review_id", "product_id", "review", "timestamp", 
                "row_order", "source_file"
            ]
        )
    
    df = df.copy()

    # row order 

    df['row_order'] = (
        np.arange(
            len(df), dtype=np.int64
        ) + row_offset 
    )

    # product id

    df['product_id'] = (
        product_id
    )

    # review 
    df["review"] = (
        combine_review_columns(
            df=df, 
            review_columns=review_columns,
        )
    )

    # timestamp 

    if (
        timestamp_columns is not None and timestamp_columns in df.columns
    ): 

        df["timestamp"] = (
            clean_timestamp(
                df[timestamp_columns]
            )
        )

    else: 
        df['timestamp'] = pd.Series(
            pd.NaT, index=df.index, dtype="datetime64[ns]", 
        )

    # review length 

    review_length = (
        df['review'].str.len().fillna(0)
    )

    valid_review_mask = (
        review_length >= min_review_length
    )

    df = df[
        valid_review_mask
    ].copy()


    # source file 

    df['source_file'] = (
        source_file
    )

    # review id 

    df['review_id'] = (
        df['product_id'].astype(str) + "__" + df['row_order'].astype(str) 
    )

    # output schema 
    columns = [
        "review_id", "product_id", "review", "timestamp", "row_order", "source_file"
    ] 

    return (
        df[columns].reset_index(drop=True) 
    )

# data preprocessor 

class DataPreprocessor: 
    def __init__(self, config: ExperimentConfig, ): 
        self.config = config 
        self.config.validate() 

        self.config.create_directories() 

    # discover files func 
    def discover_files(
        self, 
    ) -> list[Path]: 
        if not self.config.data_dir.exists(): 
            raise FileNotFoundError(
                f"tidak ada data di {self.config.data_dir}" 
            )
        
        files = sorted(
            self.config.data_dir.glob(
                "*.csv" 
            )
        )

        if not files: 
            raise FileNotFoundError(
                f"tiak ada fie csv di {self.config.data_dir}"
            )
        
        return files 

    # product output 

    def get_product_output_dir(
        self, product_id: str, x
    ) -> Path: 
        safe_product_id = (
            make_safe_filename(
                product_id
            )
        )

        return (
            self.config.cleaned_dir / safe_product_id 
        )

    # checkpoint 
    def get_done_marker(
        self, product_id:str, 
    ) -> Path: 
        safe_product_id = make_safe_filename(product_id) 
        
        return (
            self.config.cleaning_done_dir / f"{safe_product_id}.done"
        )

    # clean single file 

    def clean_single_file(
        self, filepath: Path, 
    ) -> dict: 
        filepath = Path(
            filepath
        ) 

        filename = (
            filepath.name 
        )

        product_id = (
            filepath.stem.strip() 
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
            product_output_dir / "_meta.json"
        )

        # resume 
        if (
            done_marker.exists() and meta_path.exists() and not self.config.overwrite 
        ): 

            try:
                with open(
                    meta_path, "r", encoding="utf-8"
                ) as file: 
                    metadata = json.load(
                        file 
                    )
            except Exception: 
                metadata = {} 

            return { 
                "file": filename, 
                "product_id" : product_id, 
                "status" : "skipped", 
                "encoding" : metadata.get("encoding"), 
                "review_columns": metadata.get("review_columns"), 
                "timestamp_column": metadata.get("timestamp_column"), 
                "rows_output": metadata.get("rows_output"), 
                "rows_removed": metadata.get("rows_removed"), 
                "parts": metadata.get("parts"), 
                "error": None
            }
        
        # hapus file output lama 
        if product_output_dir.exists(): 
            shutil.rmtree(
                product_output_dir
            )

        product_output_dir.mkdir(
            parents=True, 
            exist_ok=True 
        )

        if done_marker.exists():
            done_marker.unlink() 
        
        try: 
            sample_df, encoding=(
                read_csv_sample(
                    filepath=filepath, 
                    nrows=self.config.sample_rows, 
                )
            )

            review_columns = (
                detect_review_columns(
                    sample_df 
                )
            )

            if not review_columns:
                raise ValueError(
                    "Tidak menemukan kolom review"
                )

            timestamp_column = (
                detect_timestamp_column(
                    df=sample_df, review_columns=review_columns
                )
            )

            total_input = 0 
            total_output = 0 

            row_offset = 0 
            part_number = 0 

            # chunk reader 

            reader = pd.read_csv(
                filepath, dtype=str, 
                keep_default_na=False, 
                chunksize=(
                    self.config.chunksize 
                ), 
                encoding=encoding, 
            )

            # process chunk 

            for chunk in reader: 
                input_rows = (len(chunk)) 

                total_input += (
                    input_rows
                )

                cleaned = clean_chunk(
                    df=chunk, product_id=product_id, 
                    source_file=filename, review_columns=review_columns, 
                    timestamp_column=timestamp_column, 
                    row_offset = row_offset, 
                    min_review_length=(
                        self.config.min_review_length
                    ), 
                )

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

                # save parquet 

                output_path = (
                    product_output_dir / (
                        f"part_{part_number:06d}.parquet"
                    )
                )

                cleaned.to_parquet(
                    output_path, index=False, 
                    compression=(
                        self.config.parquet_compression
                    ), 
                )

                part_number += 1 

                del cleaned 

                rows_removed = (
                    total_input - total_output
                )

                # metadata 

                metadata = {
                    "file":
                        filename,

                    "product_id":
                        product_id,

                    "encoding":
                        encoding,

                    "review_columns":
                        review_columns,

                    "timestamp_column":
                        timestamp_column,

                    "rows_input":
                        int(total_input),

                    "rows_output":
                        int(total_output),

                    "rows_removed":
                        int(rows_removed),

                    "parts":
                        int(part_number),

                }

                with open(
                    meta_path, "w", encoding="utf-8", 
                ) as file: 
                    jso.dump(
                        metadata, file, ensure_ascii=False, indent=2
                    )

                # Checkpoint 
                done_marker.touch() 

                # return 
                return {
                     "file":
                        filename,

                    "product_id":
                        product_id,

                    "status":
                        "success",

                    "encoding":
                        encoding,

                    "review_columns":
                        "|".join(
                            review_columns
                        ),

                    "timestamp_column":
                        timestamp_column,

                    "rows_input":
                        int(total_input),

                    "rows_output":
                        int(total_output),

                    "rows_removed":
                        int(rows_removed),

                    "parts":
                        int(part_number),

                    "error":
                        None,
                }
        
        # Error hadnling 
        except Exception as error: 
            # agar output tidak berhenti setengah jadi 
            if product_output_dir.exists():
                shutil.rmtree