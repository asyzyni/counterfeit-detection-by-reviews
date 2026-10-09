from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

from .audit import (
    aspect_tag_candidates,
    build_cleaning_audit,
    checks_to_frame,
    preset_phrase_candidates,
    product_summary,
    run_checks,
    timestamp_format_report,
)
from .build import BuildResult, build_all
from .config import CleaningConfig
from .export import export_tabel_contoh
from .file_map import FileMap, MappingError, load_file_map


# ============================================================
# EXIT CODES
# ============================================================

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_STOPPED = 2


# ============================================================
# PENULISAN
# ============================================================

def _write_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:

    df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs(
    result: BuildResult,
    file_map: FileMap,
    config: CleaningConfig,
) -> pd.DataFrame:
    """Tulis parquet + laporan. Kembalikan tabel hasil pemeriksaan."""

    output_dir = Path(config.output_dir)
    reports = config.reports_dir

    output_dir.mkdir(parents=True, exist_ok=False)
    reports.mkdir()

    result.reviews.to_parquet(
        config.reviews_path,
        index=False,
        compression=config.parquet_compression,
    )

    manifest = pd.concat(
        [
            result.manifest,
            result.excluded_files.rename(columns={"alasan": "status_alasan"}).assign(
                status=lambda d: "dikecualikan: " + d["status_alasan"]
            ).drop(columns="status_alasan"),
        ],
        ignore_index=True,
    ).sort_values("file", kind="mergesort")

    checks = checks_to_frame(run_checks(result, file_map, config))

    tables = {
        "column_map.csv": result.column_map,
        "file_manifest.csv": manifest,
        "cleaning_audit.csv": build_cleaning_audit(result, file_map, config),
        "duplicate_files.csv": result.duplicate_files,
        "aspect_tag_candidates.csv": aspect_tag_candidates(result, config),
        "preset_phrase_candidates.csv": preset_phrase_candidates(result, config),
        "product_summary.csv": product_summary(result.reviews, config),
        "excluded_files.csv": result.excluded_files,
        "timestamp_formats.csv": timestamp_format_report(result.reviews),
        "fragment_counts.csv": result.fragment_counts,
        "tabel_contoh.csv": export_tabel_contoh(result.reviews, n=3),
        "tabel_contoh_anonim.csv": export_tabel_contoh(result.reviews, n=3, anonimkan=True),
        "validation_checks.csv": checks,
    }

    for name, table in tables.items():
        _write_csv(table, reports / name)

    # --------------------------------------------------------
    # Hash semua file output (untuk cek determinisme antar-run)
    # --------------------------------------------------------

    files = sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and path.name != "output_hashes.csv"
    )
    hashes = pd.DataFrame(
        {
            "file": [str(path.relative_to(output_dir)) for path in files],
            "sha256": [_sha256(path) for path in files],
        }
    )
    _write_csv(hashes, reports / "output_hashes.csv")

    return checks


# ============================================================
# RINGKASAN KONSOL
# ============================================================

def print_summary(
    result: BuildResult,
    checks: pd.DataFrame,
    config: CleaningConfig,
) -> None:

    reviews = result.reviews
    manifest = result.manifest
    summary = product_summary(reviews, config)

    print("\n" + "=" * 64)
    print("RINGKASAN CLEANING")
    print("=" * 64)
    print(f"File diproses           : {int((manifest['status'] == 'diproses').sum())}")
    print(f"File dikecualikan       : {len(result.excluded_files)}")
    print(f"Baris sumber            : {int(manifest['n_baris_sumber'].sum()):,}")
    print(f"Baris kosong (luar tabel): {int(manifest['n_baris_kosong'].sum()):,}")
    print(f"Baris di tabel          : {len(reviews):,}")
    print(f"  is_empty              : {int(reviews['is_empty'].sum()):,}")
    print(f"  is_short              : {int(reviews['is_short'].sum()):,}")
    print(f"  is_technical_duplicate: {int(reviews['is_technical_duplicate'].sum()):,}")
    print(f"  is_text_duplicate     : {int(reviews['is_text_duplicate'].sum()):,}")
    print(f"  include_for_inference : {int(reviews['include_for_inference'].sum()):,}")
    print(f"  timestamp NaT         : {int(reviews['timestamp'].isna().sum()):,}")
    print(f"Produk (product_id)     : {reviews['product_id'].nunique()}")

    for threshold in config.product_thresholds:
        print(f"  produk >= {threshold:>2} include  : {int(summary[f'lolos_{threshold}'].sum())}")

    print(f"Grup file identik       : {result.duplicate_files['file_dup_group'].nunique()}")

    print("\nPEMERIKSAAN OTOMATIS")
    for row in checks.itertuples():
        mark = "LULUS" if row.lulus else "GAGAL"
        print(f"[{mark}] {row.nama} -> {row.detail}")


# ============================================================
# CLI
# ============================================================

def parse_args(
    argv: list[str] | None = None,
) -> CleaningConfig:

    defaults = CleaningConfig()
    parser = argparse.ArgumentParser(
        prog="python -m src.cleaning.run",
        description="Cleaning data ulasan (Data Beneran) -> reviews_clean.parquet",
    )

    def optional_int(value: str) -> int | None:
        return None if value.lower() == "none" else int(value)

    parser.add_argument("--raw-dir", type=Path, default=defaults.raw_dir)
    parser.add_argument("--mapping-filename", default=defaults.mapping_filename)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--min-review-length", type=int, default=defaults.min_review_length)
    parser.add_argument("--aspect-tag-vocab", type=Path, default=None,
                        help="daftar tag aspek yang disetujui (.txt atau .csv berkolom setuju)")
    parser.add_argument("--exclude-duplicate-file-groups", action="store_true")
    parser.add_argument("--no-nfkc", action="store_true",
                        help="matikan normalisasi Unicode NFKC pada text_light")
    parser.add_argument("--allow-date-only-technical-dup", action="store_true",
                        help="duplikat teknis juga untuk timestamp tanpa jam")
    parser.add_argument("--expected-mapping-rows", type=optional_int,
                        default=defaults.expected_mapping_rows,
                        help="angka atau 'none' untuk tidak memeriksa")
    parser.add_argument("--expected-ok-files", type=optional_int,
                        default=defaults.expected_ok_files)

    args = parser.parse_args(argv)

    return CleaningConfig(
        raw_dir=args.raw_dir,
        mapping_filename=args.mapping_filename,
        output_dir=args.output_dir,
        min_review_length=args.min_review_length,
        aspect_tag_vocab_path=args.aspect_tag_vocab,
        exclude_duplicate_file_groups=args.exclude_duplicate_file_groups,
        unicode_nfkc=not args.no_nfkc,
        technical_dup_require_time=not args.allow_date_only_technical_dup,
        expected_mapping_rows=args.expected_mapping_rows,
        expected_ok_files=args.expected_ok_files,
    )


def run(
    config: CleaningConfig,
) -> int:

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("BERHENTI: pyarrow tidak tersedia; parquet tidak bisa ditulis.")
        return EXIT_STOPPED

    config.validate()

    if Path(config.output_dir).exists():
        print(f"BERHENTI: folder output sudah ada: {config.output_dir}")
        return EXIT_STOPPED

    try:
        file_map = load_file_map(config)
    except MappingError as error:
        print(f"BERHENTI (pemetaan): {error}")
        return EXIT_STOPPED

    result = build_all(config, file_map)

    n_ok = len(file_map.ok)
    no_review_rate = len(result.skipped_no_review) / max(n_ok, 1)
    if no_review_rate > config.max_files_without_review_rate:
        print(
            f"BERHENTI: {len(result.skipped_no_review)} file ({no_review_rate:.1%}) "
            f"tanpa kolom review_text: {result.skipped_no_review}"
        )
        return EXIT_STOPPED

    checks = write_outputs(result, file_map, config)
    print_summary(result, checks, config)
    print(f"\nOutput: {config.output_dir}")

    return EXIT_OK if checks["lulus"].all() else EXIT_CHECK_FAILED


def main(argv: list[str] | None = None) -> None:
    sys.exit(run(parse_args(argv)))


if __name__ == "__main__":
    main()
