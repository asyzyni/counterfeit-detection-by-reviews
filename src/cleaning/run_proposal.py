"""Buat tabel proposal v2 setelah kamus CSV slang,baku tersedia.

    python -m src.cleaning.run_proposal --slang-csv "references/slang.csv"

Tanpa kamus run berhenti. Audit parsial dapat dilakukan lewat fungsi proposal;
tidak ada opsi CLI untuk menulis parquet final tanpa normalisasi slang.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

from .config import CleaningConfig
from .proposal import (
    build_proposal_table, format_proposal_report, load_slang_dictionary, verify_original_columns,
)

EXIT_OK = 0
EXIT_STOPPED = 2


def build_parser() -> argparse.ArgumentParser:
    defaults = CleaningConfig()
    parser = argparse.ArgumentParser(description="Tambah text_proposal; semua kolom/baris lama tetap")
    parser.add_argument("--input", type=Path, default=defaults.reviews_path)
    parser.add_argument("--output", type=Path, default=defaults.output_dir / "reviews_clean_v2.parquet")
    parser.add_argument("--report", type=Path,
                        default=defaults.output_dir.parent.parent / "reports" / "preprocessing_proposal_v2.md")
    parser.add_argument("--slang-csv", type=Path, help="Wajib: CSV pengguna dua kolom slang,baku")
    return parser


def run(args: argparse.Namespace) -> int:
    paths = (args.input.expanduser(), args.output.expanduser(), args.report.expanduser())
    input_path, output_path, report_path = paths
    if len({path.resolve() for path in paths}) != 3:
        print("BERHENTI: path input, output, dan report harus berbeda")
        return EXIT_STOPPED
    for path in (output_path, report_path):
        if path.exists():
            print(f'BERHENTI: file sudah ada, tidak boleh ditimpa: "{path}"')
            return EXIT_STOPPED
    if args.slang_csv is None:
        print("BERHENTI: kamus slang belum tersedia. Sediakan CSV dua kolom slang,baku lewat --slang-csv.")
        return EXIT_STOPPED
    try:
        dictionary_path = args.slang_csv.expanduser()
        slang = load_slang_dictionary(dictionary_path)
        original = pd.read_parquet(input_path)
        updated, stats = build_proposal_table(original, slang)
        # Periksa round-trip sebelum menyimpan, agar mismatch tidak menghasilkan file final.
        buffer = io.BytesIO()
        updated.to_parquet(buffer, index=False, compression="snappy")
        buffer.seek(0)
        restored = pd.read_parquet(buffer)
        verify_original_columns(original, restored)
        pd.testing.assert_frame_equal(updated, restored)
        report = format_proposal_report(restored, stats, input_path, output_path, dictionary_path)
    except (OSError, ValueError, AssertionError) as exc:
        print(f"BERHENTI: {exc}")
        return EXIT_STOPPED

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(buffer.getvalue())
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(report)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f'Keluaran: "{output_path}"; laporan: "{report_path}"')
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
