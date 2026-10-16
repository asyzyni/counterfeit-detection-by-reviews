"""Migrasi metadata waktu v4 -> v5; tidak menjalankan inferensi atau mengubah teks.

    python -m src.cleaning.run_timestamps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import export_sekuens as es
from src.temporal.config import HMMConfig
from src.temporal.sequences import build_sequences
from .build import build_all
from .config import CleaningConfig
from .file_map import file_sha256, load_file_map
from .run import write_outputs


TIME_COLUMNS = [
    "timestamp", "timestamp_has_time", "timestamp_status", "seq_in_product",
    "timestamp_is_estimated", "timestamp_anchor", "timestamp_anchor_source",
    "timestamp_relative_raw",
]


def preserve_original_columns(original: pd.DataFrame, rebuilt: pd.DataFrame) -> pd.DataFrame:
    """Periksa seluruh kolom lama selain waktu/urutan; salin kolom tambahan v4."""
    if not original.review_id.is_unique or not rebuilt.review_id.is_unique:
        raise ValueError("review_id tidak unik")
    if set(original.review_id) != set(rebuilt.review_id):
        raise ValueError("review_id berubah antara v4 dan hasil build")
    old = original.set_index("review_id").loc[rebuilt.review_id]
    new = rebuilt.set_index("review_id")
    unchanged = [c for c in new if c in old and c not in TIME_COLUMNS]
    pd.testing.assert_frame_equal(old[unchanged], new[unchanged], check_exact=True)
    absolute = old.timestamp_status.eq("ok")
    absolute_columns = ["timestamp", "timestamp_has_time", "timestamp_status"]
    pd.testing.assert_frame_equal(old.loc[absolute, absolute_columns],
                                  new.loc[absolute, absolute_columns], check_exact=True)
    out = rebuilt.copy()
    for column in old.columns.difference(new.columns, sort=False):
        out[column] = old[column].to_numpy()
    return out


def merge_sentiment_timestamps(sentiment: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Join 1:1 lewat ID; input CSV string menjaga p_*, logit, label dan teks persis."""
    if not sentiment.review_id.is_unique or not cleaned.review_id.is_unique:
        raise ValueError("review_id tidak unik saat join sentimen")
    if set(sentiment.review_id) != set(cleaned.review_id):
        raise ValueError("review_id sentimen dan cleaned berbeda")
    clean = cleaned.set_index("review_id").loc[sentiment.review_id]
    for col in ("review", "text_light", "text_proposal", "include_for_inference", "timestamp_raw"):
        if col in sentiment and col in clean:
            expected = clean[col].fillna("").astype(str).to_numpy()
            if not np.array_equal(sentiment[col].to_numpy(), expected):
                raise ValueError(f"Kolom {col} berbeda pada sentimen v4")
    updated = sentiment.copy()
    for col in TIME_COLUMNS:
        values = clean[col]
        if pd.api.types.is_datetime64_any_dtype(values):
            values = values.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
        updated[col] = values.astype(str).to_numpy()
    untouched = [col for col in sentiment if col not in TIME_COLUMNS]
    pd.testing.assert_frame_equal(sentiment[untouched], updated[untouched], check_exact=True)
    return updated


def temporal_sequences(frame: pd.DataFrame):
    numeric = frame.copy()
    numeric["timestamp"] = pd.to_datetime(numeric.timestamp, format="ISO8601", errors="coerce")
    numeric["include_for_inference"] = es.to_bool(numeric.include_for_inference, "include_for_inference")
    for col in (*HMMConfig().prob_cols, "row_order", "seq_in_product"):
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    return build_sequences(numeric, HMMConfig())[:2]


def run(args: argparse.Namespace) -> dict:
    clean_dir, sentiment_dir, audit_dir = args.output_root / "cleaned_v5", args.output_root / "sentiment/v5", args.output_root / "audit_v5"
    for directory in (clean_dir, sentiment_dir, audit_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Tidak menimpa folder berisi: {directory}")
    inputs = [args.cleaned_v4, args.sentiment_v4, args.mapping]
    before_hashes = {str(p): file_sha256(p) for p in inputs}
    old = pd.read_parquet(args.cleaned_v4)
    sentiment = es.read_input_bytes(args.sentiment_v4.read_bytes())
    mapping = pd.read_csv(args.mapping, dtype=str, keep_default_na=False)
    config = CleaningConfig(
        raw_dir=args.raw_dir, output_dir=clean_dir, mapping_filename=str(args.mapping.resolve()),
        expected_mapping_rows=len(mapping), expected_ok_files=int(mapping.status.eq("ok").sum()),
        scrape_dates_path=args.scrape_dates,
    )
    config.validate()
    file_map = load_file_map(config)
    result = build_all(config, file_map)
    result.reviews = preserve_original_columns(old, result.reviews)
    n_include = int(result.reviews.include_for_inference.sum())
    counts = result.reviews.timestamp_status.value_counts().to_dict()
    n_relative = sum(counts.get(s, 0) for s in ("relatif_dikonversi", "relatif_tanpa_acuan"))
    if n_include != args.expected_include or n_relative != args.expected_relative:
        raise ValueError(f"ANGKA TIDAK COCOK: include={n_include}, relatif={n_relative}; "
                         f"harapan={args.expected_include}, {args.expected_relative}")
    # Run kedua benar-benar membangun ulang; tidak memakai hasil run pertama.
    repeat = build_all(config, file_map)
    repeated = preserve_original_columns(old, repeat.reviews)
    pd.testing.assert_frame_equal(result.reviews, repeated, check_exact=True)
    pd.testing.assert_frame_equal(result.anchor_audit, repeat.anchor_audit, check_exact=True)
    updated = merge_sentiment_timestamps(sentiment, result.reviews)
    output_bytes = es.to_csv_bytes(updated)
    roundtrip = es.read_input_bytes(output_bytes)
    pd.testing.assert_frame_equal(updated, roundtrip, check_exact=True)
    _, report_before = temporal_sequences(sentiment)
    sequences_after, report_after = temporal_sequences(updated)
    exports = es.hasilkan_berkas(output_bytes)
    checks = es.periksa_invarian(updated, exports)
    if not all(c["lolos"] for c in checks):
        raise ValueError(f"Invarian ekspor gagal: {checks}")
    export_summary = es.ringkas(updated, exports)
    if export_summary["n_baris_sekuens"] != report_after["n_reviews_used"]:
        raise ValueError("Jumlah sekuens export dan src.temporal berbeda")
    summary = {
        "n_rows": len(result.reviews), "include_for_inference": n_include,
        "timestamp_status": counts,
        "relative_anchor_sources": result.reviews.loc[
            result.reviews.timestamp_relative_raw.ne(""), "timestamp_anchor_source"
        ].value_counts().to_dict(),
        "n_relative_expected": args.expected_relative, "n_relative_actual": n_relative,
        "n_anchor_anomalies": int(result.anchor_audit.mtime_anomaly.sum()),
        "before": report_before, "after": report_after,
        "unchanged_original_columns": [c for c in old if c not in TIME_COLUMNS],
        "sentiment_unchanged_columns": [c for c in sentiment if c not in TIME_COLUMNS],
        "rebuilt_twice_identical": True, "sentiment_csv_roundtrip_identical": True,
        "inference_rerun": False, "input_sha256": before_hashes,
        "input_mtime_range": [str(result.anchor_audit.mtime_local.min()), str(result.anchor_audit.mtime_local.max())],
        "export_checks": checks,
    }
    # Seluruh pemeriksaan di atas dilakukan sebelum menulis keluaran final.
    validation = write_outputs(result, file_map, config)
    if not validation.lulus.all():
        raise ValueError("Pemeriksaan cleaning gagal; lihat validation_checks.csv")
    sentiment_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    (sentiment_dir / "sentimen_ulasan_v5.csv").write_bytes(output_bytes)
    (sentiment_dir / "sekuens_produk_v5.csv").write_bytes(exports["sekuens"])
    (audit_dir / "sekuens_audit_v5.csv").write_bytes(exports["audit"])
    (audit_dir / "ekspor_sekuens_dikeluarkan.csv").write_bytes(exports["dikeluarkan"])
    # Snapshot observasi/lengths yang benar-benar dibentuk src.temporal (tanpa fit HMM).
    np.savez_compressed(sentiment_dir / "hmm_sequences_v5.npz", X=sequences_after.X,
                        P=sequences_after.P, lengths=sequences_after.lengths,
                        product_ids=np.asarray(sequences_after.product_ids, dtype=str))
    summary["input_unchanged"] = all(file_sha256(Path(p)) == h for p, h in before_hashes.items())
    if not summary["input_unchanged"]:
        raise ValueError("Input berubah selama run")
    for path in (audit_dir / "timestamp_migration_summary.json", sentiment_dir / "sentimen_ulasan_v5_provenance.json"):
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_rows", "include_for_inference", "timestamp_status", "before", "after")}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=CleaningConfig().raw_dir)
    parser.add_argument("--mapping", type=Path, default=Path("references/pemetaan_file_toko_produk_v4.csv"))
    parser.add_argument("--scrape-dates", type=Path, default=CleaningConfig().scrape_dates_path)
    parser.add_argument("--cleaned-v4", type=Path, default=Path("outputs/cleaned/reviews_clean_v4.parquet"))
    parser.add_argument("--sentiment-v4", type=Path, default=Path("outputs/sentiment/v4/sentimen_ulasan_v4.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--expected-include", type=int, default=72881)
    parser.add_argument("--expected-relative", type=int, default=1513)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
