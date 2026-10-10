"""Jalankan inferensi sentimen RM 1 pada tabel ulasan bersih dan simpan CSV.

Contoh (Mac M2, MPS):
    export HF_TOKEN=...            # jangan ditulis di kode/notebook/log
    export INDOBERT_SENTIMENT_REVISION=<hash commit 40 karakter>
    python -m src.run_sentiment_inference --device mps --batch-size 16

Contoh uji coba kecil:
    python -m src.run_sentiment_inference --device mps --limit 200 --output outputs/sentiment/uji_200.csv

Kaggle GPU: set HF_TOKEN lewat Secrets, lalu --device cuda --batch-size 64.
Dari folder model lokal (tanpa jaringan): --model-dir "<folder>".

Keluaran: CSV (kolom asli + p_negatif, p_netral, p_positif + label_argmax) dan ringkasan
sanity check (.json dan .md) di sampingnya. File keluaran yang sudah ada TIDAK ditimpa.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .sentiment_inference import (
    LABEL_NAMES,
    PROB_COLUMNS,
    SentimentBundle,
    SentimentModelConfig,
    argmax_label,
    format_summary_md,
    load_sentiment_model,
    predict_sentiment_probs,
    run_model_checks,
    summarize_predictions,
)

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_STOPPED = 2

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    experiment = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Inferensi sentimen RM 1 (IndoBERT SmSA, 3 kelas)")
    parser.add_argument(
        "--input",
        default=str(experiment.cleaned_dir / "reviews_clean.parquet"),
        help="Tabel ulasan bersih (.parquet atau .csv)",
    )
    parser.add_argument("--text-col", default="text_light")
    parser.add_argument(
        "--output",
        default=str(experiment.sentiment_dir / "sentimen_ulasan.csv"),
        help="CSV keluaran; tidak boleh sudah ada",
    )
    parser.add_argument("--repo-id", default=SentimentModelConfig().repo_id)
    parser.add_argument("--revision", default=None, help="Hash commit 40 karakter (atau env INDOBERT_SENTIMENT_REVISION)")
    parser.add_argument("--model-dir", default=None, help="Folder model lokal (melewati Hugging Face)")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Hanya N baris pertama (uji coba)")
    parser.add_argument("--short-words", type=int, default=2, help="Ambang 'sangat pendek' (jumlah kata) untuk ringkasan")
    parser.add_argument("--sample-checks", type=int, default=16, help="Jumlah teks untuk pemeriksaan model")
    parser.add_argument("--save-logits", action="store_true", help="Simpan logit_negatif/netral/positif")
    parser.add_argument("--save-diagnostics", action="store_true", help="Simpan kolom n_token dan terpotong")
    parser.add_argument("--no-argmax", action="store_true", help="Jangan tambahkan kolom label_argmax")
    parser.add_argument("--no-tokenizer-strict", action="store_true", help="Peringatan (bukan error) bila model_max_length tokenizer != 128")
    parser.add_argument("--continue-on-failed-checks", action="store_true")
    return parser


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError(f"Format input tidak didukung: {path.suffix} (pakai .parquet atau .csv)")


def _sample_texts(texts: pd.Series, n: int) -> list[str]:
    valid = [str(t) for t in texts.tolist() if isinstance(t, str) and t.strip()]
    if len(valid) <= n:
        return valid
    positions = np.linspace(0, len(valid) - 1, n).astype(int)
    return [valid[i] for i in positions]


def run(args: argparse.Namespace, bundle: SentimentBundle | None = None) -> int:
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    summary_json = output_path.with_name(output_path.stem + "_ringkasan.json")
    summary_md = output_path.with_name(output_path.stem + "_ringkasan.md")

    existing = [str(p) for p in (output_path, summary_json, summary_md) if p.exists()]
    if existing:
        print("BERHENTI: file keluaran sudah ada dan tidak akan ditimpa:\n  " + "\n  ".join(existing))
        print("Pilih --output lain atau pindahkan file lama (bukan dihapus) lebih dulu.")
        return EXIT_STOPPED
    if not input_path.is_file():
        print(f"BERHENTI: input tidak ditemukan: {input_path}")
        return EXIT_STOPPED

    started = time.time()
    df = _read_table(input_path)
    if args.text_col not in df.columns:
        print(f"BERHENTI: kolom teks {args.text_col!r} tidak ada. Kolom tersedia: {list(df.columns)}")
        return EXIT_STOPPED
    conflict = [c for c in PROB_COLUMNS if c in df.columns]
    if conflict:
        print(f"BERHENTI: tabel input sudah punya kolom {conflict}; kemungkinan sudah pernah diinferensi.")
        return EXIT_STOPPED
    if args.limit:
        df = df.head(args.limit).copy()
    df = df.reset_index(drop=True)
    print(f"Input: {input_path} ({len(df):,} baris)")

    if bundle is None:
        config = SentimentModelConfig(
            repo_id=args.repo_id,
            cache_dir=args.cache_dir,
            model_dir=args.model_dir,
            device=args.device,
            batch_size=args.batch_size,
            strict_tokenizer_check=not args.no_tokenizer_strict,
        )
        if args.revision:
            config.revision = args.revision
        bundle = load_sentiment_model(config)
    bundle.batch_size = args.batch_size
    print(f"Model siap di device: {bundle.device}")

    # --- pemeriksaan model sebelum run penuh ---
    checks = run_model_checks(bundle, _sample_texts(df[args.text_col], args.sample_checks))
    for check in checks:
        print(f"  [{'OK' if check['lulus'] else 'GAGAL'}] {check['nama']} -- {check['detail']}")
    if not all(c["lulus"] for c in checks) and not args.continue_on_failed_checks:
        print("BERHENTI: ada pemeriksaan model yang gagal (lihat di atas).")
        return EXIT_CHECK_FAILED

    # --- inferensi ---
    probs = predict_sentiment_probs(
        df[args.text_col],
        bundle=bundle,
        return_diagnostics=True,
        return_logits=args.save_logits,
    )
    summary = summarize_predictions(
        df[args.text_col], probs, max_length=bundle.max_length, short_words=args.short_words
    )

    keep = list(PROB_COLUMNS)
    if args.save_logits:
        keep += [c for c in probs.columns if c.startswith("logit_")]
    if args.save_diagnostics:
        keep += ["n_token", "terpotong"]
    out = pd.concat([df, probs[keep].reset_index(drop=True)], axis=1)
    if not args.no_argmax:
        out["label_argmax"] = argmax_label(probs).reset_index(drop=True)

    # --- validasi sebelum menulis ---
    assert len(out) == len(df), "jumlah baris keluaran != masukan"
    assert summary["n_nan_pada_diinferensi"] == 0, "ada NaN pada baris yang diinferensi"
    assert summary["deviasi_maks_jumlah_prob"] <= 1e-6, "jumlah probabilitas != 1"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")

    meta = dict(bundle.meta)
    meta.update(
        {
            "input": str(input_path),
            "output": str(output_path),
            "jumlah_baris_input": len(df),
            "waktu_mulai": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
            "durasi_detik": round(time.time() - started, 1),
            "python": platform.python_version(),
        }
    )
    summary_json.write_text(
        json.dumps({"meta": meta, "ringkasan": summary, "pemeriksaan": checks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(format_summary_md(summary, meta, checks), encoding="utf-8")

    print("\n" + "=" * 64)
    print("RINGKASAN SANITY CHECK")
    print("=" * 64)
    print(f"Baris total / diinferensi / kosong : {summary['n_total']:,} / {summary['n_diinferensi']:,} / {summary['n_kosong']:,}")
    print(f"Sangat pendek (<= {args.short_words} kata)       : {summary['n_sangat_pendek']:,}")
    print(f"Terpotong (> {bundle.max_length} token)          : {summary['n_terpotong']:,} ({summary['persen_terpotong']}%)")
    print(f"Rata-rata p_netral                 : {summary['rata2_p_netral']}")
    for name in LABEL_NAMES:
        item = summary["distribusi_argmax"][name]
        print(f"  argmax {name:<8}: {item['n']:,} ({item['persen']}%)")
    print(f"Keluaran: {output_path}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
