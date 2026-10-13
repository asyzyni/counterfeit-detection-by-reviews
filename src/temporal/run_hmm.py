"""Jalankan RM 2 (HMM) dengan satu perintah.

Contoh:
    python -m src.temporal.run_hmm --input "outputs/sentiment/sentimen_ulasan_v2.csv" \\
        --split-ids "outputs/labels/split_ids.csv"

Tanpa --split-ids Anda wajib menambahkan --temporary-split (acak, TEMPORARY, bukan untuk dilaporkan).
File keluaran yang sudah ada tidak ditimpa kecuali --overwrite.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from ..config import ExperimentConfig
from .config import BACKENDS, REPRESENTATIONS, HMMConfig
from .pipeline import run_rm2
from .sequences import build_sequences
from .splits import load_split_ids, temporary_product_split


def build_parser() -> argparse.ArgumentParser:
    exp = ExperimentConfig()
    p = argparse.ArgumentParser(description="RM 2: HMM pola temporal sentimen per produk")
    p.add_argument("--input", required=True, help="Tabel keluaran RM 1 (.csv atau .parquet)")
    p.add_argument("--split-ids", default=None, help="CSV/parquet: product_id, split (train/test)")
    p.add_argument("--temporary-split", action="store_true", help="Split acak SEMENTARA (bukan untuk dilaporkan)")
    p.add_argument("--representation", default="prob", choices=REPRESENTATIONS)
    p.add_argument("--n-states", type=int, default=exp.hmm_components)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--n-iter", type=int, default=exp.hmm_n_iter)
    p.add_argument("--min-reviews", type=int, default=exp.min_reviews_per_product)
    p.add_argument("--seed", type=int, default=exp.random_state)
    p.add_argument("--backend", default="auto", choices=BACKENDS)
    p.add_argument("--stability-seeds", type=int, nargs="*", default=[], help="Seed tambahan untuk uji stabilitas")
    p.add_argument("--features-out", default=str(exp.features_dir / "hmm_features.parquet"))
    p.add_argument("--model-out", default=str(exp.models_dir / "hmm_model.pkl"))
    p.add_argument("--meta-out", default=str(exp.models_dir / "hmm_meta.json"))
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    path = Path(args.input).expanduser()
    if not path.is_file():
        print(f'BERHENTI: input tidak ditemukan: "{path}"')
        return 2
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, encoding="utf-8-sig")

    cfg = HMMConfig(
        representation=args.representation, n_states=args.n_states, n_init=args.n_init, n_iter=args.n_iter,
        min_reviews=args.min_reviews, seed=args.seed, backend=args.backend, overwrite=args.overwrite,
        features_path=Path(args.features_out), model_path=Path(args.model_out), meta_path=Path(args.meta_out),
    )
    if args.split_ids:
        train_ids, test_ids = load_split_ids(args.split_ids, cfg.product_col)
        temporary = False
    elif args.temporary_split:
        seqs, _, _ = build_sequences(df, cfg)
        train_ids, test_ids = temporary_product_split(seqs.product_ids, seed=cfg.seed)
        temporary = True
    else:
        print("BERHENTI: berikan --split-ids (daftar Train/Test sungguhan) atau --temporary-split.")
        return 2

    try:
        result = run_rm2(df, cfg, train_ids, test_ids, split_is_temporary=temporary, stability_seeds=args.stability_seeds)
    except FileExistsError as exc:
        print(f"BERHENTI: {exc}")
        return 2

    meta = result.meta
    print(f"Backend: {meta['backend']} | restart terbaik: {meta['best_restart']} | konvergen: {meta['converged']}")
    print(f"Produk: {meta['n_products']} | dikeluarkan (T<{cfg.min_reviews}): {meta['sequences']['n_products_excluded_short']}")
    print(f"Rata-rata p_negatif per state (terurut): {meta['state_neg_mass_train']}")
    for w in meta["warnings"]:
        print("PERINGATAN:", w)
    if temporary:
        print("PERINGATAN: split TEMPORARY, hasil bukan untuk dilaporkan.")
    for name, p in result.paths.items():
        print(f'{name}: "{p}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
