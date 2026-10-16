from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .audit_data_baru import turunkan_pemetaan
from .config import CleaningConfig
from .file_map import MAPPING_COLUMNS

STATUS_DIRI = "dikecualikan: baris pemetaan merujuk file pemetaan itu sendiri"


class PemetaanError(ValueError):
    """Salinan pemetaan tidak memenuhi syarat."""


def bangun_pemetaan(old: pd.DataFrame, raw_files: dict[str, int], pengecualian: pd.DataFrame,
                    mapping_name: str) -> pd.DataFrame:
    """old: pemetaan lama (str). raw_files: nama CSV -> ukuran byte. Kembalikan salinan v4."""
    out = old[MAPPING_COLUMNS].copy()
    out.loc[out["file"] == mapping_name, "status"] = STATUS_DIRI

    new_rows = []
    for name in sorted(set(raw_files) - set(out["file"]) - {mapping_name}):
        row = turunkan_pemetaan(name)
        row["ukuran_byte"] = str(raw_files[name])
        new_rows.append({col: str(row.get(col, "")) for col in MAPPING_COLUMNS})
    out = pd.concat([out, pd.DataFrame(new_rows, columns=MAPPING_COLUMNS)], ignore_index=True)

    decisions = pengecualian.copy()
    decisions["keputusan"] = decisions["keputusan"].astype(str).str.strip().str.lower()
    unknown = sorted(set(decisions["keputusan"]) - {"kecualikan", "pertahankan"})
    if unknown:
        raise PemetaanError(f"Nilai keputusan tidak dikenal: {unknown}")
    excluded = decisions[decisions["keputusan"] == "kecualikan"]
    missing = sorted(set(excluded["file"]) - set(out["file"]))
    if missing:
        raise PemetaanError(f"Berkas di daftar pengecualian tidak ada di pemetaan: {missing}")
    for r in excluded.itertuples():
        mask = out["file"] == r.file
        out.loc[mask, "status"] = f"dikecualikan: {r.alasan}"
        detail = f"v4: dikecualikan (pasangan: {r.berkas_pasangan}; {r.pct_tumpang_tindih}%)"
        out.loc[mask, "catatan"] = out.loc[mask, "catatan"].map(lambda c: f"{c}; {detail}" if c else detail)

    periksa_pemetaan(out)
    return out


def periksa_pemetaan(mapping: pd.DataFrame) -> None:
    dup = mapping["file"][mapping["file"].duplicated()].tolist()
    if dup:
        raise PemetaanError(f"Berkas muncul lebih dari sekali: {dup}")
    ok = mapping[mapping["status"].str.strip() == "ok"]
    if (ok["product_key"].str.strip() == "").any():
        raise PemetaanError(f"product_key kosong: {ok.loc[ok['product_key'].str.strip() == '', 'file'].tolist()}")
    pair = ok["nama_toko"].str.strip() + "_" + ok["nama_produk"].str.strip()
    per_key = pair.groupby(ok["product_key"]).nunique()
    per_pair = ok["product_key"].groupby(pair).nunique()
    bad = per_key[per_key > 1].index.tolist() + per_pair[per_pair > 1].index.tolist()
    if bad:
        raise PemetaanError(f"product_key <-> nama_toko_nama_produk tidak 1-ke-1: {bad}")


def build_parser() -> argparse.ArgumentParser:
    defaults = CleaningConfig()
    parser = argparse.ArgumentParser(description="Salinan pemetaan v4 (lama + berkas baru + pengecualian)")
    parser.add_argument("--raw-dir", type=Path, default=defaults.raw_dir)
    parser.add_argument("--pemetaan-lama", type=Path, default=defaults.mapping_path)
    parser.add_argument("--pengecualian", type=Path, default=Path("references/file_dikecualikan_v4.csv"))
    parser.add_argument("--output", type=Path, default=Path("references/pemetaan_file_toko_produk_v4.csv"))
    return parser


def run(args: argparse.Namespace) -> int:
    if args.output.exists():
        print(f'BERHENTI: file sudah ada, tidak boleh ditimpa: "{args.output}"')
        return 2
    read = dict(dtype=str, keep_default_na=False, encoding="utf-8-sig")
    old = pd.read_csv(args.pemetaan_lama, **read)
    pengecualian = pd.read_csv(args.pengecualian, **read)
    raw_files = {p.name: p.stat().st_size for p in Path(args.raw_dir).iterdir()
                 if p.is_file() and p.suffix.lower() == ".csv"}
    try:
        mapping = bangun_pemetaan(old, raw_files, pengecualian, Path(args.pemetaan_lama).name)
    except PemetaanError as exc:
        print(f"BERHENTI: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", encoding="utf-8-sig", newline="") as f:
        mapping.to_csv(f, index=False, lineterminator="\n")
    n_ok = int((mapping["status"].str.strip() == "ok").sum())
    print(f'Ditulis: "{args.output}"')
    print(f"baris={len(mapping)} (lama={len(old)}, baru={len(mapping) - len(old)}), status ok={n_ok}, "
          f"dikecualikan={len(mapping) - n_ok}")
    print(f"Cleaning: --expected-mapping-rows {len(mapping)} --expected-ok-files {n_ok}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
