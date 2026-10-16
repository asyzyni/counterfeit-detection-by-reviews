"""Audit berkas baru, aturan pengecualian salinan, dan salinan pemetaan v4 (data sintetis)."""

from __future__ import annotations

import shutil
from argparse import Namespace

import pandas as pd
import pytest

from src.cleaning import run as cleaning_run
from src.cleaning.audit_data_baru import (
    build_parser as audit_parser, is_salinan, kunci_ulasan, normalisasi_nama, putuskan, run as audit_run,
    stem_berkas, tumpang_tindih, turunkan_pemetaan,
)
from src.cleaning.pemetaan_v4 import (
    STATUS_DIRI, PemetaanError, bangun_pemetaan, build_parser as map_parser, run as map_run,
)

from .conftest import MAPPING_NAME, make_config

READ = dict(dtype=str, keep_default_na=False, encoding="utf-8-sig")


# ============================================================
# NAMA
# ============================================================

@pytest.mark.parametrize("name, expected", [
    ("A_B copy.csv", True), ("A_B  copy.csv", True), ("A_B (2).csv", True), ("A_B copy 2.csv", True),
    ("A_B.csv", False), ("A_B .csv", False), ("A_B.csv.csv", False), ("A_Copycat.csv", False),
])
def test_is_salinan(name, expected):
    assert is_salinan(name) is expected


def test_normalisasi_nama_menyamakan_varian():
    names = ["JIU_PHONCELL_Infinix Smart 20 8:256gb.csv", "JIU_PHONCEL_Infinix Smart 20 8:256gb.csv",
             "jiu_phoncell_infinix smart 20 8:256gb copy.csv", "JIU_PHONCELL_Infinix  Smart 20 8:256gb (2).csv"]
    assert len({normalisasi_nama(n) for n in names}) == 1
    assert normalisasi_nama("A_HP Satu.csv") != normalisasi_nama("A_HP Dua.csv")


def test_turunkan_pemetaan_underscore_terakhir_dan_catatan():
    row = turunkan_pemetaan("Riawan_shop88_Infinix Smart 20 8:128GB.csv.csv")
    assert (row["nama_toko"], row["nama_produk"]) == ("Riawan_shop88", "Infinix Smart 20 8:128GB")
    assert row["product_key"] == "riawan_shop88|infinix smart 20 8:128gb"
    assert row["sumber_nama"] == "underscore terakhir"
    assert ".csv berulang" in row["catatan"]
    quoted = turunkan_pemetaan('Leon_cell99_itel A200 6.65" .csv')
    assert quoted["nama_produk"] == 'itel A200 6.65"' and "kutip" in quoted["catatan"]
    spaced = turunkan_pemetaan("TAC Home_ Xiaomi Redmi 13C.csv")
    assert spaced["product_key"] == "tac home|xiaomi redmi 13c"
    copy = turunkan_pemetaan("Toko_HP copy.csv")
    assert copy["nama_produk"] == "HP copy" and "salinan" in copy["catatan"]
    assert turunkan_pemetaan("TanpaUnderscore.csv")["status"] == "perlu_cek"
    assert stem_berkas("x.CSV.csv") == "x"


# ============================================================
# TUMPANG TINDIH & KEPUTUSAN
# ============================================================

def test_tumpang_tindih_multiset_dan_hash():
    keys = {"a.csv": ["x", "x", "y", "z"], "b.csv": ["x", "y"], "c.csv": [], "d.csv": []}
    pairs = tumpang_tindih(keys, {"a.csv": "1", "b.csv": "2", "c.csv": "9", "d.csv": "9"})
    ab = pairs[(pairs.file_a == "a.csv") & (pairs.file_b == "b.csv")].iloc[0]
    assert ab.n_ulasan_sama == 2 and ab.pct_a_dalam_b == 50.0 and ab.pct_b_dalam_a == 100.0
    cd = pairs[(pairs.file_a == "c.csv") & (pairs.file_b == "d.csv")].iloc[0]
    assert cd.hash_identik and cd.pct_a_dalam_b == 100.0


def test_kunci_ulasan_nfkc_spasi_kapital():
    assert kunci_ulasan("  Bagus  BANGET ") == kunci_ulasan("bagus banget")


def _pairs(rows):
    return pd.DataFrame(rows, columns=["file_a", "file_b", "n_ulasan_a", "n_ulasan_b", "n_ulasan_sama",
                                       "pct_a_dalam_b", "pct_b_dalam_a", "hash_identik"])


def test_putuskan_hanya_satu_dari_pasangan_identik_dan_wajib_nama_salinan():
    pairs = _pairs([
        ("T_A (2).csv", "T_A.csv", 10, 10, 10, 100.0, 100.0, True),   # salinan identik
        ("T_B.csv", "T_C.csv", 10, 10, 10, 100.0, 100.0, False),      # tanpa nama salinan
        ("T_D copy.csv", "T_D.csv", 10, 20, 7, 70.0, 35.0, False),    # < 80%
        ("T_E copy.csv", "T_E.csv", 20, 10, 10, 50.0, 100.0, False),  # salinan lebih lengkap
    ])
    files = ["T_A (2).csv", "T_A.csv", "T_B.csv", "T_C.csv", "T_D copy.csv", "T_D.csv",
             "T_E copy.csv", "T_E.csv"]
    got = putuskan(files, pairs).set_index("file")["keputusan"]
    assert got["T_A (2).csv"] == "kecualikan" and got["T_A.csv"] == "pertahankan"
    assert got["T_B.csv"] == got["T_C.csv"] == "pertahankan"
    assert got["T_D copy.csv"] == got["T_D.csv"] == "pertahankan"
    assert got["T_E.csv"] == "kecualikan" and got["T_E copy.csv"] == "pertahankan"


def test_putuskan_rantai_salinan_tidak_mengecualikan_semua():
    pairs = _pairs([
        ("X_P (2).csv", "X_P copy.csv", 10, 10, 10, 100.0, 100.0, True),
        ("X_P (2).csv", "X_P.csv", 10, 10, 9, 90.0, 90.0, False),
        ("X_P copy.csv", "X_P.csv", 10, 10, 9, 90.0, 90.0, False),
    ])
    got = putuskan(["X_P (2).csv", "X_P copy.csv", "X_P.csv"], pairs).set_index("file")["keputusan"]
    assert (got == "pertahankan").sum() >= 1 and got["X_P.csv"] == "pertahankan"


# ============================================================
# SALINAN PEMETAAN
# ============================================================

def test_bangun_pemetaan_menolak_kunci_tidak_1_ke_1():
    old = pd.DataFrame([{"file": "A_HP.csv", "nama_toko": "A", "nama_produk": "HP", "product_key": "a|hp",
                         "status": "ok", "sumber_nama": "nama file", "catatan": "", "ukuran_byte": "1"}])
    exclusions = pd.DataFrame(columns=["file", "alasan", "berkas_pasangan", "pct_tumpang_tindih", "keputusan"])
    with pytest.raises(PemetaanError, match="1-ke-1"):
        bangun_pemetaan(old, {"A_HP.csv": 1, "a_hp .csv": 2}, exclusions, MAPPING_NAME)


def test_alur_audit_pemetaan_cleaning_end_to_end(raw_dir, tmp_path):
    shutil.copy(raw_dir / "Toko C_HP Tiga.csv", raw_dir / "Toko C_HP Tiga copy.csv")
    pd.read_csv(raw_dir / "Toko A_HP Satu.csv", dtype=str, keep_default_na=False).to_csv(
        raw_dir / "Toko D_HP Lima.csv", index=False)
    old_mapping = raw_dir / MAPPING_NAME
    manifest = tmp_path / "file_manifest.csv"
    pd.read_csv(old_mapping, **READ)[["file"]].to_csv(manifest, index=False)
    before = {p.name: p.read_bytes() for p in raw_dir.iterdir()}

    out = tmp_path / "audit"
    args = audit_parser().parse_args([
        "--raw-dir", str(raw_dir), "--pemetaan-lama", str(old_mapping), "--manifest-lama", str(manifest),
        "--out-md", str(out / "a.md"), "--out-pasangan", str(out / "p.csv"),
        "--out-pengecualian", str(out / "e.csv")])
    assert audit_run(args) == 0
    assert audit_run(args) == 2  # tidak menimpa
    decisions = pd.read_csv(out / "e.csv", **READ).set_index("file")["keputusan"]
    assert decisions["Toko C_HP Tiga copy.csv"] == "kecualikan"
    assert decisions["Toko C_HP Tiga.csv"] == "pertahankan"
    assert "Toko D_HP Lima.csv" in decisions.index

    mapping_v4 = tmp_path / "pemetaan_v4.csv"
    margs = map_parser().parse_args(["--raw-dir", str(raw_dir), "--pemetaan-lama", str(old_mapping),
                                     "--pengecualian", str(out / "e.csv"), "--output", str(mapping_v4)])
    assert map_run(margs) == 0
    assert map_run(margs) == 2
    v4 = pd.read_csv(mapping_v4, **READ).set_index("file")
    assert v4.loc[MAPPING_NAME, "status"] == STATUS_DIRI
    assert v4.loc["Toko C_HP Tiga copy.csv", "status"].startswith("dikecualikan:")
    assert v4.loc["Toko D_HP Lima.csv", "product_key"] == "toko d|hp lima"
    n_ok = int((v4["status"] == "ok").sum())

    # Cleaning lama, tanpa perubahan kode: path absolut salinan pemetaan lewat --mapping-filename.
    config = make_config(raw_dir, tmp_path / "cleaned_v4", mapping_filename=str(mapping_v4.resolve()),
                         expected_mapping_rows=len(v4), expected_ok_files=n_ok)
    assert cleaning_run.run(config) == cleaning_run.EXIT_OK
    reviews = pd.read_parquet(config.reviews_path)
    assert "Toko C_HP Tiga copy.csv" not in set(reviews["source_file"])
    assert "Toko D_HP Lima.csv" in set(reviews["source_file"])
    assert MAPPING_NAME not in set(reviews["source_file"])
    assert {p.name: p.read_bytes() for p in raw_dir.iterdir()} == before  # data mentah tidak berubah


def test_namespace_default_tidak_menunjuk_output_lama():
    args = audit_parser().parse_args([])
    assert isinstance(args, Namespace)
    assert "v4" in str(args.out_pengecualian) and "v4" in str(map_parser().parse_args([]).output)
