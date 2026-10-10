"""Tes pipeline cleaning dengan CSV mini sintetis (tiga keluarga skema)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.cleaning.build import build_all, parse_timestamp
from src.cleaning.export import export_tabel_contoh
from src.cleaning.file_map import MappingError, load_file_map
from src.cleaning.run import EXIT_OK, EXIT_STOPPED, run
from src.cleaning.schema import classify_value
from src.cleaning.text_clean import clean_light

from .conftest import make_config, write_raw_dir


def _build(raw_dir: Path, tmp_path: Path, **overrides):
    config = make_config(raw_dir, tmp_path / "out", **overrides)
    file_map = load_file_map(config)
    return build_all(config, file_map, log=lambda *_: None), file_map, config


def _roles(result, file: str) -> dict[str, str]:
    cm = result.column_map[result.column_map["file"] == file]
    return dict(zip(cm["column"], cm["role"]))


def _row(result, file: str, row_order: int) -> pd.Series:
    df = result.reviews
    match = df[(df["source_file"] == file) & (df["row_order"] == row_order)]
    assert len(match) == 1, f"baris {file}#{row_order} tidak ada di tabel"
    return match.iloc[0]


# ============================================================
# PEMETAAN
# ============================================================

def test_mapping_skips_self_row_and_non_ok(raw_dir: Path, tmp_path: Path) -> None:
    config = make_config(raw_dir, tmp_path / "out")
    file_map = load_file_map(config)

    assert len(file_map.ok) == 3
    assert set(file_map.excluded["file"]) == {"TOKO SAJA.csv", "pemetaan_file_toko_produk.csv"}


def test_mapping_count_mismatch_stops(raw_dir: Path, tmp_path: Path) -> None:
    config = make_config(raw_dir, tmp_path / "out", expected_ok_files=99)

    with pytest.raises(MappingError):
        load_file_map(config)


# ============================================================
# KELUARGA A: web-scraper Shopee
# ============================================================

def test_family_a_roles(raw_dir: Path, tmp_path: Path) -> None:
    result, _, _ = _build(raw_dir, tmp_path)
    roles = _roles(result, "Toko A_HP Satu.csv")

    # kolom `data` berisi ulasan (bukan metadata seperti di kode lama)
    assert roles["data"] == "review_text"
    assert roles["timestamp"] == "timestamp"
    assert roles["data3"] == "metadata"          # nama pengguna tersamar
    assert roles["data4"] == "review_text"       # campuran label + teks
    assert roles["web_scraper_start_url"] == "url_media"


def test_family_a_rows(raw_dir: Path, tmp_path: Path) -> None:
    result, _, _ = _build(raw_dir, tmp_path)
    file = "Toko A_HP Satu.csv"

    first = _row(result, file, 0)
    assert first["review"] == "barang bagus sesuai pesanan"
    assert first["aspect_tags"] == "Kualitas:"
    assert first["timestamp"] == pd.Timestamp("2026-03-07 10:00")
    assert bool(first["timestamp_has_time"])

    html_row = _row(result, file, 3)
    assert html_row["review"] == "mantap <b>original</b> https://contoh.id/x | baterai 5000mAh awet"
    assert html_row["text_light"] == "mantap original baterai 5000mAh awet"
    assert html_row["fragment_cols"] == "data | data4"
    assert not bool(html_row["timestamp_has_time"])

    repeated = _row(result, file, 4)
    assert repeated["text_light"] == "baguss banget 👍👍"
    assert pd.isna(repeated["timestamp"])
    assert repeated["timestamp_status"] == "relatif"

    manifest = result.manifest.set_index("file").loc[file]
    assert manifest["n_baris_sumber"] == 8
    assert manifest["n_baris_kosong"] == 2
    assert manifest["n_baris_keluar"] == 6


def test_duplicates(raw_dir: Path, tmp_path: Path) -> None:
    result, _, _ = _build(raw_dir, tmp_path)
    file = "Toko A_HP Satu.csv"

    original = _row(result, file, 1)
    scraped_twice = _row(result, file, 5)
    later = _row(result, file, 7)

    # teks + timestamp berjam sama -> duplikat teknis (yang kedua saja)
    assert not original["is_technical_duplicate"]
    assert scraped_twice["is_technical_duplicate"]
    assert not scraped_twice["include_for_inference"]

    # teks sama, timestamp beda -> hanya ditandai, tetap ikut
    assert original["is_text_duplicate"] and later["is_text_duplicate"]
    assert later["include_for_inference"]
    assert original["dup_group_size"] == 3


# ============================================================
# KELUARGA B: Lazada
# ============================================================

def test_family_b_roles_and_rows(raw_dir: Path, tmp_path: Path) -> None:
    result, _, _ = _build(raw_dir, tmp_path)
    file = "Toko B_HP Dua.csv"
    roles = _roles(result, file)

    assert roles["review-attribute"] == "aspect_tag"
    assert roles["img-more-text"] == "metadata"
    assert roles["time"] == "timestamp"
    assert roles["item-content-main-content-reviews-item (3)"] == "review_text"

    first = _row(result, file, 0)
    assert first["review"] == "hp nya bagus sekali | murah"
    assert first["aspect_tags"] == "Harga:"
    assert "Harga" not in first["review"]

    weird = _row(result, file, 3)
    assert weird["timestamp_status"] == "tak_terparsing"
    assert pd.isna(weird["timestamp"])

    relative = _row(result, file, 1)
    assert relative["timestamp_status"] == "relatif"


def test_aspect_vocab_moves_fragment(raw_dir: Path, tmp_path: Path) -> None:
    vocab = tmp_path / "vocab.csv"
    pd.DataFrame({"fragmen": ["harga", "bagus"], "setuju": ["ya", ""]}).to_csv(vocab, index=False)

    result, _, _ = _build(raw_dir, tmp_path, aspect_tag_vocab_path=vocab)
    file = "Toko B_HP Dua.csv"

    # baris 4 hanya berisi "harga" -> pindah ke aspect_tags, baris jadi kosong
    reviews = result.reviews
    assert reviews[(reviews["source_file"] == file) & (reviews["row_order"] == 4)].empty

    manifest = result.manifest.set_index("file").loc[file]
    assert manifest["n_kosong_hanya_fragmen_non_teks"] == 1
    assert manifest["cek_seimbang"]

    # "bagus" tidak disetujui -> tetap di review
    assert _row(result, file, 0)["review"] == "hp nya bagus sekali | murah"


# ============================================================
# KELUARGA C: kolom acak Shopee
# ============================================================

def test_family_c_roles_and_rows(raw_dir: Path, tmp_path: Path) -> None:
    result, _, _ = _build(raw_dir, tmp_path)
    file = "Toko C_HP Tiga.csv"
    roles = _roles(result, file)

    assert roles["XYk98l"] == "timestamp"
    assert roles["meQyXP"] == "review_text"
    assert roles["K5v3lN (3)"] == "aspect_tag"
    assert roles["HcSdrS src (2)"] == "url_media"
    assert roles["rFzVcr"] == "metadata"
    assert roles["QSiE2A"] == "seller_reply"

    first = _row(result, file, 0)
    assert first["review"] == "aman | ori | Pengiriman cepat dan aman"
    assert "terima kasih" not in first["review"].lower()
    assert first["aspect_tags"] == "Produk:"

    short = _row(result, file, 5)
    assert short["is_short"] and not short["include_for_inference"]


# ============================================================
# INVARIAN UMUM
# ============================================================

def test_table_invariants(raw_dir: Path, tmp_path: Path) -> None:
    result, _, _ = _build(raw_dir, tmp_path)
    reviews = result.reviews

    assert reviews["review_id"].is_unique
    assert reviews["review_id"].notna().all()
    assert result.manifest["cek_seimbang"].all()

    included = reviews[reviews["include_for_inference"]]
    assert (included["text_light"].str.strip() != "").all()

    for _, group in reviews.groupby("product_id"):
        ordered = group.sort_values("seq_in_product")
        assert ordered["seq_in_product"].tolist() == list(range(1, len(group) + 1))
        # NaT di akhir
        stamps = ordered["timestamp"]
        assert not (stamps.isna().cummax() & stamps.notna()).any()


def test_duplicate_file_groups(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_raw_dir(raw, extra_duplicate=True)

    result, _, _ = _build(raw, tmp_path)
    assert set(result.duplicate_files["file"]) == {"Toko C_HP Empat.csv", "Toko C_HP Tiga.csv"}
    assert set(result.reviews.loc[result.reviews["file_dup_group"] != "", "file_dup_group"]) == {"FD01"}

    result_ex, _, _ = _build(raw, tmp_path, exclude_duplicate_file_groups=True)
    assert "Toko C_HP Tiga.csv" not in set(result_ex.reviews["source_file"])
    assert "Toko C_HP Tiga.csv" in set(result_ex.excluded_files["file"])


# ============================================================
# FUNGSI KECIL
# ============================================================

@pytest.mark.parametrize(
    ("raw", "expected_status", "has_time"),
    [
        ("2026-03-07", "ok", False),
        ("2025-09-08 11:53", "ok", True),
        ("2025-09-08 11:53 | Variasi: Hitam", "ok", True),
        ("2 minggu lalu", "relatif", False),
        ("kemarin", "relatif", False),
        ("07/03/2026", "tak_terparsing", False),
        ("", "kosong", False),
    ],
)
def test_parse_timestamp(raw: str, expected_status: str, has_time: bool) -> None:
    formats = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")
    value, parsed_has_time, status = parse_timestamp(raw, formats)

    assert status == expected_status
    assert parsed_has_time == has_time
    assert pd.isna(value) == (status != "ok")


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        ("Kualitas:", "label_aspek"),
        ("🔋Daya Tahan Baterai:", "label_aspek"),
        ("0:12", "angka"),
        ("+3", "angka"),
        ("z*****r", "nama_tersamar"),
        ("https://down-id.img.susercontent.com/x", "url"),
        ("👍👍", "teks"),
        ("Respon Penjual:", "penanda_penjual"),
        ("bagus banget", "teks"),
    ],
)
def test_classify_value(value: str, kind: str) -> None:
    assert classify_value(value) == kind


def test_clean_light_is_minimal() -> None:
    # tidak lowercase, tidak ubah slang/ejaan, emoji & angka utuh
    assert clean_light("Brg ORI gan,  mantulll 👍\n5000mAh") == "Brg ORI gan, mantull 👍 5000mAh"
    # huruf matematis -> huruf biasa (NFKC)
    assert clean_light("𝙗𝙖𝙜𝙪𝙨") == "bagus"
    # ZWJ dalam emoji dipertahankan
    assert clean_light("keluarga 👨‍👩‍👧") == "keluarga 👨‍👩‍👧"


def test_export_tabel_contoh_anonim(raw_dir: Path, tmp_path: Path) -> None:
    result, _, _ = _build(raw_dir, tmp_path)

    table = export_tabel_contoh(result.reviews, n=3, anonimkan=True)
    assert list(table.columns) == ["Nama Toko", "Nama Produk", "Review", "Timestamp"]
    assert table["Nama Toko"].str.match(r"^Toko [A-Z]+$").all()
    assert table["Nama Produk"].str.match(r"^Produk \d+$").all()

    # data kerja tetap memakai nama asli
    assert set(result.reviews["nama_toko"]) == {"Toko A", "Toko B", "Toko C"}


# ============================================================
# CLI: folder output & determinisme
# ============================================================

def test_run_refuses_existing_output(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "hasil_lama.parquet").write_bytes(b"x")

    assert run(make_config(raw_dir, out)) == EXIT_STOPPED


def test_run_accepts_empty_output_dir(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()

    assert run(make_config(raw_dir, out)) == EXIT_OK


def test_run_is_deterministic(raw_dir: Path, tmp_path: Path) -> None:
    out1, out2 = tmp_path / "run1", tmp_path / "run2"

    assert run(make_config(raw_dir, out1)) == EXIT_OK
    assert run(make_config(raw_dir, out2)) == EXIT_OK

    hashes1 = pd.read_csv(out1 / "reports" / "output_hashes.csv")
    hashes2 = pd.read_csv(out2 / "reports" / "output_hashes.csv")

    assert len(hashes1) > 5
    pd.testing.assert_frame_equal(hashes1, hashes2)
