import os
from unittest.mock import patch

import pandas as pd
import pytest

from src.cleaning.build import (
    add_flags, audit_timestamp_anchors, build_all, load_scrape_dates,
    parse_timestamp, resolve_timestamp_anchor,
)
from src.cleaning.config import CleaningConfig
from src.cleaning.file_map import load_file_map
from .conftest import make_config


FORMATS = CleaningConfig().timestamp_formats


@pytest.mark.parametrize(("raw", "expected"), [
    ("3 hari lalu", "2026-07-26"), ("3 minggu lalu", "2026-07-08"),
    ("2 bulan lalu", "2026-05-29"), ("12 jam lalu", "2026-07-29"),
    ("kemarin", "2026-07-28"), ("hari ini", "2026-07-29"),
    ("45 menit lalu", "2026-07-29"), ("2 tahun lalu", "2024-07-29"),
    ("SEBULAN LALU", "2026-06-29"), ("Seminggu Lalu", "2026-07-22"),
    ("3 HARI YANG LALU", "2026-07-26"),
])
def test_relative_fixed_anchor(raw, expected):
    assert parse_timestamp(raw, FORMATS, pd.Timestamp("2026-07-29")) == (
        pd.Timestamp(expected), False, "relatif_dikonversi",
    )


@pytest.mark.parametrize(("raw", "anchor", "expected"), [
    ("1 bulan lalu", "2026-03-31", "2026-02-28"),
    ("1 tahun lalu", "2024-02-29", "2023-02-28"),
    ("1 hari lalu", "2026-07-29 18:00", "2026-07-28"),
])
def test_calendar_offsets(raw, anchor, expected):
    assert parse_timestamp(raw, FORMATS, pd.Timestamp(anchor))[0] == pd.Timestamp(expected)


@pytest.mark.parametrize("anchor", [None, pd.NaT])
def test_no_anchor(anchor):
    value, time, status = parse_timestamp("3 hari lalu", FORMATS, anchor)
    assert pd.isna(value) and not time and status == "relatif_tanpa_acuan"


@pytest.mark.parametrize("raw", ["beberapa hari lalu", "3 hari lalu mungkin", "baru saja", "3 detik lalu", "07/03/2026", ""])
def test_unknown_not_guessed(raw):
    value, time, status = parse_timestamp(raw, FORMATS, pd.Timestamp("2026-07-29"))
    assert pd.isna(value) and not time and status == "tak_terparsing"


@pytest.mark.parametrize(("raw", "expected", "has_time"), [
    ("2026-07-29", "2026-07-29", False),
    ("2026-07-29 12:34 | Variasi: Biru", "2026-07-29 12:34", True),
    ("2026-07-29 12:34:56", "2026-07-29 12:34:56", True),
])
def test_absolute_regression(raw, expected, has_time):
    for anchor in (None, pd.Timestamp("2027-01-01")):
        assert parse_timestamp(raw, FORMATS, anchor) == (pd.Timestamp(expected), has_time, "ok")


def test_anchor_priority_anomaly_missing_and_timezone(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("timestamp\n2026-07-29\n")
    # UTC malam -> tanggal berikutnya di Jakarta; tidak bergantung TZ host.
    ns = pd.Timestamp("2026-07-28 18:00", tz="UTC").value
    os.utime(p, ns=(ns, ns))
    cfg = CleaningConfig(raw_dir=tmp_path, scrape_dates_path=tmp_path / "dates.csv")
    safe = resolve_timestamp_anchor(p, pd.Timestamp("2026-07-29"), cfg, {})
    assert not safe["mtime_anomaly"]
    assert safe["timestamp_anchor"] == pd.Timestamp("2026-07-29")
    bad = resolve_timestamp_anchor(p, pd.Timestamp("2026-07-30"), cfg, {})
    assert bad["mtime_anomaly"] and bad["timestamp_anchor_source"] == "tidak_ada"
    assert pd.isna(bad["timestamp_anchor"])
    override = resolve_timestamp_anchor(p, pd.Timestamp("2026-07-30"), cfg, {"a.csv": pd.Timestamp("2026-08-01")})
    assert override["mtime_anomaly"] and override["timestamp_anchor_source"] == "scrape_dates.csv"
    assert override["timestamp_anchor"] == pd.Timestamp("2026-08-01")
    missing = resolve_timestamp_anchor(tmp_path / "missing.csv", pd.NaT, cfg, {})
    assert missing["timestamp_anchor_source"] == "tidak_ada" and pd.isna(missing["timestamp_anchor"])
    cfg.scrape_dates_path.write_text("file,tanggal_scrape\na.csv,2026-08-02\n")
    assert load_scrape_dates(cfg) == {"a.csv": pd.Timestamp("2026-08-02")}
    cfg.scrape_dates_path.write_text("file,tanggal_scrape\na.csv,today\n")
    with pytest.raises(ValueError, match="invalid"):
        load_scrape_dates(cfg)


def test_audit_scans_all_cells_and_rejects_anomaly(raw_dir, tmp_path):
    path = raw_dir / "Toko A_HP Satu.csv"
    ns = pd.Timestamp("2026-01-01", tz="Asia/Jakarta").value
    os.utime(path, ns=(ns, ns))
    cfg = make_config(raw_dir, tmp_path / "out", scrape_dates_path=None)
    audit = audit_timestamp_anchors(cfg).set_index("file")
    assert audit.loc[path.name, "latest_absolute"] == pd.Timestamp("2026-03-07 10:00")
    assert audit.loc[path.name, "mtime_anomaly"]
    result = build_all(cfg, load_file_map(cfg), log=lambda *_: None)
    row = result.reviews.query('source_file == @path.name and row_order == 4').iloc[0]
    assert row.timestamp_status == "relatif_tanpa_acuan"
    assert row.timestamp_relative_raw == "2 minggu lalu"
    assert pd.isna(row.timestamp) and not row.timestamp_is_estimated


def test_build_per_file_reproducible_and_stable_order(raw_dir, tmp_path):
    # Dua jam sistem berbeda, metadata input sama: seluruh tabel sama.
    cfg = make_config(raw_dir, tmp_path / "out", scrape_dates_path=None)
    for i, name in enumerate(sorted(raw_dir.glob("*.csv"))):
        ns = pd.Timestamp("2026-08-01", tz="Asia/Jakarta").value + i * 86400 * 10**9
        os.utime(name, ns=(ns, ns))
    file_map = load_file_map(cfg)
    with patch("time.time", return_value=1_800_000_000):
        a = build_all(cfg, file_map, log=lambda *_: None)
    with patch("time.time", return_value=2_000_000_000):
        b = build_all(cfg, file_map, log=lambda *_: None)
    pd.testing.assert_frame_equal(a.reviews, b.reviews)
    pd.testing.assert_frame_equal(a.anchor_audit, b.anchor_audit)
    relative = a.reviews[a.reviews.timestamp_is_estimated]
    assert relative.timestamp_anchor.nunique() == 2
    assert relative.timestamp_anchor_source.eq("mtime").all()
    assert not relative.timestamp_has_time.any()
    assert not relative.is_technical_duplicate.any()
    # Semua timestamp kembar: source_file dan row_order menentukan urutan.
    tied = a.reviews.assign(timestamp=pd.Timestamp("2026-08-01"), product_id="P")
    ordered = add_flags(tied.sample(frac=1, random_state=42), cfg)
    expected = tied.sort_values(["source_file", "row_order"])
    assert ordered.review_id.tolist() == expected.review_id.tolist()
    assert ordered.seq_in_product.tolist() == list(range(1, len(ordered) + 1))


def test_estimates_never_technical_duplicates(raw_dir, tmp_path):
    cfg = make_config(raw_dir, tmp_path / "out", technical_dup_require_time=False)
    result = build_all(cfg, load_file_map(cfg), log=lambda *_: None)
    row = result.reviews[result.reviews.timestamp_is_estimated].iloc[[0]]
    duplicated = pd.concat([row, row.assign(review_id="new", row_order=999)])
    out = add_flags(duplicated, cfg)
    assert not out.is_technical_duplicate.any()
    assert out.include_for_inference.all()
