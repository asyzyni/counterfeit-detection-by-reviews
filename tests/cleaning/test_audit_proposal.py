"""Jejak dan audit dengan ulasan serta kamus sintetis, tanpa akses jaringan."""

from collections import Counter
import json

import pandas as pd
import pytest

from src.cleaning.audit_proposal import (
    build_parser, count_protected, make_top_replacements, run, sha256, trace_proposal, verify_tables,
)
from src.cleaning.config import PROTECTED_TERMS
from src.cleaning.proposal import _proposal_transformer, build_proposal_table


def synthetic_table(texts):
    return pd.DataFrame({
        "review_id": [f"r{number}" for number in range(len(texts))],
        "seq_in_product": list(range(len(texts))), "review": texts,
        "text_light": texts, "text_lex": ["tetap"] * len(texts),
        "include_for_inference": [True] * len(texts),
        "is_technical_duplicate": [False] * len(texts),
        "is_text_duplicate": [False] * len(texts),
    })


def test_trace_actual_replacements_case_once_overlap_multiword_protection():
    dictionary = {"bgs": "bagus", "bagus": "baik", "gak bgs": "kurang baik", "oke": "oke",
                  "ori": "SALAH", "bukan": "SALAH", "sesuai": "SALAH"}
    text = "BGS bgs bagus ORI bukan ORI sesuai deskripsi gak\tbgs xbgs oke OKE"
    events = []
    with_trace = _proposal_transformer(dictionary, PROTECTED_TERMS, lambda *args: events.append(args))
    plain = _proposal_transformer(dictionary, PROTECTED_TERMS)
    assert with_trace(text) == plain(text)
    result, stats = with_trace(text)
    assert result == "BAGUS bagus baik ORI bukan ORI sesuai deskripsi kurang baik xbgs oke OKE"
    assert stats["jumlah_slang_dinormalkan"] == 4
    assert events[:4] == [
        ("bgs", "bagus", "BGS", "BAGUS"), ("bgs", "bagus", "bgs", "bagus"),
        ("bagus", "baik", "bagus", "baik"), ("gak bgs", "kurang baik", "gak\tbgs", "kurang baik"),
    ]
    regenerated, summary, frequencies, positions = trace_proposal(pd.Series([text, "bgs bgs"]), dictionary)
    assert regenerated.iloc[0] == result
    assert summary["jumlah_slang_dinormalkan"] == 6
    assert frequencies == Counter({("bgs", "bagus"): 4, ("bagus", "baik"): 1,
                                   ("gak bgs", "kurang baik"): 1})
    assert positions[("bgs", "bagus")] == {0, 1}


def test_protected_counts_nested_terms_whitespace_and_per_row_loss():
    before = pd.Series(["bukan\tORI ORI sesuai\ndeskripsi", "ori"])
    after = pd.Series(["bukan ORI ORI sesuai deskripsi ori", ""])
    counts = {row["istilah"]: row for row in count_protected(before, after)}
    assert counts["ori"]["sebelum"] == counts["ori"]["sesudah"] == 3
    assert counts["ori"]["kemunculan_hilang_per_baris"] == 1
    assert counts["ori"]["baris_kehilangan"] == 1
    assert counts["ori"]["kemunculan_bertambah_per_baris"] == 1
    assert counts["bukan ori"]["sebelum"] == counts["bukan ori"]["sesudah"] == 1
    assert counts["sesuai deskripsi"]["sebelum"] == counts["sesuai deskripsi"]["sesudah"] == 1


def test_top_frequency_distinct_contexts_and_deterministic_sampling():
    original = synthetic_table(["bgs bgs"] * 8 + ["aman"])
    updated, _ = build_proposal_table(original, {"bgs": "bagus"})
    frequencies = Counter({("bgs", "bagus"): 16, ("aman", "aman sekali"): 1})
    positions = {("bgs", "bagus"): set(range(8)), ("aman", "aman sekali"): {8}}
    top = make_top_replacements(original, updated, frequencies, positions)
    pd.testing.assert_frame_equal(top, make_top_replacements(original, updated, frequencies, positions))
    expected_positions = pd.Series(range(8)).sample(n=3, random_state=42).tolist()
    assert [top.iloc[0][f"contoh_{number}_posisi_baris"] for number in range(1, 4)] == expected_positions
    assert top.iloc[0]["frekuensi_penggantian"] == 16
    assert top.iloc[0]["jumlah_ulasan_terdampak"] == 8
    assert top.iloc[1]["jumlah_contoh"] == 1
    assert top.iloc[1]["contoh_2_review"] == top.iloc[1]["contoh_3_review"] == ""
    for number, position in enumerate(expected_positions, start=1):
        assert top.iloc[0][f"contoh_{number}_review_id"] == f"r{position}"
        assert top.iloc[0][f"contoh_{number}_review"] == "bgs bgs"
        assert top.iloc[0][f"contoh_{number}_text_proposal"] == "bagus bagus"


def test_verification_detects_old_column_change_and_regeneration_mismatch():
    original = synthetic_table(["bgs", "ORI", "👍"])
    updated, _ = build_proposal_table(original, {"bgs": "bagus"})
    stats, top = verify_tables(original, updated, {"bgs": "bagus"}, 3, 8)
    assert stats["n_kosong"] == 1
    assert stats["text_light_identik"] and stats["text_lex_identik"]
    assert stats["regenerasi_text_proposal_identik"]
    with pytest.raises(ValueError, match="Jumlah baris"):
        verify_tables(original, updated, {"bgs": "bagus"}, 4, 8)
    with pytest.raises(ValueError, match="Jumlah kolom"):
        verify_tables(original, updated, {"bgs": "bagus"}, 3, 9)
    corrupted = updated.copy()
    corrupted.loc[0, "text_lex"] = "rusak"
    with pytest.raises(AssertionError):
        verify_tables(original, corrupted, {"bgs": "bagus"})
    corrupted = updated.copy()
    corrupted.loc[0, "text_proposal"] = "bgs"
    with pytest.raises(AssertionError):
        verify_tables(original, corrupted, {"bgs": "bagus"})


def test_verification_stops_on_protected_loss_before_artifacts():
    # URL removal genuinely removes this pre-existing isolated token; audit must surface it.
    original = synthetic_table(["https://contoh.test/ori"])
    updated, _ = build_proposal_table(original, {"bgs": "bagus"})
    with pytest.raises(ValueError, match="PROTECTED_TERMS kehilangan"):
        verify_tables(original, updated, {"bgs": "bagus"})


def test_verification_absent_text_lex_is_reported_without_inventing_column():
    original = synthetic_table(["bgs", "ORI"]).drop(columns="text_lex")
    updated, _ = build_proposal_table(original, {"bgs": "bagus"})
    stats, _ = verify_tables(original, updated, {"bgs": "bagus"}, 2, 7)
    assert not stats["text_lex_tersedia"]
    assert stats["text_lex_identik"] is None
    assert stats["text_light_identik"] and stats["seluruh_kolom_lama_identik"]
    assert "text_lex" not in original and "text_lex" not in updated
    assert "text_lex" not in stats["kolom_lama_diverifikasi"]


def test_cli_report_top_csv_hash_and_no_overwrite(tmp_path, capsys):
    original = synthetic_table(["bgs bgs", "ORI", "👍"])
    updated, _ = build_proposal_table(original, {"bgs": "bagus"})
    source, target, dictionary, report, top = [tmp_path / name for name in
        ("old.parquet", "v2.parquet", "slang.csv", "report.md", "top.csv")]
    original.to_parquet(source, index=False)
    updated.to_parquet(target, index=False)
    dictionary.write_text("slang,baku\nbgs,bagus\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in (source, target, dictionary)}
    args = build_parser().parse_args([
        "--input", str(source), "--output", str(target), "--slang-csv", str(dictionary),
        "--report", str(report), "--top-csv", str(top), "--expected-rows", "3",
        "--expected-columns", "8", "--expected-input-sha256", sha256(source),
    ])
    assert run(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["verifikasi"]["sha256_input_tidak_berubah"]
    assert output["verifikasi"]["jumlah_slang_dinormalkan"] == 2
    assert all(path.read_bytes() == content for path, content in before.items())
    assert pd.read_csv(top).iloc[0]["frekuensi_penggantian"] == 2
    assert "random_state=42" in report.read_text()
    preserved = {path: path.read_bytes() for path in (report, top)}
    assert run(args) == 2
    assert "tidak boleh ditimpa" in capsys.readouterr().out
    assert all(path.read_bytes() == content for path, content in preserved.items())


def test_cli_wrong_hash_writes_nothing(tmp_path):
    original = synthetic_table(["bgs"])
    updated, _ = build_proposal_table(original, {"bgs": "bagus"})
    source, target, dictionary, report, top = [tmp_path / name for name in
        ("old.parquet", "v2.parquet", "slang.csv", "report.md", "top.csv")]
    original.to_parquet(source, index=False)
    updated.to_parquet(target, index=False)
    dictionary.write_text("slang,baku\nbgs,bagus\n", encoding="utf-8")
    args = build_parser().parse_args([
        "--input", str(source), "--output", str(target), "--slang-csv", str(dictionary),
        "--report", str(report), "--top-csv", str(top), "--expected-input-sha256", "0" * 64,
    ])
    assert run(args) == 2
    assert not report.exists() and not top.exists()
