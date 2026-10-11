"""Aturan proposal dan kontrak v2; kamus di sini hanya fixture sintetis untuk tes."""

import numpy as np
import pandas as pd
import pytest

from src.cleaning.config import PROTECTED_TERMS
from src.cleaning.proposal import (
    build_proposal_table, clean_proposal, clean_proposal_series,
    load_slang_dictionary, verify_original_columns,
)
from src.cleaning.run_proposal import EXIT_OK, EXIT_STOPPED, build_parser, run
from src.sentiment_inference import PROB_COLUMNS, SentimentBundle, predict_sentiment_probs


@pytest.mark.parametrize(("text", "expected"), [
    ("Bagus https://contoh.id/x?order=123456 www.contoh.id aman", "Bagus aman"),
    ("Bagus 👍🏽 👨‍👩‍👧 🇮🇩 ❤️ 1️⃣ #*©®$ aman", "Bagus aman"),
    ("Énak_中文 / oke: (YA)", "Énak 中文 oke YA"),
    ("Bagus!!!....,,??''---", "Bagus!.,?'-"),
    ("!?!!??", "!?!?"),
    ("bagusss mantaaap BAGUSSS 5000mAh", "bagus mantap BAGUS 5000mAh"),
    ("123456 081234567890 12345 128gb 5g a07 iphone 17 a123456 123456gb",
     "12345 128gb 5g a07 iphone 17 a123456 123456gb"),
    ("Don't ubah-Huruf, Ya.", "Don't ubah-Huruf, Ya."),
    ("  Bagus\tAMAN\n sekali  ", "Bagus AMAN sekali"),
    ("👍🏽 👨‍👩‍👧 1️⃣ https://contoh.id 123456", ""),
    (None, ""), (np.nan, ""), (pd.NA, ""),
])
def test_each_proposal_rule(text, expected):
    assert clean_proposal(text) == expected


def test_protected_terms_and_phrases_survive_dictionary():
    # Pemetaan konflik disengaja untuk membuktikan perlindungan, bukan kamus produksi.
    fake_dictionary = {term: "DIUBAH" for term in PROTECTED_TERMS}
    fake_dictionary.update({"bukan": "DIUBAH", "sesuai": "DIUBAH", "deskripsi": "DIUBAH"})
    for text in (" | ".join(PROTECTED_TERMS), " | ".join(PROTECTED_TERMS).upper()):
        assert clean_proposal(text, fake_dictionary) == text.replace(" | ", " ")
    assert clean_proposal("bukan   ORI sesuai\tdeskripsi", fake_dictionary) == "bukan ORI sesuai deskripsi"


def test_configurable_protection_applies_to_typo_and_slang():
    assert clean_proposal("mantaaap bgs", {"bgs": "bagus"}, ("mantaaap", "bgs")) == "mantaaap bgs"


def test_slang_boundaries_case_and_one_pass():
    fake_dictionary = {"bgs": "bagus", "bagus": "hebat"}
    assert clean_proposal("bgs BGS Bgs xbgs bgsx", fake_dictionary) == "bagus BAGUS Bagus xbgs bgsx"


def test_order_counts_empty_and_determinism():
    series = pd.Series(["BGS!!! 👍 123456 mantaaap https://a.id", "👍", "ori", None], index=[4, 2, 8, 1])
    result, stats = clean_proposal_series(series, {"bgs": "bagus"})
    assert result.tolist() == ["BAGUS! mantap", "", "ori", ""]
    assert result.index.equals(series.index)
    assert stats["n_berubah"] == 2 and stats["n_kosong"] == 2
    assert stats["jumlah_slang_dinormalkan"] == 1
    for rule, count in stats["baris_berubah_per_aturan"].items():
        assert count == (2 if rule in ("hapus_emoji_karakter_khusus", "rapikan_spasi") else 1)
    again, again_stats = clean_proposal_series(series, {"bgs": "bagus"})
    pd.testing.assert_series_equal(result, again)
    assert stats == again_stats


@pytest.fixture
def original():
    return pd.DataFrame({
        "review_id": ["b2", "a1", "a2"], "seq_in_product": [2, 1, 2],
        "review": ["bgs!!!", "👍", "ori"], "text_light": ["bgs!!!", "👍", "ori"],
        "text_lex": ["jangan_ubah"] * 3,
        "include_for_inference": [False, True, True],
        "is_technical_duplicate": [True, False, False],
        "is_text_duplicate": [True, True, True],
    })


def test_table_preserves_all_original_columns_order_and_flags(original):
    result, stats = build_proposal_table(original, {"bgs": "bagus"})
    pd.testing.assert_frame_equal(original, result[list(original)])
    assert result.text_proposal.tolist() == ["bagus!", "", "ori"]
    assert result.text_proposal_kosong.tolist() == [False, True, False]
    assert stats["n_technical_duplicate_masih_include"] == 0
    assert stats["n_text_duplicate_masih_include"] == 2
    assert stats["n_kosong_masih_include"] == 1
    assert stats["urutan_review_id_seq_identik"]
    # Audit tanpa kamus eksplisit parsial, tidak mengaku slang telah dikerjakan.
    _, partial = build_proposal_table(original, None)
    assert not partial["slang_dijalankan"]
    assert partial["jumlah_slang_dinormalkan"] == 0


def test_cleaned_empty_texts_can_be_inferred_as_nan(original):
    result, _ = build_proposal_table(original, {})

    def tokenizer(texts, return_tensors=None, **kwargs):
        ids = [[1, 3, 2] for _ in texts]
        return {"input_ids": np.array(ids)} if return_tensors else {"input_ids": ids}

    bundle = SentimentBundle(tokenizer, lambda batch: np.zeros((len(batch["input_ids"]), 3)),
                             {"negative": 0, "neutral": 1, "positive": 2}, "cpu")
    probs = predict_sentiment_probs(result.text_proposal, bundle=bundle)
    assert probs.loc[1, list(PROB_COLUMNS)].isna().all()
    assert probs.loc[[0, 2], list(PROB_COLUMNS)].notna().all().all()


def test_row_or_order_change_is_rejected(original):
    for updated in (original.iloc[::-1], original.iloc[:-1]):
        with pytest.raises(ValueError, match="jumlah, urutan"):
            verify_original_columns(original, updated)


@pytest.mark.parametrize("contents", ["a,b\nx,y\n", "slang,baku\n", "slang,baku\nbgs,\n",
                                     "slang,baku\nbgs,bagus\nBGS,lain\n"])
def test_dictionary_validation(tmp_path, contents):
    path = tmp_path / "slang.csv"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_slang_dictionary(path)


def test_cli_no_dictionary_stops_and_no_files_written(tmp_path, original):
    source = tmp_path / "old.parquet"
    original.to_parquet(source, index=False)
    args = build_parser().parse_args(["--input", str(source), "--output", str(tmp_path / "v2.parquet"),
                                      "--report", str(tmp_path / "report.md")])
    assert run(args) == EXIT_STOPPED
    assert sorted(p.name for p in tmp_path.iterdir()) == ["old.parquet"]


def test_cli_round_trip_report_and_no_overwrite(tmp_path, original):
    source, target, report, dictionary = [tmp_path / name for name in
                                         ("old.parquet", "v2.parquet", "report.md", "slang.csv")]
    original.to_parquet(source, index=False)
    original_bytes = source.read_bytes()
    dictionary.write_text("slang,baku\nbgs,bagus\n")  # fixture, bukan kamus produksi
    args = build_parser().parse_args(["--input", str(source), "--output", str(target),
                                      "--report", str(report), "--slang-csv", str(dictionary)])
    assert run(args) == EXIT_OK
    pd.testing.assert_frame_equal(original, pd.read_parquet(target)[list(original)])
    assert source.read_bytes() == original_bytes
    assert "PROTECTED_TERMS" in report.read_text()
    assert "jumlah_slang_dinormalkan: 1" in report.read_text()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert run(args) == EXIT_STOPPED
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
