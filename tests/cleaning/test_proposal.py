"""Aturan proposal dan kontrak v3; kamus di sini hanya fixture sintetis untuk tes."""

import numpy as np
import pandas as pd
import pytest

from src.cleaning.config import PROTECTED_TERMS
from src.cleaning.proposal import (
    build_proposal_table, clean_proposal, clean_proposal_series, format_proposal_report,
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


@pytest.mark.parametrize(("text", "expected"), [
    ("𝙗𝙖𝙜𝙪𝙨", "bagus"),
    ("ｂａｇｕｓ", "bagus"),
    ("bagus\u00a0sekali", "bagus sekali"),
    ("bagus\u2028sekali", "bagus sekali"),
    ("bagus\u2029sekali", "bagus sekali"),
    ("𝙗𝙖𝙜𝙪𝙨 👍🏽 👨‍👩‍👧 🇮🇩 ❤️ 1️⃣", "bagus"),
    ("bagus ｈｔｔｐｓ：／／ｃｏｎｔｏｈ．ｉｄ／ｂｇｓ", "bagus"),
    ("bagus 𝙝𝙩𝙩𝙥𝙨://contoh.id/bgs", "bagus"),
])
def test_unicode_is_normalized_before_other_rules(text, expected):
    assert clean_proposal(text) == expected


@pytest.mark.parametrize("invisible", ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"])
def test_zero_width_characters_are_removed_without_splitting_words(invisible):
    assert clean_proposal(f"{invisible}ba{invisible}gus{invisible}") == "bagus"
    assert clean_proposal(f"b{invisible}gs", {"bgs": "bagus"}) == "bagus"


def test_slang_matches_after_unicode_normalization():
    fake_dictionary = {"bgs": "bagus", "gk bgs": "kurang bagus"}
    assert clean_proposal("𝙗𝙜𝙨 ｂｇｓ b\u200bgs", fake_dictionary) == "bagus bagus bagus"
    assert clean_proposal("gk\u00a0bgs", fake_dictionary) == "kurang bagus"


def test_nfkc_counts_changed_rows_separately_from_unicode_separators():
    series = pd.Series(["𝙗𝙖𝙜𝙪𝙨", "ｂａｇｕｓ", "bagus\u00a0sekali", "ba\u200bgus",
                        "bagus\u2028sekali", "bagus", "𝙗𝙖𝙜𝙪𝙨 ｂａｇｕｓ", None])
    result, stats = clean_proposal_series(series, {})
    assert result.tolist() == ["bagus", "bagus", "bagus sekali", "bagus",
                               "bagus sekali", "bagus", "bagus bagus", ""]
    rules = stats["baris_berubah_per_aturan"]
    assert list(rules)[:3] == ["nfkc", "normalisasi_pemisah_unicode", "hapus_url"]
    # NFKC menghitung baris (bukan karakter); NBSP sudah menjadi spasi pada tahap ini.
    assert rules["nfkc"] == 4
    assert rules["normalisasi_pemisah_unicode"] == 2
    assert stats["n_berubah"] == 6


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
        if rule in ("nfkc", "normalisasi_pemisah_unicode"):
            assert count == 0
        else:
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


def test_unicode_normalization_only_changes_new_proposal_columns(original):
    original["review"] = ["𝙗𝙖𝙜𝙪𝙨", "ｂｇｓ\u00a0sekali", "b\u200bgs"]
    original["text_light"] = ["sumber lain", "ｂｇｓ", "b\u200bgs"]
    original["text_lex"] = ["jangan\u00a0ubah", "𝙗𝙖𝙜𝙪𝙨", "tetap\u2028sama"]
    original["nama_toko"] = ["Ｔｏｋｏ", "Toko\ufeff", "Café"]
    original["nama_produk"] = ["𝙗𝙖𝙧𝙖𝙣𝙜", "Produk\u200d", "128\u00a0GB"]
    before = original.copy(deep=True)
    result, stats = build_proposal_table(original, {"bgs": "bagus"})
    assert result.text_proposal.tolist() == ["bagus", "bagus sekali", "bagus"]
    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_frame_equal(result[list(before)], before)
    verify_original_columns(before, result)
    assert stats["seluruh_kolom_lama_identik"]


def test_report_identifies_original_source_and_nfkc_count(original, tmp_path):
    original["review"] = ["𝙗𝙖𝙜𝙪𝙨", "ｂｇｓ", "bagus"]
    result, stats = build_proposal_table(original, {"bgs": "bagus"})
    report = format_proposal_report(result, stats, tmp_path / "old.parquet",
                                    tmp_path / "v3.parquet", tmp_path / "slang.csv")
    assert "review asli + NFKC" in report
    assert "| nfkc | 2 |" in report
    assert "| normalisasi_pemisah_unicode | 0 |" in report


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


def test_cli_defaults_use_v3_output_and_report():
    args = build_parser().parse_args([])
    assert args.output.as_posix().endswith("outputs/cleaned/reviews_clean_v3.parquet")
    assert args.report.as_posix().endswith("reports/preprocessing_proposal_v3.md")


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
