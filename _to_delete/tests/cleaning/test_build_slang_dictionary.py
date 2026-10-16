"""Kontrak konversi dengan fixture sintetis; tidak membaca kamus produksi."""

import csv
import hashlib
from collections import Counter

import pytest

from src.cleaning.build_slang_dictionary import convert_rows, main
from src.cleaning.config import PROTECTED_TERMS


SOURCE_FIELDS = ["slang", "formal", "in_dictionary", "context", "categories"]
REASONS = {
    "kosong", "duplikat_pasangan", "slang_sama_baku", "protected_terms",
    "multi_baku", "kunci_satu_huruf", "kunci_dua_huruf",
}


def source_row(slang, formal, **extra):
    return {
        "slang": slang, "formal": formal, "in_dictionary": "False",
        "context": "@nama_fixture komentar sintetis", "categories": "['afiksasi']",
        **extra,
    }


def write_source(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def cli_paths(tmp_path):
    return {
        "input": tmp_path / "source.csv", "output": tmp_path / "slang.csv",
        "excluded": tmp_path / "excluded.csv", "profile": tmp_path / "profile.md",
    }


def cli_args(paths):
    return [part for name, path in paths.items() for part in ("--" + name, str(path))]


def kept_rows():
    return [source_row(key, value) for key, value in [
        ("bgs", "bagus"), ("gpp", "tidak apa-apa"), ("gmn", "bagaimana"),
        ("bgt", "banget"), ("nggak", "tidak"), ("skrg", "sekarang"), ("udh", "sudah"),
    ]]


def test_normalized_duplicate_pairs_count_once_and_preserve_formal_case():
    accepted, excluded, stats = convert_rows([
        source_row("  BGS  ", "  Bagus  Sekali "),
        source_row("bgs", "Bagus\tSekali", context="konteks lain"),
        source_row("NgGaK", "tidak"),
    ])
    assert accepted == [
        {"slang": "bgs", "baku": "Bagus Sekali"}, {"slang": "nggak", "baku": "tidak"},
    ]
    assert excluded == [{"slang": "bgs", "baku": "Bagus Sekali", "alasan": "duplikat_pasangan"}]
    assert stats["n_baris"] == 3
    assert stats["n_kunci_unik"] == stats["n_pasangan_unik"] == stats["n_diterima"] == 2
    assert stats["n_kunci_dikeluarkan"] == 0
    assert stats["alasan"]["duplikat_pasangan"] == 1


def test_conflicting_key_loses_every_mapping_without_majority_choice():
    rows = [source_row("gpp", "tidak apa-apa")] * 8 + [source_row("GPP", "tidak masalah")]
    accepted, excluded, stats = convert_rows(rows)
    assert accepted == []
    assert len(excluded) == len(rows)
    assert stats["n_kunci_multi_baku"] == 1
    assert Counter(row["alasan"] for row in excluded) == {"multi_baku": 2, "duplikat_pasangan": 7}


@pytest.mark.parametrize(("first_formal", "first_reason"), [
    ("BGS", "slang_sama_baku"), ("ori", "protected_terms"),
])
def test_ambiguity_is_checked_before_identity_and_protected_filtering(first_formal, first_reason):
    accepted, excluded, stats = convert_rows([
        source_row("bgs", first_formal), source_row("bgs", "bagus"),
    ])
    assert accepted == []
    assert stats["n_kunci_multi_baku"] == 1
    assert {row["baku"]: row["alasan"] for row in excluded} == {
        first_formal: first_reason, "bagus": "multi_baku",
    }


@pytest.mark.parametrize("term", PROTECTED_TERMS)
@pytest.mark.parametrize("side", ["slang", "formal"])
def test_protected_terms_match_casefolded_substrings_on_both_sides(term, side):
    # Term dibungkus tanpa batas kata agar tes membedakan substring dari token.
    values = {"slang": "bgs", "formal": "bagus"}
    values[side] = "xxx" + term.upper() + "yyy"
    accepted, excluded, _ = convert_rows([source_row(**values)])
    assert accepted == []
    assert len(excluded) == 1
    assert excluded[0]["alasan"] == "protected_terms"


def test_short_keys_identity_and_empty_rows_are_reported_separately():
    accepted, excluded, stats = convert_rows([
        source_row("é", "iya"), source_row("ga", "tidak"), source_row(" GAK ", "tidak"),
        source_row("  AMAN ", "aman"), source_row("", "baik"), source_row("oke", " \t "),
    ])
    assert accepted == [{"slang": "gak", "baku": "tidak"}]
    assert Counter(row["alasan"] for row in excluded) == {
        "kunci_satu_huruf": 1, "kunci_dua_huruf": 1, "slang_sama_baku": 1, "kosong": 2,
    }
    assert set(stats["alasan"]) == REASONS
    assert stats["alasan"]["protected_terms"] == 0
    assert stats["n_diterima"] + sum(stats["alasan"].values()) == stats["n_baris"]


def test_context_categories_and_dictionary_flag_do_not_filter_or_leak():
    rows = [source_row("bgs", "bagus", in_dictionary=flag, categories=categories)
            for flag, categories in [("True", "['elongasi']"), ("misteri", "[]")]]
    accepted, excluded, _ = convert_rows(rows)
    assert accepted == [{"slang": "bgs", "baku": "bagus"}]
    assert excluded == [{"slang": "bgs", "baku": "bagus", "alasan": "duplikat_pasangan"}]
    assert all(set(row) == {"slang", "baku"} for row in accepted)
    assert all(set(row) == {"slang", "baku", "alasan"} for row in excluded)


def test_row_order_does_not_change_dictionary_exclusions_or_counters():
    rows = kept_rows() + [source_row("BGS", "bagus"), source_row("gpp", "tidak masalah"),
                          source_row("ga", "tidak"), source_row("", "")]
    accepted, excluded, stats = convert_rows(rows)
    reversed_accepted, reversed_excluded, reversed_stats = convert_rows(list(reversed(rows)))
    assert accepted == reversed_accepted
    assert excluded == reversed_excluded
    for name in ("n_baris", "n_kunci_unik", "n_pasangan_unik", "n_kunci_multi_baku",
                 "n_diterima", "n_kunci_dikeluarkan", "persen_kunci_dikeluarkan",
                 "gerbang_30_persen", "alasan"):
        assert stats[name] == reversed_stats[name]
    assert accepted == sorted(accepted, key=lambda row: (row["slang"], row["baku"]))
    assert excluded == sorted(excluded, key=lambda row: (row["slang"], row["baku"], row["alasan"]))


def test_cli_writes_exact_headers_utf8_and_no_context_even_when_source_has_newline(tmp_path):
    paths = cli_paths(tmp_path)
    rows = kept_rows() + [source_row("yéy", "senang", context="@nama_fixture baris satu\nbaris dua"),
                          source_row("ga", "tidak")]
    write_source(paths["input"], rows)
    original = paths["input"].read_bytes()
    args = cli_args(paths) + ["--expected-sha256", hashlib.sha256(original).hexdigest()]
    assert main(args) == 0
    for name, fields in [("output", ["slang", "baku"]), ("excluded", ["slang", "baku", "alasan"])]:
        content = paths[name].read_bytes()
        assert not content.startswith(b"\xef\xbb\xbf")
        text = content.decode("utf-8")
        assert text.splitlines()[0] == ",".join(fields)
        assert "@nama_fixture" not in text
        assert "context" not in text
        with paths[name].open(encoding="utf-8", newline="") as stream:
            assert csv.DictReader(stream).fieldnames == fields
    assert "yéy,senang" in paths["output"].read_text(encoding="utf-8")
    assert paths["input"].read_bytes() == original
    assert paths["profile"].is_file()


@pytest.mark.parametrize("slang", ["cafÃ©", "Ânggak"])
def test_profile_detects_mojibake_before_key_casefold(tmp_path, slang):
    paths = cli_paths(tmp_path)
    write_source(paths["input"], [source_row(slang, "kopi")])
    assert main(cli_args(paths)) == 0
    lines = paths["profile"].read_text(encoding="utf-8").splitlines()
    marker_line = next(line for line in lines if "Pasangan dengan penanda" in line)
    assert marker_line.endswith(": 1.")


@pytest.mark.parametrize("existing_name", ["output", "excluded", "profile"])
def test_cli_refuses_any_existing_output_without_partial_writes(tmp_path, existing_name):
    paths = cli_paths(tmp_path)
    write_source(paths["input"], kept_rows())
    paths[existing_name].write_text("jangan ditimpa", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert main(cli_args(paths)) == 2
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_cli_hash_mismatch_stops_before_writing_anything(tmp_path):
    paths = cli_paths(tmp_path)
    write_source(paths["input"], kept_rows())
    original = paths["input"].read_bytes()
    assert main(cli_args(paths) + ["--expected-sha256", "0" * 64]) == 2
    assert list(tmp_path.iterdir()) == [paths["input"]]
    assert paths["input"].read_bytes() == original


@pytest.mark.parametrize(("extra_rows", "expected_code", "expected_rejected"), [
    ([], 0, 3), ([source_row("tp", "tetapi")], 2, 4),
])
def test_cli_gate_uses_unique_keys_not_duplicate_rows_and_is_strictly_over_30_percent(
    tmp_path, extra_rows, expected_code, expected_rejected,
):
    # Tujuh kunci diterima + tiga identitas = tepat 30%; banyak duplikat tidak
    # menaikkan proporsi kunci yang hilang. Kunci dua huruf keempat melewati batas.
    rows = kept_rows() + [source_row(key, key) for key in ("aman", "bagus", "cepat")]
    rows += [source_row("bgs", "bagus")] * 30 + extra_rows
    _, _, stats = convert_rows(rows)
    assert stats["n_kunci_dikeluarkan"] == expected_rejected
    assert stats["gerbang_30_persen"] is (expected_code == 2)
    assert stats["persen_kunci_dikeluarkan"] == pytest.approx(expected_rejected / (10 + len(extra_rows)) * 100)
    paths = cli_paths(tmp_path)
    write_source(paths["input"], rows)
    assert main(cli_args(paths)) == expected_code
    # Di gerbang audit masih ditulis, tetapi kode keluar mencegah tahap berikut.
    assert all(path.is_file() for path in paths.values())
    with paths["output"].open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 7
