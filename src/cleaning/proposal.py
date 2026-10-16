from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from .config import PROTECTED_TERMS
from .text_clean import URL_IN_TEXT_RE, WHITESPACE_RE, REPEATED_LETTER_RE


ALLOWED_PUNCTUATION = ".,!?' -"
REPEATED_PUNCTUATION_RE = re.compile(r"([.,!?'\-])\1+")
LONG_NUMBER_RE = re.compile(r"(?<!\w)\d{6,}(?!\w)")
KEYCAP_RE = re.compile("[0-9#*]\ufe0f?\u20e3")
UNICODE_SEPARATOR_TRANSLATION = str.maketrans({
    **{ch: None for ch in "\u200b\u200c\u200d\u2060\ufeff"},
    **{ch: " " for ch in "\u2028\u2029\u00a0"},
})


def load_slang_dictionary(path: str | Path) -> dict[str, str]:
    """Baca CSV UTF-8 dua kolom slang,baku; tidak mengunduh/mengarang kamus."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Kamus slang tidak ditemukan: "{path}"; sediakan CSV slang,baku.')
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if list(table.columns) != ["slang", "baku"]:
        raise ValueError("Kamus harus CSV dua kolom berurutan: slang,baku")
    if table.empty:
        raise ValueError("Kamus slang kosong; sediakan pasangan slang,baku yang telah diperiksa")
    dictionary: dict[str, str] = {}
    for slang, baku in table.itertuples(index=False, name=None):
        key = WHITESPACE_RE.sub(" ", slang).strip().casefold()
        value = WHITESPACE_RE.sub(" ", baku).strip()
        if not key or not value:
            raise ValueError("Kamus slang memuat slang/baku kosong")
        if key in dictionary and dictionary[key] != value:
            raise ValueError(f"Kamus memuat pemetaan bertentangan untuk {key!r}")
        dictionary[key] = value
    return dictionary


def _phrase_pattern(terms: Sequence[str]) -> re.Pattern:
    choices = [r"\s+".join(re.escape(word) for word in term.split())
               for term in sorted(set(terms), key=lambda term: (-len(term), term)) if term.strip()]
    return re.compile(r"(?<!\w)(?:" + "|".join(choices) + r")(?!\w)" if choices else r"(?!)", re.I)


def _unprotected(text: str, pattern: re.Pattern, transform: Callable[[str], str]) -> str:
    parts: list[str] = []
    end = 0
    for match in pattern.finditer(text):
        parts.extend((transform(text[end:match.start()]), match.group()))
        end = match.end()
    parts.append(transform(text[end:]))
    return "".join(parts)


def _strip_special(text: str) -> str:
    # Hapus seluruh emoji keycap (termasuk digitnya), bukan menyisakan angka palsu.
    text = KEYCAP_RE.sub(" ", text)
    return "".join(ch if ch.isalpha() or ch.isdecimal() or ch.isspace()
                   or ch in ALLOWED_PUNCTUATION else " " for ch in text)


def _proposal_transformer(
    slang: Mapping[str, str] | None,
    protected_terms: Sequence[str],
    on_slang: Callable[[str, str, str, str], None] | None = None,
) -> Callable[[object], tuple[str, dict[str, int]]]:
    """Opsional: laporkan (kunci, baku, token asal, hasil) untuk penggantian aktual."""
    protected = _phrase_pattern(protected_terms)
    dictionary = {WHITESPACE_RE.sub(" ", key).strip().casefold(): value
                  for key, value in (slang or {}).items()}
    slang_pattern = _phrase_pattern(list(dictionary))

    def transform(value: object) -> tuple[str, dict[str, int]]:
        text = "" if value is None or pd.isna(value) else str(value)
        changes: dict[str, int] = {}
        replacements = 0

        def normalize_slang(part: str) -> str:
            def replace(match: re.Match) -> str:
                nonlocal replacements
                original = match.group()
                key = WHITESPACE_RE.sub(" ", original).casefold()
                baku = dictionary[key]
                replacement = baku
                # Tidak lowercase seluruh ulasan; pertahankan pola kapital token sumber.
                if original.isupper():
                    replacement = replacement.upper()
                elif original.istitle():
                    replacement = replacement[:1].upper() + replacement[1:]
                replacements += int(replacement != original)
                if replacement != original and on_slang is not None:
                    on_slang(key, baku, original, replacement)
                return replacement
            return slang_pattern.sub(replace, part)

        steps = [
            ("nfkc", lambda s: unicodedata.normalize("NFKC", s)),
            ("normalisasi_pemisah_unicode", lambda s: s.translate(UNICODE_SEPARATOR_TRANSLATION)),
            ("hapus_url", lambda s: URL_IN_TEXT_RE.sub(" ", s)),
            ("hapus_emoji_karakter_khusus", _strip_special),
            ("ringkas_tanda_baca", lambda s: REPEATED_PUNCTUATION_RE.sub(r"\1", s)),
            ("hapus_angka_panjang", lambda s: LONG_NUMBER_RE.sub(" ", s)),
            ("normalisasi_typo_huruf_berulang", lambda s: _unprotected(
                s, protected, lambda part: REPEATED_LETTER_RE.sub(r"\1", part))),
            ("normalisasi_slang", lambda s: _unprotected(s, protected, normalize_slang)),
            ("rapikan_spasi", lambda s: WHITESPACE_RE.sub(" ", s).strip()),
        ]
        for name, step in steps:
            updated = step(text)
            changes[name] = int(updated != text)
            text = updated
        changes["jumlah_slang_dinormalkan"] = replacements
        return text, changes

    return transform


def clean_proposal(
    text: object,
    slang: Mapping[str, str] | None = None,
    protected_terms: Sequence[str] = PROTECTED_TERMS,
) -> str:

    return _proposal_transformer(slang, protected_terms)(text)[0]


def clean_proposal_series(
    series: pd.Series,
    slang: Mapping[str, str] | None = None,
    protected_terms: Sequence[str] = PROTECTED_TERMS,
) -> tuple[pd.Series, dict]:
    """Hitung baris berubah pada tiap tahap dan jumlah kemunculan slang diganti."""

    transform = _proposal_transformer(slang, protected_terms)
    counts = {key: 0 for key in transform("")[1]}
    texts = []
    for value in series:
        text, changes = transform(value)
        texts.append(text)
        for key, count in changes.items():
            counts[key] += count
    result = pd.Series(texts, index=series.index, name="text_proposal", dtype="object")
    original = series.fillna("").astype(str)
    stats = {
        "n_baris": len(series), "n_berubah": int((result != original).sum()),
        "n_kosong": int(result.eq("").sum()),
        "slang_dijalankan": slang is not None,
        "jumlah_slang_dinormalkan": counts.pop("jumlah_slang_dinormalkan"),
        "baris_berubah_per_aturan": counts,
    }
    return result, stats


def build_proposal_table(reviews: pd.DataFrame, slang: Mapping[str, str] | None) -> tuple[pd.DataFrame, dict]:
    """Tambah dua kolom tanpa filter/sort/ubah kolom lama; None untuk audit parsial."""

    required = ("review", "review_id", "seq_in_product", "include_for_inference",
                "is_technical_duplicate", "is_text_duplicate")
    missing = [name for name in required if name not in reviews]
    if missing:
        raise ValueError(f"Kolom wajib tidak tersedia: {missing}")
    if any(name in reviews for name in ("text_proposal", "text_proposal_kosong")):
        raise ValueError("Kolom text_proposal/text_proposal_kosong sudah ada; tidak boleh ditimpa")
    result = reviews.copy()
    result["text_proposal"], stats = clean_proposal_series(reviews["review"], slang)
    result["text_proposal_kosong"] = result["text_proposal"].eq("")
    verify_original_columns(reviews, result)
    included = reviews["include_for_inference"].eq(True).fillna(False)
    technical = reviews["is_technical_duplicate"].eq(True).fillna(False)
    duplicate = reviews["is_text_duplicate"].eq(True).fillna(False)
    stats.update({
        "n_include_for_inference": int(included.sum()),
        "n_technical_duplicate": int(technical.sum()),
        "n_technical_duplicate_masih_include": int((technical & included).sum()),
        "n_text_duplicate": int(duplicate.sum()),
        "n_text_duplicate_masih_include": int((duplicate & included).sum()),
        "n_kosong_masih_include": int((result["text_proposal_kosong"] & included).sum()),
        "jumlah_baris_identik": len(reviews) == len(result),
        "urutan_review_id_seq_identik": reviews[["review_id", "seq_in_product"]].equals(
            result[["review_id", "seq_in_product"]]),
        "seluruh_kolom_lama_identik": reviews.equals(result[list(reviews.columns)]),
    })
    return result, stats


def verify_original_columns(original: pd.DataFrame, updated: pd.DataFrame) -> None:
    if len(original) != len(updated) or not original.equals(updated[list(original.columns)]):
        raise ValueError("BERHENTI: jumlah, urutan, tipe, atau nilai kolom lama berbeda; minta konfirmasi pengguna")


def format_proposal_report(table: pd.DataFrame, stats: dict, input_path: Path,
                           output_path: Path | None, dictionary_path: Path | None) -> str:
    """Laporan audit dengan 20 contoh acak, seed 42, dari seluruh baris."""

    lines = ["# Audit preprocessing proposal Bab 3.3", "",
             f'- Masukan: "{input_path}"',
             f'- Keluaran: "{output_path}"' if output_path else "- Keluaran parquet belum ditulis.",
             f'- Kamus: "{dictionary_path}"' if dictionary_path else
             "- PARSIAL: kamus slang belum tersedia; slang belum dijalankan. Angka di bawah belum final.", ""]
    for key, value in stats.items():
        if key != "baris_berubah_per_aturan":
            lines.append(f"- {key}: {value}")
    lines += ["", "## Per aturan (perbandingan sebelum/sesudah langkah tersebut)", "",
              "Satu baris dapat berubah pada lebih dari satu aturan; hitungan tidak dijumlahkan.", "",
              "| Aturan | Baris berubah |", "| --- | ---: |"]
    for key, value in stats["baris_berubah_per_aturan"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## Asumsi dan penyimpangan", "",
              "Sumber teks adalah review asli + NFKC pada tabel lama. NFKC merupakan langkah pertama, "
              "sebelum penghapusan URL; jumlah baris yang berubah ditampilkan pada aturan nfkc. "
              "Setelah NFKC, U+200B/U+200C/U+200D/U+2060/U+FEFF dihapus, sedangkan "
              "U+2028/U+2029/NBSP diganti spasi biasa pada aturan normalisasi_pemisah_unicode. "
              "NBSP yang telah menjadi spasi pada NFKC dihitung pada aturan nfkc. "
              "Kolom review, text_light, dan seluruh kolom lama dipertahankan apa adanya. "
              "Karakter di luar huruf Unicode, "
              "digit desimal, spasi, dan . , ! ? ' - diganti spasi agar kata tidak tergabung. "
              "Tanda baca yang sama dan berurutan diringkas; kombinasi seperti !? tetap. "
              "Huruf berulang dihitung sebagai normalisasi typo; tidak ada lowercase paksa.", "",
              "Asumsi angka konservatif: hanya deretan >= 6 digit yang berdiri sendiri dihapus. "
              "Angka pendek dan alfanumerik (128gb, 5g, a07, iphone 17) tetap. "
              "Nomor yang dipisah spasi/tanda baca tidak digabung untuk dihapus.", "",
              "Pencocokan slang tidak peka kapital, sekali jalan pada batas kata/frasa; "
              "kapital semua/awal token dipertahankan. Frasa terlindungi diprioritaskan. "
              "Mempertahankan kw adalah PENYIMPANGAN disengaja dari contoh normalisasi proposal 3.3.", "",
              "PROTECTED_TERMS: " + ", ".join(f"`{term}`" for term in PROTECTED_TERMS), "",
              "## Semantik duplikat dan rekomendasi", "",
              "is_technical_duplicate: produk + text_light + timestamp berjam identik; kemunculan "
              "setelah yang pertama dikecualikan dari include_for_inference. is_text_duplicate: "
              "teks sama dalam produk, masih >= 2 ulasan setelah duplikat teknis disisihkan; "
              "flag dapat menempel juga pada baris teknis dalam grup tersebut. Teks sama tidak otomatis spam.", "",
              "Rekomendasi: pertahankan flag saat ini dan tinjau sampel duplikat sebelum mengubah aturan. "
              "CLI sentimen tetap menginferensi semua teks nonkosong, termasuk flag include=False, "
              "seperti versi lama; downstream dapat memfilter. Tidak ada perubahan include_for_inference. "
              "Teks proposal kosong tetap menjadi baris dengan probabilitas NaN.", "",
              "## 20 contoh acak (seluruh tabel, random_state=42)", "",
              "| review_id | review | text_proposal |", "| --- | --- | --- |"]

    def cell(value: object) -> str:
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(
            ">", "&gt;").replace("|", "&#124;").replace("\n", "<br>").replace("\r", "")

    for row in table.sample(n=min(20, len(table)), random_state=42).itertuples():
        lines.append("| " + " | ".join(cell(value) for value in
                                       (row.review_id, row.review, row.text_proposal)) + " |")
    return "\n".join(lines) + "\n"
