from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd


# ============================================================
# CATATAN
#
# Pembersihan MINIMAL untuk Transformer (IndoBERT):
# - TIDAK stemming, TIDAK hapus stopword, TIDAK normalisasi slang,
#   TIDAK lowercase, TIDAK ubah ejaan; emoji dipertahankan.
# - normalize_text_series (src/preprocessing.py) TIDAK dipakai ulang:
#   regex `(.)\1{2,}` di sana juga memotong angka ("5000mAh" -> "500mAh")
#   dan tidak membuang karakter kontrol/zero-width.
# ============================================================


HTML_TAG_RE = re.compile(r"<[^>]+>")

URL_IN_TEXT_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)

# Hanya HURUF yang berulang >2 dipendekkan menjadi 2
# ("bagusss" -> "baguss"); angka, tanda baca, emoji tidak disentuh.
REPEATED_LETTER_RE = re.compile(r"([^\W\d_])\1{2,}")

_EMOJI_CLASS = "\U0001F000-\U0001FAFF☀-➿️"

# ZWJ di luar rangkaian emoji (ZWJ di dalam emoji 👨‍👩‍👧 dipertahankan).
STRAY_ZWJ_RE = re.compile(
    rf"(?<![{_EMOJI_CLASS}])‍|‍(?![{_EMOJI_CLASS}])"
)

WHITESPACE_RE = re.compile(r"\s+")

SENTENCE_END = (".", "!", "?", "…", ",", ";", ":")


# ============================================================
# PENGGABUNGAN FRAGMEN
# ============================================================

def join_review(
    fragments: list[str],
    separator: str = " | ",
) -> str:
    """Kolom `review`: fragmen apa adanya (di-strip), pemisah ' | '."""

    return separator.join(fragment.strip() for fragment in fragments)


def join_for_light(
    fragments: list[str],
) -> str:
    """
    Untuk text_light, pemisah ' | ' diganti tanda baca alami:
    fragmen yang belum diakhiri tanda baca diberi '. ', selain itu ' '.
    Alasannya: '|' bukan token bermakna bagi IndoBERT.
    """

    parts: list[str] = []

    for index, fragment in enumerate(fragments):
        fragment = fragment.strip()
        if not fragment:
            continue
        if index < len(fragments) - 1 and not fragment.endswith(SENTENCE_END):
            fragment += "."
        parts.append(fragment)

    return " ".join(parts)


# ============================================================
# LANGKAH PEMBERSIHAN (urutan tetap)
# ============================================================

def _strip_html(text: str) -> str:
    return HTML_TAG_RE.sub(" ", html.unescape(text))


def _strip_url(text: str) -> str:
    return URL_IN_TEXT_RE.sub(" ", text)


def _strip_control(text: str) -> str:
    text = STRAY_ZWJ_RE.sub("", text)

    out: list[str] = []

    for ch in text:
        category = unicodedata.category(ch)
        if category == "Cc":
            # \n, \t, dsb. -> spasi
            out.append(" ")
        elif category == "Cf" and ch != "‍":
            # zero-width space/non-joiner, BOM, tanda arah teks
            continue
        else:
            out.append(ch)

    return "".join(out)


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _reduce_repeated_letters(text: str) -> str:
    return REPEATED_LETTER_RE.sub(r"\1\1", text)


def _normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def light_steps(
    nfkc: bool = True,
) -> list[tuple[str, callable]]:

    steps = [
        ("hapus_html", _strip_html),
        ("hapus_url", _strip_url),
        ("hapus_kontrol_zero_width", _strip_control),
    ]

    if nfkc:
        steps.append(("normalisasi_nfkc", _nfkc))

    steps += [
        ("kurangi_huruf_berulang", _reduce_repeated_letters),
        ("rapikan_spasi", _normalize_whitespace),
    ]

    return steps


def clean_light(
    text: str,
    nfkc: bool = True,
) -> str:

    for _, step in light_steps(nfkc):
        text = step(text)

    return text


def clean_light_series(
    series: pd.Series,
    nfkc: bool = True,
) -> tuple[pd.Series, dict[str, int]]:
    """Terapkan langkah berurutan; hitung baris yang berubah per langkah."""

    current = series.astype(str)
    changed: dict[str, int] = {}

    for name, step in light_steps(nfkc):
        updated = current.map(step)
        changed[name] = int((updated != current).sum())
        current = updated

    return current, changed


def normalize_fragment_key(
    text: str,
) -> str:
    """Kunci pencocokan fragmen (kandidat & kosakata tag aspek)."""

    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = _normalize_whitespace(text)

    return text.rstrip(":").strip()
