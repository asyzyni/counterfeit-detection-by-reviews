from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from .config import CleaningConfig


# ============================================================
# PERAN KOLOM
# ============================================================

ROLE_REVIEW = "review_text"
ROLE_TIMESTAMP = "timestamp"
ROLE_RATING = "rating"
ROLE_ASPECT = "aspect_tag"
ROLE_URL = "url_media"
ROLE_METADATA = "metadata"
ROLE_SELLER = "seller_reply"
ROLE_OTHER = "lainnya"
ROLE_CHECK = "perlu_cek"


# ============================================================
# POLA NILAI (dipakai untuk profil kolom DAN penyaringan fragmen)
# ============================================================

DATE_ABS_RE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?\s*(?:\|.*)?$",
    re.S,
)

DATE_REL_RE = re.compile(
    r"^\s*(?:\d+\s*(?:detik|menit|jam|hari|minggu|bulan|tahun)\s*(?:yang\s*)?lalu"
    r"|kemarin|hari ini|baru saja)\s*$",
    re.I,
)

URL_RE = re.compile(
    r"^\s*(?:https?://|www\.)\S+\s*$",
    re.I,
)

MASKED_NAME_RE = re.compile(
    r"^\s*\S\*{3,}\S\s*$",
)

NULL_TEXT_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "<na>",
}

SELLER_MARKER_VALUES = {
    "respon penjual:",
    "respon penjual",
    "seller response",
    "balasan penjual:",
}

# Gaya bahasa balasan penjual (bukan opini pembeli).
SELLER_STYLE_RE = re.compile(
    r"pelanggan yang terhormat"
    r"|\b(?:toko|produk|customer service|cs|admin) kami\b"
    r"|\bkami (?:akan|mohon|minta|harap|berharap)\b"
    r"|(?:terima ?kasih|makasih|trimakasih|terimakasih)\b.{0,30}\batas\b.{0,20}"
    r"\b(?:ulasan|penilaian|review|rating|kepercayaan|dukungan|orderan)"
    r"|\b(?:sudah|telah) berbelanja\b",
    re.I | re.S,
)

# Nama pengguna tanpa sensor: satu token huruf/angka/._ yang memuat angka, '_' atau '.'.
HANDLE_RE = re.compile(r"^(?=.*[\d_.])[A-Za-z0-9._]+$")

# Petunjuk awal berbasis nama kolom (BUKAN keputusan akhir).
NAME_HINTS: list[tuple[str, str]] = [
    (r"^web_scraper_start_url$|src$", ROLE_URL),
    (r"^(timestamp|time|seller-time|XYk98l)$", ROLE_TIMESTAMP),
    (r"^(review-attribute|K5v3lN|skuInfo-label)$", ROLE_ASPECT),
    (
        r"^(web_scraper_order|pagination|skuInfo-value|seller-name"
        r"|item-content-like-content-text|rating-media-list__video-cover"
        r"|img-more-text|lzKjkR|jtmC98|T79_5U|rFzVcr|rating)$",
        ROLE_METADATA,
    ),
    (
        r"^(review\s*\d*|revwie|ulasan|data\d*|item-content-main-content-reviews-item"
        r"|meQyXP|f35Wh2|YNedDV|FtqM1T|QSiE2A|phone)$",
        ROLE_REVIEW,
    ),
]


# ============================================================
# KLASIFIKASI SATU NILAI
# ============================================================

def base_column_name(
    column: str,
) -> str:
    """'f35Wh2 (2)' -> 'f35Wh2'; 'review 2' tetap."""

    return re.sub(r"\s*\(\d+\)$", "", str(column).strip())


def name_hint(
    column: str,
) -> str:

    base = base_column_name(column)

    for pattern, role in NAME_HINTS:
        if re.search(pattern, base):
            return role

    return ""


def _has_letter(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def _has_symbol_emoji(text: str) -> bool:
    return any(unicodedata.category(ch) == "So" for ch in text)


def classify_value(
    value: object,
) -> str:
    """
    Jenis satu nilai sel:
    kosong | tanggal | tanggal_relatif | url | angka | label_aspek |
    nama_tersamar | penanda_penjual | simbol | teks
    """

    text = "" if value is None else str(value).strip()

    if text.lower() in NULL_TEXT_VALUES:
        return "kosong"

    if DATE_ABS_RE.match(text):
        return "tanggal"

    if DATE_REL_RE.match(text):
        return "tanggal_relatif"

    if URL_RE.match(text):
        return "url"

    if text.casefold() in SELLER_MARKER_VALUES:
        return "penanda_penjual"

    if MASKED_NAME_RE.match(text):
        return "nama_tersamar"

    if not _has_letter(text):
        if any(ch.isdigit() for ch in text):
            return "angka"
        if _has_symbol_emoji(text):
            return "teks"
        return "simbol"

    # "Kualitas:", "⚡Kinerja:", "Kapasitas Penyimpanan:"
    if text.endswith(":") and len(text.split()) <= 4:
        return "label_aspek"

    return "teks"


# ============================================================
# PROFIL KOLOM
# ============================================================

@dataclass
class ColumnProfile:
    file: str
    column: str
    column_base: str
    position: int
    n_rows: int
    n_nonempty: int
    fill_rate: float
    date_rate: float
    rel_date_rate: float
    numeric_rate: float
    url_rate: float
    label_rate: float
    masked_name_rate: float
    seller_marker_rate: float
    text_rate: float
    seller_style_rate: float
    handle_rate: float
    one_word_rate: float
    int_1_5_rate: float
    mean_chars: float
    mean_words: float
    unique_ratio: float
    n_unique: int
    samples: str
    name_hint: str
    role: str = ""
    confidence: str = ""
    evidence: str = ""


def profile_column(
    file: str,
    column: str,
    position: int,
    values: pd.Series,
    n_samples: int = 5,
) -> ColumnProfile:

    stripped = values.astype(str).str.strip()
    kinds = stripped.map(classify_value)
    nonempty = stripped[kinds != "kosong"]
    nonempty_kinds = kinds[kinds != "kosong"]

    n_rows = len(values)
    n_nonempty = len(nonempty)

    def rate(*names: str) -> float:
        if n_nonempty == 0:
            return 0.0
        return float(nonempty_kinds.isin(names).mean())

    if n_nonempty:
        mean_chars = float(nonempty.str.len().mean())
        mean_words = float(nonempty.str.split().str.len().mean())
        n_unique = int(nonempty.nunique())
        unique_ratio = n_unique / n_nonempty
        seller_style_rate = float(nonempty.map(lambda v: bool(SELLER_STYLE_RE.search(v))).mean())
        handle_rate = float(nonempty.str.match(HANDLE_RE).mean())
        one_word_rate = float((nonempty.str.split().str.len() == 1).mean())
        int_1_5_rate = float(nonempty.str.fullmatch(r"[1-5]").mean())
    else:
        mean_chars = mean_words = unique_ratio = 0.0
        seller_style_rate = handle_rate = one_word_rate = int_1_5_rate = 0.0
        n_unique = 0

    # Contoh: nilai unik pertama sesuai urutan baris (deterministik).
    samples = " ‖ ".join(
        value.replace("\n", " ")[:80]
        for value in pd.unique(nonempty)[:n_samples]
    )

    return ColumnProfile(
        file=file,
        column=column,
        column_base=base_column_name(column),
        position=position,
        n_rows=n_rows,
        n_nonempty=n_nonempty,
        fill_rate=n_nonempty / n_rows if n_rows else 0.0,
        date_rate=rate("tanggal", "tanggal_relatif"),
        rel_date_rate=rate("tanggal_relatif"),
        numeric_rate=rate("angka"),
        url_rate=rate("url"),
        label_rate=rate("label_aspek"),
        masked_name_rate=rate("nama_tersamar"),
        seller_marker_rate=rate("penanda_penjual"),
        text_rate=rate("teks"),
        seller_style_rate=seller_style_rate,
        handle_rate=handle_rate,
        one_word_rate=one_word_rate,
        int_1_5_rate=int_1_5_rate,
        mean_chars=mean_chars,
        mean_words=mean_words,
        unique_ratio=unique_ratio,
        n_unique=n_unique,
        samples=samples,
        name_hint=name_hint(column),
    )


# ============================================================
# PENETAPAN PERAN
# ============================================================

def _rates_text(p: ColumnProfile) -> str:
    return (
        f"isi {p.fill_rate:.0%}; teks {p.text_rate:.0%}; "
        f"tanggal {p.date_rate:.0%}; url {p.url_rate:.0%}; "
        f"angka {p.numeric_rate:.0%}; label {p.label_rate:.0%}; "
        f"nama-tersamar {p.masked_name_rate:.0%}; "
        f"rata2 {p.mean_words:.1f} kata; unik {p.unique_ratio:.0%}"
    )


def assign_role(
    p: ColumnProfile,
    seller_cooccurrence: float | None,
    config: CleaningConfig,
) -> tuple[str, str, str]:
    """Kembalikan (peran, keyakinan, bukti) dari profil nilai."""

    strong = config.role_strong_rate
    weak = config.role_weak_rate
    rates = _rates_text(p)

    if p.n_nonempty == 0:
        return ROLE_OTHER, "tinggi", "kolom kosong seluruhnya"

    if p.url_rate >= strong:
        return ROLE_URL, "tinggi", f"{rates} -> nilai berupa URL/media"

    if p.date_rate >= strong:
        return ROLE_TIMESTAMP, "tinggi", f"{rates} -> nilai berupa tanggal"

    if p.numeric_rate >= strong:
        if p.int_1_5_rate >= strong:
            return ROLE_RATING, "tinggi", f"{rates} -> bilangan bulat 1..5"
        return (
            ROLE_METADATA,
            "tinggi",
            f"{rates} -> angka bukan skala 1..5 (hitungan/durasi/jumlah media)",
        )

    if p.date_rate >= 0.5 and p.text_rate == 0:
        return (
            ROLE_TIMESTAMP,
            "sedang",
            f"{rates} -> mayoritas tanggal; sisanya tak terparsing -> NaT",
        )

    if p.seller_marker_rate >= strong:
        return ROLE_METADATA, "tinggi", f"{rates} -> penanda balasan penjual"

    if p.label_rate >= strong:
        return ROLE_ASPECT, "tinggi", f"{rates} -> label berakhiran ':'"

    if p.masked_name_rate >= config.masked_name_rate and p.mean_words <= 3.5:
        return (
            ROLE_METADATA,
            "tinggi",
            f"{rates} -> nama pengguna (pola tersamar 'x***y', pendek)",
        )

    if (
        p.n_nonempty >= 5
        and p.one_word_rate >= strong
        and p.unique_ratio >= 0.7
        and p.handle_rate >= weak
    ):
        return (
            ROLE_METADATA,
            "tinggi",
            f"{rates}; {p.handle_rate:.0%} berpola akun (huruf+angka/_/.), satu kata, "
            f"hampir semua unik -> nama pengguna",
        )

    if p.n_nonempty >= 10 and p.n_unique <= 2 and p.mean_words <= 3:
        return ROLE_METADATA, "tinggi", f"{rates} -> nilai konstan (label UI)"

    if p.name_hint == ROLE_METADATA and p.mean_words <= 4:
        return (
            ROLE_METADATA,
            "sedang",
            f"{rates}; nama kolom petunjuk metadata & nilai pendek",
        )

    if seller_cooccurrence is not None:
        if seller_cooccurrence >= config.seller_reply_cooccurrence:
            return (
                ROLE_SELLER,
                "tinggi",
                f"{rates}; {seller_cooccurrence:.0%} baris terisi bersamaan "
                f"dengan penanda balasan penjual -> balasan penjual",
            )
        if seller_cooccurrence >= weak and p.seller_style_rate > 0:
            return (
                ROLE_CHECK,
                "rendah",
                f"{rates}; {seller_cooccurrence:.0%} baris bersamaan dengan "
                f"penanda balasan penjual -> mungkin campuran balasan penjual",
            )

    if p.seller_style_rate >= 0.5:
        return (
            ROLE_SELLER,
            "sedang",
            f"{rates}; {p.seller_style_rate:.0%} bergaya balasan penjual "
            f"('toko kami', 'terima kasih atas penilaian', 'Halo kak') -> balasan penjual",
        )

    if p.seller_style_rate >= 0.2:
        return (
            ROLE_CHECK,
            "rendah",
            f"{rates}; {p.seller_style_rate:.0%} bergaya balasan penjual "
            f"-> campuran pembeli/penjual",
        )

    if p.name_hint == ROLE_ASPECT and p.label_rate >= weak:
        return ROLE_ASPECT, "sedang", f"{rates}; nama kolom petunjuk label aspek"

    if p.text_rate >= strong:
        return ROLE_REVIEW, "tinggi", f"{rates} -> teks bebas"

    if p.text_rate >= 0.5:
        return (
            ROLE_REVIEW,
            "sedang",
            f"{rates} -> mayoritas teks bebas; nilai non-teks disaring per fragmen",
        )

    if p.text_rate >= 0.2 and p.text_rate + p.label_rate >= strong:
        return (
            ROLE_REVIEW,
            "sedang",
            f"{rates} -> campuran label aspek + teks; label dipisah per fragmen",
        )

    return ROLE_CHECK, "rendah", f"{rates} -> profil tidak dominan"


def _seller_marker_rows(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    config: CleaningConfig,
) -> pd.Series | None:

    marker_cols = [
        p.column for p in profiles
        if p.n_nonempty > 0 and p.seller_marker_rate >= config.role_strong_rate
    ]

    if not marker_cols:
        return None

    mask = pd.Series(False, index=df.index)
    for col in marker_cols:
        mask |= df[col].astype(str).str.strip() != ""

    return mask


def map_file_columns(
    file: str,
    df: pd.DataFrame,
    config: CleaningConfig,
) -> list[ColumnProfile]:
    """Profil + peran untuk tiap kolom satu file."""

    profiles = [
        profile_column(
            file=file,
            column=col,
            position=position,
            values=df[col],
            n_samples=config.n_profile_samples,
        )
        for position, col in enumerate(df.columns)
    ]

    marker_rows = _seller_marker_rows(df, profiles, config)

    for p in profiles:

        cooccurrence: float | None = None

        if marker_rows is not None and p.n_nonempty > 0:
            filled = df[p.column].astype(str).str.strip() != ""
            cooccurrence = float((filled & marker_rows).sum() / filled.sum())

        p.role, p.confidence, p.evidence = assign_role(p, cooccurrence, config)

    # --------------------------------------------------------
    # Satu kolom timestamp per file: isi terbanyak, lalu yang
    # paling kiri. Kolom tanggal lain -> metadata.
    # --------------------------------------------------------

    ts_profiles = [p for p in profiles if p.role == ROLE_TIMESTAMP]

    if len(ts_profiles) > 1:
        chosen = min(ts_profiles, key=lambda p: (-p.n_nonempty, p.position))
        for p in ts_profiles:
            if p is not chosen:
                p.role = ROLE_METADATA
                p.evidence += (
                    f"; tanggal sekunder (timestamp utama: '{chosen.column}')"
                )

    return profiles


def profiles_to_frame(
    profiles: list[ColumnProfile],
) -> pd.DataFrame:

    frame = pd.DataFrame([vars(p) for p in profiles])

    float_cols = frame.select_dtypes("float").columns
    frame[float_cols] = frame[float_cols].round(4)

    return frame


def role_summary(
    column_map: pd.DataFrame,
) -> Counter:

    return Counter(column_map["role"])
