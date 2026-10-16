"""Ringkasan tabel untuk EDA sebelum/sesudah inferensi sentimen (pandas murni, deterministik).

Kolom probabilitas selalu dipilih berdasarkan NAMA (p_negatif, p_netral, p_positif).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

PROB_COLUMNS = ("p_negatif", "p_netral", "p_positif")       # urutan model
LABEL_NAMES = ("negatif", "netral", "positif")
URUTAN_NASKAH = ("p_positif", "p_netral", "p_negatif")       # S+, S~, S-

# Satu warna tetap per sentimen (palet tervalidasi; kuning wajib disertai legenda/label).
WARNA = {"negatif": "#e34948", "netral": "#eda100", "positif": "#2a78d6"}
WARNA_NETRAL_UI = "#52514e"

KATA_TUGAS = frozenset(
    "yang dan di ke dari ini itu untuk dengan ada juga sudah saya aku nya ya tapi atau karena "
    "jadi sih kak min lagi aja saja kalau kok pun buat dalam akan bisa udah sama".split()
)
TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


# ============================================================
# UMUM
# ============================================================

def pilih_prob(df: pd.DataFrame, urutan: Sequence[str] = PROB_COLUMNS) -> pd.DataFrame:
    """Ambil kolom probabilitas menurut nama; gagal jelas bila ada yang hilang."""
    missing = [c for c in urutan if c not in df.columns]
    if missing:
        raise KeyError(f"Kolom probabilitas tidak ada: {missing}")
    return df[list(urutan)].astype(float)


def as_bool(series: pd.Series) -> pd.Series:
    """bool dari kolom bool atau teks True/False (CSV)."""
    if series.dtype == bool:
        return series
    norm = series.astype(str).str.strip().str.lower()
    unknown = set(norm) - {"true", "false", "1", "0"}
    if unknown:
        raise ValueError(f"Nilai bukan boolean: {sorted(unknown)[:5]}")
    return norm.isin({"true", "1"})


def persen(n: float, total: float, digits: int = 2) -> float:
    return round(100.0 * n / total, digits) if total else float("nan")


class PenyimpanGrafik:
    """Simpan grafik berurutan: 01_nama.png, 02_nama.png, ... (dpi 150)."""

    def __init__(self, folder: Path, dpi: int = 150):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.nomor = 0
        self.berkas: list[Path] = []

    def simpan(self, fig, nama: str) -> Path:
        self.nomor += 1
        path = self.folder / f"{self.nomor:02d}_{nama}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        self.berkas.append(path)
        return path


# ============================================================
# EDA SEBELUM INFERENSI
# ============================================================

def corong_data(manifest: pd.DataFrame, reviews: pd.DataFrame, n_berkas_csv: int,
                min_ulasan: int = 3) -> pd.DataFrame:
    """Tahapan berkas -> baris -> layak; satu baris per tahap (kolom satuan: berkas/baris/produk)."""
    processed = manifest["status"].eq("diproses")
    include = as_bool(reviews["include_for_inference"])
    nonempty = ~as_bool(reviews["text_proposal_kosong"])
    usable = include & nonempty
    per_product = reviews.loc[usable, "product_id"].value_counts()
    big = per_product[per_product >= min_ulasan].index
    rows = [
        ("CSV data di folder mentah", "berkas", n_berkas_csv),
        ("Berkas dipetakan (status ok)", "berkas", int(len(manifest[~manifest["status"].str.startswith("dikecualikan")]))),
        ("Berkas diproses (ada kolom ulasan)", "berkas", int(processed.sum())),
        ("Baris sumber (rekaman CSV)", "baris", int(manifest.loc[processed, "n_baris_sumber"].sum())),
        ("Baris bersih (ulasan)", "baris", int(len(reviews))),
        ("include_for_inference", "baris", int(include.sum())),
        ("include & text_proposal tidak kosong", "baris", int(usable.sum())),
        (f"... pada produk >= {min_ulasan} ulasan", "baris",
         int(reviews.loc[usable, "product_id"].isin(big).sum())),
    ]
    return pd.DataFrame(rows, columns=["tahap", "satuan", "jumlah"])


def ringkasan_produk(reviews: pd.DataFrame) -> pd.DataFrame:
    """Per produk: toko, nama produk, jumlah ulasan, jumlah include."""
    include = as_bool(reviews["include_for_inference"])
    out = (reviews.assign(_inc=include)
           .groupby("product_id", sort=True)
           .agg(nama_toko=("nama_toko", "first"), nama_produk=("nama_produk", "first"),
                n_ulasan=("review_id", "size"), n_include=("_inc", "sum"))
           .reset_index())
    out["n_include"] = out["n_include"].astype("int64")
    return out.sort_values(["n_ulasan", "product_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def jumlah_kata(texts: pd.Series) -> pd.Series:
    return texts.fillna("").astype(str).str.split().str.len().fillna(0).astype("int64")


def kualitas_teks(reviews: pd.DataFrame, text_col: str = "text_proposal", short_words: int = 2) -> pd.DataFrame:
    """Proporsi masalah kualitas teks terhadap seluruh baris bersih."""
    n = len(reviews)
    texts = reviews[text_col].fillna("").astype(str)
    words = jumlah_kata(texts)
    kosong = as_bool(reviews["text_proposal_kosong"])
    items = [
        ("text_proposal kosong", int(kosong.sum())),
        (f"sangat pendek (1..{short_words} kata)", int((~kosong & (words <= short_words)).sum())),
        ("is_short (text_light < min_review_length)", int(as_bool(reviews["is_short"]).sum())),
        ("duplikat teknis (is_technical_duplicate)", int(as_bool(reviews["is_technical_duplicate"]).sum())),
        ("duplikat teks (is_text_duplicate)", int(as_bool(reviews["is_text_duplicate"]).sum())),
        ("balasan penjual dicurigai (seller_reply_suspect)", int(as_bool(reviews["seller_reply_suspect"]).sum())),
    ]
    return pd.DataFrame([(name, k, persen(k, n)) for name, k in items], columns=["indikator", "n", "persen"])


def tabel_terpotong(n_token: pd.Series, max_lengths: Iterable[int]) -> pd.DataFrame:
    """Persentase teks (non-NaN) yang melebihi tiap max_length (token termasuk [CLS]/[SEP])."""
    tokens = pd.Series(n_token).dropna().astype(int)
    rows = [(m, int((tokens > m).sum()), persen(int((tokens > m).sum()), len(tokens), 3)) for m in max_lengths]
    return pd.DataFrame(rows, columns=["max_length", "n_terpotong", "persen_terpotong"])


def parse_ts(series: pd.Series) -> pd.Series:
    """ISO 8601 dengan/tanpa jam (format campuran aman); kosong/tak terbaca -> NaT."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series.astype("string").str.strip().replace("", pd.NA), format="ISO8601", errors="coerce")


def ulasan_per_bulan(reviews: pd.DataFrame) -> pd.Series:
    ts = parse_ts(reviews["timestamp"])
    return ts.dropna().dt.to_period("M").value_counts().sort_index()


def ringkasan_waktu(reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(reviews)
    ts = parse_ts(reviews["timestamp"])
    has_time = as_bool(reviews["timestamp_has_time"])
    items = [("timestamp valid", int(ts.notna().sum())),
             ("timestamp dengan jam", int((ts.notna() & has_time).sum())),
             ("timestamp tanpa jam (tanggal saja)", int((ts.notna() & ~has_time).sum())),
             ("timestamp tidak valid (NaT)", int(ts.isna().sum()))]
    out = pd.DataFrame([(k, v, persen(v, n)) for k, v in items], columns=["kategori", "n", "persen"])
    status = reviews["timestamp_status"].value_counts().rename_axis("timestamp_status").reset_index(name="n")
    return out, status


def tokenisasi(text: str) -> list[str]:
    return [t.casefold() for t in TOKEN_RE.findall(str(text))]


def top_ngram(texts: Iterable[str], n: int = 1, k: int = 20,
              stopwords: frozenset[str] = KATA_TUGAS) -> pd.DataFrame:
    """Frekuensi kata (n=1) atau bigram (n=2); unigram tanpa kata tugas, bigram dari token utuh
    dengan membuang bigram yang kedua katanya kata tugas."""
    counts: Counter = Counter()
    for text in texts:
        tokens = tokenisasi(text)
        if n == 1:
            counts.update(t for t in tokens if t not in stopwords)
        else:
            grams = zip(*(tokens[i:] for i in range(n)))
            counts.update(" ".join(g) for g in grams if not all(w in stopwords for w in g))
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return pd.DataFrame(rows, columns=["ngram", "frekuensi"])


# ============================================================
# EDA SESUDAH INFERENSI
# ============================================================

def prob_maks(df: pd.DataFrame) -> pd.Series:
    return pilih_prob(df).max(axis=1)


def entropi(df: pd.DataFrame) -> pd.Series:
    """Entropi Shannon (nat) dari tiga probabilitas; maksimum ln 3 ~ 1,0986."""
    p = pilih_prob(df).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log(p), 0.0)
    out = terms.sum(axis=1)
    out[np.isnan(p).any(axis=1)] = np.nan
    return pd.Series(out, index=df.index, name="entropi")


def label_dari_prob(df: pd.DataFrame) -> pd.Series:
    """Argmax menurut nama kolom (NaN bila ada NaN)."""
    p = pilih_prob(df)
    ok = p.notna().all(axis=1)
    out = pd.Series(np.nan, index=df.index, dtype=object)
    out[ok] = p[ok].idxmax(axis=1).str.removeprefix("p_")
    return out


def kelas_panjang(words: pd.Series, batas: Sequence[int] = (2, 5, 10, 20, 50)) -> pd.Series:
    """Kelas panjang (kata): 1-2, 3-5, 6-10, 11-20, 21-50, >50."""
    edges = [0, *batas, np.inf]
    labels = [f"{a + 1}-{b}" for a, b in zip(edges[:-2], edges[1:-1])] + [f">{batas[-1]}"]
    return pd.cut(words, bins=edges, labels=labels, right=True)


def rata2_per_kelas(df: pd.DataFrame, kelas: pd.Series) -> pd.DataFrame:
    p = pilih_prob(df)
    out = p.groupby(kelas, observed=False).mean()
    out.insert(0, "n", kelas.value_counts().reindex(out.index).fillna(0).astype(int))
    return out


def rata2_per_produk(df: pd.DataFrame, min_ulasan: int = 3) -> pd.DataFrame:
    p = pilih_prob(df)
    g = pd.concat([df[["product_id", "nama_toko", "nama_produk"]], p], axis=1).groupby("product_id")
    out = g.agg(nama_toko=("nama_toko", "first"), nama_produk=("nama_produk", "first"),
                n=("p_negatif", "size"), p_positif=("p_positif", "mean"),
                p_netral=("p_netral", "mean"), p_negatif=("p_negatif", "mean")).reset_index()
    return out[out["n"] >= min_ulasan].reset_index(drop=True)


def matriks_konfusi(a: pd.Series, b: pd.Series, labels: Sequence[str] = LABEL_NAMES) -> pd.DataFrame:
    """Baris = versi a, kolom = versi b; hanya pasangan non-NaN."""
    ok = a.notna() & b.notna()
    tab = pd.crosstab(pd.Categorical(a[ok], categories=labels), pd.Categorical(b[ok], categories=labels),
                      dropna=False)
    tab.index.name, tab.columns.name = "lama", "baru"
    return tab
