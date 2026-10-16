"""Ekspor sekuens sentimen per produk (masukan HMM, Tabel 3.2 proposal) dan jejak auditnya.

Contoh:
    python -m src.export_sekuens --input "outputs/sentiment/sentimen_ulasan_v2.csv" \
        --out-sekuens "outputs/sentiment/sekuens_produk.csv" \
        --out-audit "outputs/audit/sekuens_audit.csv" --min-ulasan 3

Keluaran:
    (A) --out-sekuens : product_id, nama_toko, nama_produk, timestamp, seq, p_positif, p_netral, p_negatif
                        (tanpa review_id; kunci ke B adalah (product_id, seq) = (product_id, seq_sekuens))
    (B) --out-audit   : SEMUA baris input (urutan input) + product_id_lama, masuk_sekuens, alasan_keluar, seq_sekuens
    dan di folder --out-audit: ekspor_sekuens_dikeluarkan.csv, ekspor_sekuens_laporan.md,
    ekspor_sekuens_ringkasan.json, CHECKSUMS.txt. File yang sudah ada TIDAK ditimpa.

Input dibaca sebagai teks (dtype=str, tanpa konversi NA) sehingga nilai kolom input, termasuk
probabilitas, disalin apa adanya ke A dan B. Kolom probabilitas selalu dipilih menurut NAMA.
label_argmax hanya ikut tersalin ke B dan tidak dipakai di sini.

product_id baru = nama_toko + "_" + nama_produk (huruf asli, hanya strip spasi di ujung), mengikuti pola
nama file sumber "toko_produk". product_id lama disimpan di B sebagai product_id_lama. Sebelum apa pun
ditulis, pemetaan lama <-> baru wajib 1-ke-1; bila tidak, ekspor berhenti dan menampilkan konfliknya.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shlex
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_STOPPED = 2

CSV_ENCODING = "utf-8-sig"
LINE_TERMINATOR = "\n"

PROB_COLUMNS_SEKUENS = ["p_positif", "p_netral", "p_negatif"]  # urutan proposal: S+, S~, S-
SEKUENS_COLUMNS = ["product_id", "nama_toko", "nama_produk", "timestamp", "seq", *PROB_COLUMNS_SEKUENS]
REQUIRED_COLUMNS = [
    "review_id", "product_id", "nama_toko", "nama_produk", "timestamp", "timestamp_status",
    "timestamp_has_time", "row_order", "include_for_inference", "text_proposal_kosong",
    "is_text_duplicate", *PROB_COLUMNS_SEKUENS,
]
AUDIT_EXTRA_COLUMNS = ["masuk_sekuens", "alasan_keluar", "seq_sekuens"]
PRODUCT_ID_LAMA = "product_id_lama"
PEMISAH_ID = "_"
DIKELUARKAN_COLUMNS = [
    "review_id", "product_id", PRODUCT_ID_LAMA, "alasan_keluar", "timestamp_status", "text_proposal_kosong", "include_for_inference",
]

# Prioritas alasan keluar (satu alasan utama per baris).
ALASAN_TEKS_KOSONG = "teks_kosong"
ALASAN_BUKAN_INCLUDE = "bukan_include_for_inference"
ALASAN_TIMESTAMP = "timestamp_tidak_valid"
ALASAN_PRODUK_KECIL = "produk_kurang_dari_3_ulasan"
ALASAN_URUTAN = [ALASAN_TEKS_KOSONG, ALASAN_BUKAN_INCLUDE, ALASAN_TIMESTAMP, ALASAN_PRODUK_KECIL]

NAMA_DIKELUARKAN = "ekspor_sekuens_dikeluarkan.csv"
NAMA_LAPORAN = "ekspor_sekuens_laporan.md"
NAMA_RINGKASAN = "ekspor_sekuens_ringkasan.json"
NAMA_CHECKSUMS = "CHECKSUMS.txt"

PROB_SUM_TOL = 1e-6

_TRUE = {"true", "1"}
_FALSE = {"false", "0"}


class EksporError(ValueError):
    """Input tidak memenuhi syarat ekspor."""


class KonflikPemetaanError(EksporError):
    """Pemetaan product_id lama <-> baru tidak 1-ke-1."""

    def __init__(self, konflik: dict):
        self.konflik = konflik
        super().__init__(format_konflik(konflik))


# ------------------------------------------------------------
# Fungsi murni
# ------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_input_bytes(data: bytes) -> pd.DataFrame:
    """Baca CSV sebagai teks apa adanya (tanpa konversi NA/angka)."""
    return pd.read_csv(io.BytesIO(data), encoding=CSV_ENCODING, dtype=str, keep_default_na=False)


def to_bool(series: pd.Series, name: str) -> pd.Series:
    """Ubah kolom teks True/False (atau 1/0) menjadi bool; nilai lain ditolak."""
    norm = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(norm) - _TRUE - _FALSE)
    if unknown:
        raise EksporError(f"Kolom {name!r} berisi nilai bukan boolean: {unknown[:5]}")
    return norm.isin(_TRUE)


def parse_timestamp(series: pd.Series) -> pd.Series:
    """ISO 8601 (dengan atau tanpa jam); kosong atau tak terbaca -> NaT."""
    return pd.to_datetime(series.astype(str).str.strip().replace("", None), format="ISO8601", errors="coerce")


def bentuk_product_id(df: pd.DataFrame) -> pd.Series:
    """product_id baru: nama_toko + "_" + nama_produk, huruf asli, strip spasi di ujung saja."""
    return df["nama_toko"].astype(str).str.strip() + PEMISAH_ID + df["nama_produk"].astype(str).str.strip()


def periksa_pemetaan(df: pd.DataFrame) -> dict:
    """Bandingkan partisi product_id lama dengan partisi product_id baru (tanpa memilih atau menggabung).

    Hasil: jumlah ID, konflik lama->banyak ejaan (nama_toko, nama_produk), konflik baru<-banyak ID lama,
    jumlah nama kosong, dan contoh pemetaan. Konflik berisi jumlah baris tiap varian.
    """
    peta = pd.DataFrame({
        "lama": df["product_id"].astype(str),
        "baru": bentuk_product_id(df),
        "nama_toko": df["nama_toko"].astype(str),
        "nama_produk": df["nama_produk"].astype(str),
    })
    varian = peta.groupby(["lama", "nama_toko", "nama_produk"], sort=True).size().rename("baris").reset_index()
    lama_banyak = varian[varian["lama"].map(varian["lama"].value_counts()) > 1]
    pasangan = peta.groupby(["baru", "lama"], sort=True).size().rename("baris").reset_index()
    baru_banyak = pasangan[pasangan["baru"].map(pasangan["baru"].value_counts()) > 1]
    contoh = peta.drop_duplicates("lama").head(3)
    return {
        "n_produk_lama": int(peta["lama"].nunique()),
        "n_produk_baru": int(peta["baru"].nunique()),
        "n_pasangan_lama_baru": int(len(pasangan)),
        "lama_banyak_ejaan": lama_banyak.to_dict("records"),
        "baru_dari_banyak_lama": baru_banyak.to_dict("records"),
        "n_nama_toko_kosong": int((df["nama_toko"].astype(str).str.strip() == "").sum()),
        "n_nama_produk_kosong": int((df["nama_produk"].astype(str).str.strip() == "").sum()),
        "contoh": [{"lama": r.lama, "baru": r.baru} for r in contoh.itertuples()],
    }


def pemetaan_valid(peta: dict) -> bool:
    return (
        not peta["lama_banyak_ejaan"]
        and not peta["baru_dari_banyak_lama"]
        and peta["n_produk_lama"] == peta["n_produk_baru"] == peta["n_pasangan_lama_baru"]
        and peta["n_nama_toko_kosong"] == 0
        and peta["n_nama_produk_kosong"] == 0
    )


def format_konflik(peta: dict) -> str:
    lines = [f"Pemetaan product_id lama <-> baru tidak 1-ke-1 (lama={peta['n_produk_lama']}, baru={peta['n_produk_baru']})."]
    if peta["lama_banyak_ejaan"]:
        lines.append("product_id lama dengan lebih dari satu ejaan (nama_toko, nama_produk):")
        lines += [f"  {r['lama']!r}: ({r['nama_toko']!r}, {r['nama_produk']!r}) {r['baris']} baris" for r in peta["lama_banyak_ejaan"]]
    if peta["baru_dari_banyak_lama"]:
        lines.append("product_id baru yang berasal dari lebih dari satu product_id lama:")
        lines += [f"  {r['baru']!r} <- {r['lama']!r}: {r['baris']} baris" for r in peta["baru_dari_banyak_lama"]]
    if peta["n_nama_toko_kosong"] or peta["n_nama_produk_kosong"]:
        lines.append(f"nama_toko kosong: {peta['n_nama_toko_kosong']} baris; nama_produk kosong: {peta['n_nama_produk_kosong']} baris")
    return "\n".join(lines)


def ganti_product_id(df: pd.DataFrame) -> pd.DataFrame:
    """product_id berisi ID baru (posisi kolom tetap); ID lama disisipkan tepat sesudahnya sebagai product_id_lama."""
    if PRODUCT_ID_LAMA in df.columns:
        raise EksporError(f"Input sudah punya kolom {PRODUCT_ID_LAMA!r}")
    out = df.copy()
    pos = list(out.columns).index("product_id")
    out.insert(pos + 1, PRODUCT_ID_LAMA, out["product_id"])
    out["product_id"] = bentuk_product_id(df)
    return out


def check_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def tentukan_alasan(df: pd.DataFrame, min_ulasan: int = 3) -> pd.Series:
    """Satu alasan_keluar per baris menurut prioritas; string kosong bila baris masuk sekuens."""
    alasan = pd.Series("", index=df.index, dtype=object)
    kosong = to_bool(df["text_proposal_kosong"], "text_proposal_kosong")
    include = to_bool(df["include_for_inference"], "include_for_inference")
    ts_ok = df["timestamp_status"].astype(str).str.strip().isin(
        ("ok", "relatif_dikonversi")
    ) & parse_timestamp(df["timestamp"]).notna()

    alasan[kosong] = ALASAN_TEKS_KOSONG
    alasan[(alasan == "") & ~include] = ALASAN_BUKAN_INCLUDE
    alasan[(alasan == "") & ~ts_ok] = ALASAN_TIMESTAMP

    lolos = alasan == ""
    n_per_produk = df.loc[lolos, "product_id"].map(df.loc[lolos, "product_id"].value_counts())
    alasan[n_per_produk.index[n_per_produk < min_ulasan]] = ALASAN_PRODUK_KECIL
    return alasan


def urutkan_sekuens(df: pd.DataFrame) -> pd.Index:
    """Timestamp, source_file, row_order; review_id terakhir untuk input lama."""
    kunci = pd.DataFrame(
        {
            "product_id": df["product_id"].astype(str),
            "ts": parse_timestamp(df["timestamp"]),
            "row_order": pd.to_numeric(df["row_order"], errors="raise"),
            "review_id": df["review_id"].astype(str),
        },
        index=df.index,
    )
    if "source_file" in df:
        kunci["source_file"] = df["source_file"].astype(str)
    order = ["product_id", "ts"] + (["source_file"] if "source_file" in kunci else [])
    return kunci.sort_values(order + ["row_order", "review_id"], kind="mergesort").index


def bangun_audit(df: pd.DataFrame, min_ulasan: int = 3) -> pd.DataFrame:
    """B: semua baris input (urutan input) + masuk_sekuens, alasan_keluar, seq_sekuens."""
    if not df["review_id"].is_unique:
        raise EksporError("review_id tidak unik")
    alasan = tentukan_alasan(df, min_ulasan)
    masuk = alasan == ""
    urut = urutkan_sekuens(df.loc[masuk])
    seq = pd.Series("", index=df.index, dtype=object)
    seq_masuk = df.loc[urut, "product_id"].groupby(df.loc[urut, "product_id"], sort=False).cumcount() + 1
    seq[urut] = seq_masuk.astype(str)

    audit = df.copy()
    audit["masuk_sekuens"] = masuk.map({True: "True", False: "False"})
    audit["alasan_keluar"] = alasan
    audit["seq_sekuens"] = seq
    return audit


def bangun_sekuens(audit: pd.DataFrame) -> pd.DataFrame:
    """A: hanya baris masuk_sekuens True dari B, terurut per produk lalu seq."""
    masuk = audit[audit["masuk_sekuens"] == "True"]
    urut = urutkan_sekuens(masuk)  # review_id hanya dipakai internal sebagai pemecah seri
    out = masuk.loc[urut].copy()
    out["seq"] = out["seq_sekuens"]
    return out[SEKUENS_COLUMNS].reset_index(drop=True)


def bangun_dikeluarkan(audit: pd.DataFrame) -> pd.DataFrame:
    return audit.loc[audit["masuk_sekuens"] == "False", DIKELUARKAN_COLUMNS].reset_index(drop=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator=LINE_TERMINATOR)
    return buf.getvalue().encode(CSV_ENCODING)


def hasilkan_berkas(input_bytes: bytes, min_ulasan: int = 3) -> dict[str, bytes]:
    """Bytes deterministik untuk A, B, dan daftar dikeluarkan."""
    df = read_input_bytes(input_bytes)
    missing = check_columns(df)
    if missing:
        raise EksporError(f"Kolom wajib tidak ada di input: {missing}")
    peta = periksa_pemetaan(df)
    if not pemetaan_valid(peta):
        raise KonflikPemetaanError(peta)
    audit = bangun_audit(ganti_product_id(df), min_ulasan)
    return {
        "sekuens": to_csv_bytes(bangun_sekuens(audit)),
        "audit": to_csv_bytes(audit),
        "dikeluarkan": to_csv_bytes(bangun_dikeluarkan(audit)),
    }


def periksa_invarian(input_df: pd.DataFrame, berkas: dict[str, bytes], min_ulasan: int = 3) -> list[dict]:
    """Periksa invarian dari bytes yang AKAN ditulis (dibaca ulang dari CSV)."""
    a = read_input_bytes(berkas["sekuens"])
    b = read_input_bytes(berkas["audit"])
    d = read_input_bytes(berkas["dikeluarkan"])
    hasil: list[dict] = []

    def cek(nama: str, ok: bool, bukti: str) -> None:
        hasil.append({"invarian": nama, "lolos": bool(ok), "bukti": bukti})

    cek("baris_B_sama_input", len(b) == len(input_df), f"B={len(b)}, input={len(input_df)}")
    cek("review_id_unik_B", b["review_id"].is_unique, f"unik={b['review_id'].nunique()} dari {len(b)}")
    input_cols = list(input_df.columns)
    pos = input_cols.index("product_id")
    kolom_b = input_cols[: pos + 1] + [PRODUCT_ID_LAMA] + input_cols[pos + 1:] + AUDIT_EXTRA_COLUMNS
    lain = [c for c in input_cols if c != "product_id"]
    cek(
        "B_kolom_input_identik",
        list(b.columns) == kolom_b
        and b[lain].equals(input_df[lain])
        and b[PRODUCT_ID_LAMA].equals(input_df["product_id"].rename(PRODUCT_ID_LAMA)),
        f"{len(input_cols)} kolom input (product_id lama -> {PRODUCT_ID_LAMA}) + product_id baru + "
        f"{AUDIT_EXTRA_COLUMNS}; nilai dan urutan baris dibandingkan sel per sel",
    )

    id_baru = b["product_id"]
    cek("product_id_baru_terisi", (id_baru.str.strip() != "").all() and id_baru.equals(bentuk_product_id(b)),
        f"kosong={int((id_baru.str.strip() == '').sum())}; product_id == nama_toko + '_' + nama_produk di semua baris")
    peta = b.groupby([PRODUCT_ID_LAMA, "product_id"]).size()
    n_lama, n_baru = b[PRODUCT_ID_LAMA].nunique(), id_baru.nunique()
    cek("pemetaan_1_ke_1", len(peta) == n_lama == n_baru,
        f"product_id lama={n_lama}, baru={n_baru}, pasangan unik={len(peta)}")

    # Hitung ulang dengan pengelompokan product_id lama: alasan dan seq harus identik per baris.
    lama = bangun_audit(input_df, min_ulasan)
    cek("seq_sama_dengan_pengelompokan_lama",
        lama["seq_sekuens"].equals(b["seq_sekuens"]) and lama["alasan_keluar"].equals(b["alasan_keluar"]),
        f"seq_sekuens dan alasan_keluar dibandingkan pada {len(b)} baris (review_id sama, urutan input)")

    cek("kolom_A_persis", list(a.columns) == SEKUENS_COLUMNS and "review_id" not in a.columns, f"{list(a.columns)}")

    kunci = ["product_id", "seq"]
    b_masuk = b[b["masuk_sekuens"] == "True"].rename(columns={"seq_sekuens": "seq"})
    unik_a = not a.duplicated(kunci).any()
    unik_b = not b_masuk.duplicated(kunci).any()
    cek("kunci_product_id_seq_unik", unik_a and unik_b,
        f"duplikat (product_id, seq) di A={int(a.duplicated(kunci).sum())}, di B masuk={int(b_masuk.duplicated(kunci).sum())}")
    detail_ok = False
    if unik_a and unik_b and len(a) == len(b_masuk):
        a_idx = a.set_index(kunci).sort_index()
        bm = b_masuk.set_index(kunci).sort_index()
        detail_ok = a_idx.index.equals(bm.index) and a_idx.equals(bm[a_idx.columns])
    cek("A_sama_B_masuk", detail_ok,
        f"A={len(a)}, B masuk={len(b_masuk)}; dicocokkan lewat (product_id, seq) vs (product_id, seq_sekuens); "
        "probabilitas (teks), nama, dan timestamp identik")

    seq = pd.to_numeric(a["seq"], errors="coerce")
    per_produk = seq.groupby(a["product_id"])
    berurutan = all(
        np.array_equal(np.sort(s.to_numpy()), np.arange(1, len(s) + 1)) for _, s in per_produk
    ) and not seq.isna().any()
    urut_naik = per_produk.apply(lambda s: s.is_monotonic_increasing).all() if len(a) else True
    cek("seq_berurutan_1_sampai_T", berurutan and urut_naik, f"{a['product_id'].nunique()} produk diperiksa")
    t = a["product_id"].value_counts()
    t_min = int(t.min()) if len(t) else 0
    cek("T_minimal", len(t) == 0 or t_min >= min_ulasan, f"T min={t_min}, ambang={min_ulasan}")

    p = a[PROB_COLUMNS_SEKUENS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = bool(np.isfinite(p).all())
    dev = float(np.abs(p.sum(axis=1) - 1.0).max()) if len(p) and finite else (0.0 if finite else float("nan"))
    cek("p_tanpa_NaN_Inf", finite, f"sel non-finite={int((~np.isfinite(p)).sum())}")
    cek("p_jumlah_1", finite and dev <= PROB_SUM_TOL, f"deviasi maks={dev:.3e} (batas {PROB_SUM_TOL:g})")

    keluar = b["masuk_sekuens"] == "False"
    alasan_ok = b.loc[keluar, "alasan_keluar"].isin(ALASAN_URUTAN).all() and (b.loc[~keluar, "alasan_keluar"] == "").all()
    seq_ok = (b.loc[keluar, "seq_sekuens"] == "").all() and (b.loc[~keluar, "seq_sekuens"] != "").all()
    cek("satu_alasan_per_baris", alasan_ok and seq_ok and b["masuk_sekuens"].isin(["True", "False"]).all(),
        f"keluar={int(keluar.sum())} (alasan dari {ALASAN_URUTAN}), masuk={int((~keluar).sum())} (alasan kosong)")
    cek("dikeluarkan_sama_B_false", len(d) == int(keluar.sum()) and list(d["review_id"]) == list(b.loc[keluar, "review_id"]),
        f"dikeluarkan={len(d)}, B masuk_sekuens False={int(keluar.sum())}")
    return hasil


def ringkas(input_df: pd.DataFrame, berkas: dict[str, bytes]) -> dict:
    a = read_input_bytes(berkas["sekuens"])
    b = read_input_bytes(berkas["audit"])
    t = a["product_id"].value_counts()
    vc = b["alasan_keluar"].value_counts()
    b_masuk = b[b["masuk_sekuens"] == "True"]
    produk_hilang = sorted(set(b["product_id"]) - set(a["product_id"]))
    peta = periksa_pemetaan(input_df)
    return {
        "bentuk_product_id": {
            "lama": "kolom product_id input (huruf kecil, pembatas '|'), disimpan di B sebagai product_id_lama",
            "baru": "nama_toko + '_' + nama_produk (huruf asli, strip spasi di ujung saja)",
        },
        "pemetaan": {k: v for k, v in peta.items()},
        "n_baris_input": int(len(input_df)),
        "n_baris_sekuens": int(len(a)),
        "n_baris_dikeluarkan": int((b["masuk_sekuens"] == "False").sum()),
        "baris_per_alasan_keluar": {k: int(vc.get(k, 0)) for k in ALASAN_URUTAN},
        "n_produk_input": int(b["product_id"].nunique()),
        "n_produk_sekuens": int(a["product_id"].nunique()),
        "n_produk_hilang": int(len(produk_hilang)),
        "produk_hilang": produk_hilang,
        "panjang_sekuens_T": {
            "min": int(t.min()) if len(t) else None,
            "median": float(t.median()) if len(t) else None,
            "p95": float(np.percentile(t.to_numpy(), 95)) if len(t) else None,
            "maks": int(t.max()) if len(t) else None,
        },
        "n_sekuens_tanpa_jam": int((b_masuk["timestamp_has_time"].str.strip().str.lower() == "false").sum()),
        "n_sekuens_is_text_duplicate": int((b_masuk["is_text_duplicate"].str.strip().str.lower() == "true").sum()),
    }


def cek_reproduksibilitas(input_path: Path, berkas: dict[str, bytes], min_ulasan: int) -> dict:
    """Jalankan ulang dari disk, tulis ke folder sementara (di luar outputs/), bandingkan hash."""
    with tempfile.TemporaryDirectory(prefix="ekspor_sekuens_repro_") as tmp:
        kedua = hasilkan_berkas(input_path.read_bytes(), min_ulasan)
        detail = {}
        for key, data in kedua.items():
            path = Path(tmp) / f"{key}.csv"
            path.write_bytes(data)
            detail[key] = {"run1": sha256_bytes(berkas[key]), "run2": sha256_file(path)}
    return {"folder_sementara": tmp, "hash": detail,
            "identik": all(v["run1"] == v["run2"] for v in detail.values())}


def _md_sel(teks: str) -> str:
    """Escape '|' agar tidak memecah sel tabel markdown."""
    return str(teks).replace("|", "\\|")


def format_laporan(ctx: dict) -> str:
    r = ctx["ringkasan"]
    t = r["panjang_sekuens_T"]
    lines = [
        "# Laporan ekspor sekuens produk",
        "",
        f"Dibuat: {ctx['waktu']}",
        "",
        "## Perintah",
        "",
        "```",
        ctx["perintah"],
        "```",
        "",
        "## Jumlah baris",
        "",
        f"- Input: {r['n_baris_input']:,}",
        f"- Masuk sekuens (A): {r['n_baris_sekuens']:,}",
        f"- Dikeluarkan: {r['n_baris_dikeluarkan']:,}",
        "",
        "| alasan_keluar (prioritas) | baris |",
        "|---|---:|",
    ]
    lines += [f"| {i}. `{k}` | {v:,} |" for i, (k, v) in enumerate(r["baris_per_alasan_keluar"].items(), 1)]
    lines += [
        "",
        "## Produk",
        "",
        f"- Sebelum (input): {r['n_produk_input']}",
        f"- Sesudah (A): {r['n_produk_sekuens']}",
        f"- Tidak punya sekuens: {r['n_produk_hilang']}"
        + (f" ({', '.join(f'`{x}`' for x in r['produk_hilang'])})" if r["produk_hilang"] else ""),
        "",
        "## Bentuk product_id (berubah)",
        "",
        f"- Lama: {r['bentuk_product_id']['lama']}",
        f"- Baru: {r['bentuk_product_id']['baru']}",
        f"- Produk lama: {r['pemetaan']['n_produk_lama']}, produk baru: {r['pemetaan']['n_produk_baru']}, "
        f"pasangan unik lama-baru: {r['pemetaan']['n_pasangan_lama_baru']}",
        f"- product_id lama dengan lebih dari satu ejaan: {len(r['pemetaan']['lama_banyak_ejaan'])}; "
        f"product_id baru dari lebih dari satu ID lama: {len(r['pemetaan']['baru_dari_banyak_lama'])}",
        f"- nama_toko kosong: {r['pemetaan']['n_nama_toko_kosong']} baris; "
        f"nama_produk kosong: {r['pemetaan']['n_nama_produk_kosong']} baris",
        "- A tidak memuat review_id; kunci ke B: (product_id, seq) di A = (product_id, seq_sekuens) di B.",
        "",
        "| product_id lama | product_id baru |",
        "|---|---|",
        *[f"| `{_md_sel(c['lama'])}` | `{_md_sel(c['baru'])}` |" for c in r["pemetaan"]["contoh"]],
        "",
        "## Panjang sekuens T per produk",
        "",
        f"min {t['min']}, median {t['median']}, p95 {t['p95']}, maks {t['maks']} (p95 = persentil linear numpy)",
        "",
        "## Catatan isi A (hanya dilaporkan, tidak diubah)",
        "",
        f"- Ulasan dengan timestamp tanpa jam (timestamp_has_time False): {r['n_sekuens_tanpa_jam']:,}",
        f"- Ulasan is_text_duplicate True yang masih masuk: {r['n_sekuens_is_text_duplicate']:,}",
        "",
        "## Invarian",
        "",
        "| invarian | hasil | bukti |",
        "|---|---|---|",
    ]
    lines += [f"| {x['invarian']} | {'LOLOS' if x['lolos'] else 'GAGAL'} | {x['bukti']} |" for x in ctx["invarian"]]
    lines += [
        "",
        "## SHA-256",
        "",
        "| berkas | sha256 |",
        "|---|---|",
    ]
    lines += [f"| `{k}` | `{v}` |" for k, v in ctx["hash"].items()]
    lines += [
        "",
        "## Berkas di folder ini",
        "",
        f"- [{ctx['nama_audit']}]({ctx['nama_audit']})",
        f"- [{NAMA_DIKELUARKAN}]({NAMA_DIKELUARKAN})",
        f"- [{NAMA_RINGKASAN}]({NAMA_RINGKASAN})",
        f"- [{NAMA_CHECKSUMS}]({NAMA_CHECKSUMS})",
        f"- Sekuens (A): [{ctx['rel_sekuens']}]({ctx['rel_sekuens']})",
        "",
        "## Asumsi",
        "",
        "- Input dibaca sebagai teks; nilai kolom (termasuk p_*) disalin apa adanya ke A dan B, tanpa pembulatan.",
        "- Kolom probabilitas dipilih menurut nama; urutan A mengikuti proposal: p_positif, p_netral, p_negatif.",
        "- Timestamp diurai sebagai ISO 8601 (tanggal saja dianggap pukul 00:00:00); kosong/tak terbaca = tidak valid.",
        "- Pemecah seri: timestamp, source_file (bila tersedia), row_order (numerik), review_id.",
        "- Ambang produk dihitung setelah alasan 1 sampai 3 diterapkan.",
        "- product_id baru dibentuk sebelum penyaringan dan pengurutan; produk di A terurut menurut product_id baru "
        "(urutan string, peka huruf besar/kecil).",
        "- Di B, product_id (baru) menempati posisi kolom product_id input; product_id_lama disisipkan tepat sesudahnya.",
        "- review_id hanya dipakai internal sebagai pemecah seri terakhir; tidak ditulis ke A.",
        "- is_text_duplicate dan timestamp tanpa jam tidak menyebabkan baris dikeluarkan.",
        "- Reproduksibilitas dibandingkan untuk A, B, dan daftar dikeluarkan (berkas lain memuat waktu/metadata).",
        "- label_argmax tidak dipakai; hanya ikut tersalin ke B.",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ekspor sekuens sentimen per produk + jejak audit")
    parser.add_argument("--input", default="outputs/sentiment/sentimen_ulasan_v2.csv")
    parser.add_argument("--out-sekuens", default="outputs/sentiment/sekuens_produk.csv")
    parser.add_argument("--out-audit", default="outputs/audit/sekuens_audit.csv")
    parser.add_argument("--min-ulasan", type=int, default=3)
    return parser


def _relpath(target: Path, start: Path) -> str:
    import os
    return Path(os.path.relpath(target.resolve(), start.resolve())).as_posix()


def run(args: argparse.Namespace, argv: list[str] | None = None) -> int:
    input_path = Path(args.input).expanduser()
    out_sekuens = Path(args.out_sekuens).expanduser()
    out_audit = Path(args.out_audit).expanduser()
    audit_dir = out_audit.parent
    paths = {
        "sekuens": out_sekuens,
        "audit": out_audit,
        "dikeluarkan": audit_dir / NAMA_DIKELUARKAN,
        "laporan": audit_dir / NAMA_LAPORAN,
        "ringkasan": audit_dir / NAMA_RINGKASAN,
        "checksums": audit_dir / NAMA_CHECKSUMS,
    }
    existing = [f'"{p}"' for p in paths.values() if p.exists()]
    if existing:
        print("BERHENTI: file keluaran sudah ada dan tidak akan ditimpa:\n  " + "\n  ".join(existing))
        return EXIT_STOPPED
    if not input_path.is_file():
        print(f'BERHENTI: input tidak ditemukan: "{input_path}"')
        return EXIT_STOPPED
    if args.min_ulasan < 1:
        print("BERHENTI: --min-ulasan harus >= 1")
        return EXIT_STOPPED

    input_bytes = input_path.read_bytes()
    hash_input_awal = sha256_bytes(input_bytes)
    try:
        berkas = hasilkan_berkas(input_bytes, args.min_ulasan)
    except EksporError as exc:
        print(f"BERHENTI: {exc}")
        return EXIT_STOPPED
    input_df = read_input_bytes(input_bytes)

    invarian = periksa_invarian(input_df, berkas, args.min_ulasan)
    repro = cek_reproduksibilitas(input_path, berkas, args.min_ulasan)
    invarian.append({
        "invarian": "reproduksibel_byte_demi_byte",
        "lolos": repro["identik"],
        "bukti": "; ".join(f"{k}: {v['run1'][:12]}..=={v['run2'][:12]}.." if v["run1"] == v["run2"]
                           else f"{k}: {v['run1'][:12]}..!={v['run2'][:12]}.." for k, v in repro["hash"].items())
                 + " (run kedua ditulis ke folder sementara di luar outputs/)",
    })
    for x in invarian:
        print(f"[{'LOLOS' if x['lolos'] else 'GAGAL'}] {x['invarian']}: {x['bukti']}")
    if not all(x["lolos"] for x in invarian):
        print("BERHENTI: ada invarian yang gagal; tidak ada berkas yang ditulis.")
        return EXIT_CHECK_FAILED

    for key in ("sekuens", "audit", "dikeluarkan"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        with open(paths[key], "xb") as f:
            f.write(berkas[key])

    hash_input_akhir = sha256_file(input_path)
    tulis_ok = all(sha256_file(paths[k]) == sha256_bytes(berkas[k]) for k in ("sekuens", "audit", "dikeluarkan"))
    invarian.append({
        "invarian": "input_tidak_berubah",
        "lolos": hash_input_awal == hash_input_akhir,
        "bukti": f"sebelum={hash_input_awal[:16]}.., sesudah={hash_input_akhir[:16]}..",
    })
    invarian.append({
        "invarian": "berkas_tertulis_sesuai_hash",
        "lolos": tulis_ok,
        "bukti": "hash berkas di disk dibandingkan dengan bytes yang diperiksa",
    })
    for x in invarian[-2:]:
        print(f"[{'LOLOS' if x['lolos'] else 'GAGAL'}] {x['invarian']}: {x['bukti']}")

    waktu = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    perintah = "python -m src.export_sekuens " + shlex.join(argv if argv is not None else sys.argv[1:])
    hashes = {
        str(input_path): hash_input_akhir,
        str(paths["sekuens"]): sha256_bytes(berkas["sekuens"]),
        str(paths["audit"]): sha256_bytes(berkas["audit"]),
        str(paths["dikeluarkan"]): sha256_bytes(berkas["dikeluarkan"]),
    }
    ringkasan = ringkas(input_df, berkas)
    ringkasan_json = {
        "dibuat": waktu,
        "perintah": perintah,
        "min_ulasan": args.min_ulasan,
        **ringkasan,
        "invarian": invarian,
        "sha256": hashes,
        "reproduksibilitas": repro["hash"],
    }
    with open(paths["ringkasan"], "x", encoding="utf-8", newline=LINE_TERMINATOR) as f:
        json.dump(ringkasan_json, f, ensure_ascii=False, indent=2)
        f.write("\n")
    laporan = format_laporan({
        "waktu": waktu, "perintah": perintah, "ringkasan": ringkasan, "invarian": invarian, "hash": hashes,
        "nama_audit": out_audit.name, "rel_sekuens": _relpath(out_sekuens, audit_dir),
    })
    with open(paths["laporan"], "x", encoding="utf-8", newline=LINE_TERMINATOR) as f:
        f.write(laporan)
    with open(paths["checksums"], "x", encoding="utf-8", newline=LINE_TERMINATOR) as f:
        f.write(f"# dibuat {waktu}\n")
        f.write("# CATATAN: bentuk product_id berubah. Lama: huruf kecil dengan pembatas '|' (kini product_id_lama di B).\n")
        f.write("# Baru: nama_toko + '_' + nama_produk (huruf asli). A tidak memuat review_id.\n")
        f.writelines(f"# contoh: {c['lama']} -> {c['baru']}\n" for c in ringkasan["pemetaan"]["contoh"])
        f.writelines(f"{h}  {p}\n" for p, h in hashes.items())

    print("\nBerkas ditulis:")
    for p in paths.values():
        print(f'  "{p}"')
    if not all(x["lolos"] for x in invarian):
        print("PERINGATAN: invarian pasca-tulis gagal; periksa laporan.")
        return EXIT_CHECK_FAILED
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return run(build_parser().parse_args(argv), argv)


if __name__ == "__main__":
    sys.exit(main())
