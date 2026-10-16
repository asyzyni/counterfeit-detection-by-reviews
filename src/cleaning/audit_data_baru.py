"""Audit berkas CSV baru dan kandidat salinan sebelum cleaning v4 (hanya membaca data mentah).

    python -m src.cleaning.audit_data_baru \
        --out-md "reports/v4/audit_data_baru_v4.md" \
        --out-pasangan "reports/v4/tumpang_tindih_berkas_v4.csv" \
        --out-pengecualian "references/file_dikecualikan_v4.csv"

Tumpang tindih dihitung pada tingkat ULASAN: kolom ulasan dipetakan dengan fungsi cleaning
yang sama (map_file_columns + build_file_rows), bukan baris fisik CSV. Kunci ulasan =
review hasil penggabungan fragmen, NFKC, spasi diringkas, casefold. Persentase dihitung
dengan irisan multiset: pct_a_dalam_b = jumlah ulasan A yang juga ada di B / jumlah ulasan A.

Aturan keputusan (tidak memindah/menghapus berkas apa pun):
    keputusan="kecualikan" hanya bila >= 80% ulasan berkas itu ada di SATU berkas lain yang
    tetap dipertahankan, dan salah satu dari keduanya bernama salinan (" copy", "(n)").
    Berkas ber-hash identik dianggap tumpang tindih 100% walau tidak menghasilkan ulasan.
    Sisanya keputusan="pertahankan". File keluaran yang sudah ada tidak ditimpa.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .config import CleaningConfig
from .file_map import read_raw_csv
from .schema import map_file_columns, profiles_to_frame
from .build import build_file_rows

AMBANG_PCT = 80.0
EXT_RE = re.compile(r"(\.csv)+$", re.I)
SALINAN_RE = re.compile(r"(\s+copy(\s*\d+)?|\s*\(\d+\))\s*$", re.I)
HURUF_GANDA_RE = re.compile(r"([a-z])\1+")
PENGECUALIAN_COLUMNS = ["file", "alasan", "berkas_pasangan", "pct_tumpang_tindih", "keputusan"]


# ============================================================
# NAMA BERKAS
# ============================================================

def stem_berkas(name: str) -> str:
    """Nama tanpa akhiran .csv (berulang, mis. ".csv.csv")."""
    return EXT_RE.sub("", name)


def is_salinan(name: str) -> bool:
    return bool(SALINAN_RE.search(stem_berkas(name)))


def normalisasi_nama(name: str) -> str:
    """Kunci kemiripan nama: huruf kecil, tanpa akhiran salinan, tanpa tanda baca,
    huruf ganda diringkas (PHONCELL ~ PHONCEL), spasi diringkas."""
    text = SALINAN_RE.sub("", stem_berkas(unicodedata.normalize("NFKC", name))).casefold()
    text = re.sub(r"[^0-9a-z]+", " ", text)
    text = HURUF_GANDA_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def turunkan_pemetaan(name: str) -> dict:
    """Toko/produk dari nama berkas mengikuti pemetaan lama: dipisah pada underscore TERAKHIR,
    spasi di ujung dibuang, product_key = toko|produk huruf kecil. Akhiran salinan TIDAK dibuang
    dari nama_produk (sama dengan aturan lama "nama file"); hanya dicatat."""
    stem = stem_berkas(name)
    catatan = []
    if "_" not in stem:
        return {"file": name, "nama_toko": "", "nama_produk": stem.strip(), "product_key": "",
                "status": "perlu_cek", "sumber_nama": "tanpa underscore - cek manual",
                "catatan": "nama berkas tanpa '_' ; toko tidak dapat diturunkan"}
    toko, produk = stem.rsplit("_", 1)
    toko, produk = toko.strip(), produk.strip()
    sumber = "underscore terakhir" if stem.count("_") > 1 else "nama file"
    if stem.count("_") > 1:
        catatan.append("lebih dari satu underscore; dipisah pada underscore terakhir")
    if name.lower().count(".csv") > 1:
        catatan.append("akhiran .csv berulang dibuang")
    if '"' in name or "'" in name:
        catatan.append("nama memuat tanda kutip")
    if name != name.strip() or stem != stem.rstrip() or " _" in stem or "_ " in stem:
        catatan.append("spasi di ujung/di sekitar underscore dibuang")
    if is_salinan(name):
        catatan.append("nama berakhiran salinan; akhiran tetap di nama_produk - cek")
    if not toko or not produk:
        return {"file": name, "nama_toko": toko, "nama_produk": produk, "product_key": "",
                "status": "perlu_cek", "sumber_nama": sumber, "catatan": "toko/produk kosong"}
    return {"file": name, "nama_toko": toko, "nama_produk": produk,
            "product_key": f"{toko.lower()}|{produk.lower()}", "status": "ok",
            "sumber_nama": sumber, "catatan": "; ".join(catatan)}


# ============================================================
# ULASAN PER BERKAS
# ============================================================

def kunci_ulasan(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def ulasan_berkas(path: Path, config: CleaningConfig) -> pd.DataFrame:
    """Ulasan satu berkas hasil pemetaan skema cleaning (kolom: row_order, review, timestamp_raw)."""
    df, _ = read_raw_csv(path)
    profiles = profiles_to_frame(map_file_columns(path.name, df, config))
    meta = pd.Series({"file": path.name, "product_key": "audit", "nama_toko": "", "nama_produk": ""})
    rows, _, _, _ = build_file_rows(meta, df, profiles, config, set())
    return pd.DataFrame(
        [(r["row_order"], r["review"], r["timestamp_raw"]) for r in rows],
        columns=["row_order", "review", "timestamp_raw"],
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ============================================================
# TUMPANG TINDIH
# ============================================================

def tumpang_tindih(ulasan: dict[str, list[str]], sha: dict[str, str] | None = None,
                   min_pct: float = 30.0) -> pd.DataFrame:
    """Irisan multiset kunci ulasan untuk setiap pasangan berkas yang berbagi ulasan.

    ulasan: file -> daftar kunci ulasan. Hanya pasangan dengan max(pct) >= min_pct atau hash
    identik yang dikembalikan.
    """
    sha = sha or {}
    counts = {name: Counter(keys) for name, keys in ulasan.items()}
    total = {name: sum(c.values()) for name, c in counts.items()}
    index: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for name in sorted(counts):
        for key, n in counts[name].items():
            index[key].append((name, n))
    shared: Counter = Counter()
    for items in index.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (a, na), (b, nb) = items[i], items[j]
                shared[(a, b)] += min(na, nb)

    by_hash: dict[str, list[str]] = defaultdict(list)
    for name in sorted(sha):
        by_hash[sha[name]].append(name)
    for names in by_hash.values():
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                shared.setdefault((names[i], names[j]), 0)

    rows = []
    for (a, b), n in shared.items():
        na, nb = total.get(a, 0), total.get(b, 0)
        identik = bool(sha.get(a)) and sha.get(a) == sha.get(b)
        pa = 100.0 if identik else (100.0 * n / na if na else 0.0)
        pb = 100.0 if identik else (100.0 * n / nb if nb else 0.0)
        if max(pa, pb) >= min_pct or identik:
            rows.append({"file_a": a, "file_b": b, "n_ulasan_a": na, "n_ulasan_b": nb,
                         "n_ulasan_sama": n, "pct_a_dalam_b": round(pa, 1),
                         "pct_b_dalam_a": round(pb, 1), "hash_identik": identik})
    columns = ["file_a", "file_b", "n_ulasan_a", "n_ulasan_b", "n_ulasan_sama",
               "pct_a_dalam_b", "pct_b_dalam_a", "hash_identik"]
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values(["n_ulasan_sama", "file_a", "file_b"], ascending=[False, True, True],
                           kind="mergesort").reset_index(drop=True)


def _arah(pairs: pd.DataFrame) -> pd.DataFrame:
    """Satu baris per (file, pasangan) dengan pct = persen ulasan file yang ada di pasangan."""
    a = pairs.rename(columns={"file_a": "file", "file_b": "pasangan", "pct_a_dalam_b": "pct"})
    b = pairs.rename(columns={"file_b": "file", "file_a": "pasangan", "pct_b_dalam_a": "pct"})
    return pd.concat([a[["file", "pasangan", "pct", "hash_identik"]],
                      b[["file", "pasangan", "pct", "hash_identik"]]], ignore_index=True)


def putuskan(files: list[str], pairs: pd.DataFrame, ambang: float = AMBANG_PCT) -> pd.DataFrame:
    """Terapkan aturan pengecualian; berkas bernama salinan diproses lebih dulu (urut nama),
    sehingga dari satu pasangan paling banyak satu berkas dikecualikan."""
    arah = _arah(pairs)
    arah = arah[arah["pct"] >= ambang]
    excluded: dict[str, tuple[str, float, bool]] = {}
    order = sorted(files, key=lambda name: (not is_salinan(name), name))
    for name in order:
        options = arah[(arah["file"] == name) & ~arah["pasangan"].isin(excluded)]
        if not is_salinan(name):
            options = options[options["pasangan"].map(is_salinan)]
        if options.empty:
            continue
        best = options.sort_values(["pct", "hash_identik", "pasangan"], ascending=[False, False, True],
                                   kind="mergesort").iloc[0]
        excluded[name] = (best["pasangan"], float(best["pct"]), bool(best["hash_identik"]))

    rows = []
    for name in sorted(files):
        if name in excluded:
            partner, pct, identik = excluded[name]
            alasan = (("isi berkas identik (hash sama)" if identik else
                       f">= {ambang:g}% ulasan ada di berkas lain")
                      + ("; berkas ini bernama salinan" if is_salinan(name) else
                         "; berkas pasangan bernama salinan dan lebih lengkap"))
            rows.append({"file": name, "alasan": alasan, "berkas_pasangan": partner,
                         "pct_tumpang_tindih": pct, "keputusan": "kecualikan"})
        else:
            mine = _arah(pairs)
            mine = mine[mine["file"] == name].sort_values("pct", ascending=False, kind="mergesort")
            partner, pct = (mine.iloc[0]["pasangan"], float(mine.iloc[0]["pct"])) if len(mine) else ("", 0.0)
            rows.append({"file": name, "alasan": "tidak memenuhi aturan pengecualian",
                         "berkas_pasangan": partner, "pct_tumpang_tindih": pct,
                         "keputusan": "pertahankan"})
    return pd.DataFrame(rows, columns=PENGECUALIAN_COLUMNS)


# ============================================================
# LAPORAN
# ============================================================

def _md(value: object) -> str:
    return str(value).replace("|", "&#124;").replace("\n", " ")


def usulan(name: str, row: pd.Series, pemetaan: dict, n_ulasan: int, kunci_bentrok: set[str]) -> str:
    if row["keputusan"] == "kecualikan":
        return "kecualikan (aturan salinan)"
    notes = []
    if n_ulasan == 0:
        notes.append("0 ulasan terbaca (cleaning akan melewati berkas); cek skema")
    if row["pct_tumpang_tindih"] >= AMBANG_PCT:
        notes.append(f"tumpang tindih {row['pct_tumpang_tindih']:g}% tanpa nama salinan; tinjau manual")
    if pemetaan.get("product_key") in kunci_bentrok:
        notes.append("product_key sama dengan berkas lain (digabung jadi satu produk)")
    if is_salinan(name):
        notes.append("bernama salinan tetapi tidak memenuhi aturan; tinjau")
    return "; ".join(notes) or "pertahankan"


def format_laporan(ctx: dict) -> str:
    new, pairs, decisions = ctx["baru"], ctx["pasangan"], ctx["keputusan"]
    mapping_new, n_reviews = ctx["pemetaan_baru"], ctx["n_ulasan"]
    groups, kunci_bentrok = ctx["grup_nama"], ctx["kunci_bentrok"]
    lines = ["# Audit data baru v4", "",
             f'- Folder data mentah (hanya dibaca): "{ctx["raw_dir"]}"',
             f'- Manifest lama: "{ctx["manifest_lama"]}" ({ctx["n_manifest_lama"]} entri)',
             f'- Pemetaan lama: "{ctx["pemetaan_lama"]}" ({ctx["n_pemetaan_lama"]} baris)',
             f"- CSV di folder sekarang: {ctx['n_csv']} (termasuk berkas pemetaan)",
             f"- Berkas baru (tidak ada di manifest lama): {len(new)}",
             f"- Berkas dikecualikan menurut aturan: {int((decisions['keputusan'] == 'kecualikan').sum())}",
             "", "## Metode", "",
             "Ulasan dibaca dengan pemetaan skema cleaning yang sama (map_file_columns, build_file_rows); "
             "satu ulasan = satu baris keluaran cleaning, bukan baris fisik CSV. Kunci ulasan: teks "
             "review, NFKC, spasi diringkas, casefold. Persentase = irisan multiset / jumlah ulasan "
             "berkas tersebut. Berkas ber-hash identik dihitung 100%.", "",
             f"Aturan: kecualikan hanya bila >= {AMBANG_PCT:g}% ulasan berkas ada di satu berkas lain "
             "(yang dipertahankan) dan salah satunya bernama salinan (akhiran \" copy\" atau \"(n)\"). "
             "Tidak ada berkas yang dipindah/dihapus; keputusan hanya mengubah status di salinan pemetaan v4.",
             "", "## Berkas baru", "",
             "| Berkas | Ulasan | Toko | Produk | product_key | Catatan pemetaan | Keputusan | Usulan |",
             "| --- | ---: | --- | --- | --- | --- | --- | --- |"]
    dec = decisions.set_index("file")
    for name in new:
        m = mapping_new[name]
        d = dec.loc[name] if name in dec.index else None
        sug = usulan(name, d, m, n_reviews.get(name, 0), kunci_bentrok) if d is not None else ""
        lines.append("| " + " | ".join(_md(v) for v in (
            name, n_reviews.get(name, 0), m["nama_toko"], m["nama_produk"], m["product_key"],
            m["catatan"], d["keputusan"] if d is not None else "", sug)) + " |")

    lines += ["", "## Pasangan kandidat duplikat (melibatkan berkas baru, max pct >= 50 atau hash identik)", "",
              "| Berkas A | Berkas B | Ulasan A | Ulasan B | Sama | % A di B | % B di A | Hash identik |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | :-: |"]
    newset = set(new)
    focus = pairs[(pairs["file_a"].isin(newset) | pairs["file_b"].isin(newset))
                  & ((pairs[["pct_a_dalam_b", "pct_b_dalam_a"]].max(axis=1) >= 50) | pairs["hash_identik"])]
    for r in focus.itertuples():
        lines.append("| " + " | ".join(_md(v) for v in (
            r.file_a + (" (baru)" if r.file_a in newset else ""),
            r.file_b + (" (baru)" if r.file_b in newset else ""),
            r.n_ulasan_a, r.n_ulasan_b, r.n_ulasan_sama, r.pct_a_dalam_b, r.pct_b_dalam_a,
            "ya" if r.hash_identik else "")) + " |")

    lines += ["", "## Grup nama mirip (setelah normalisasi nama)", "",
              "Normalisasi: huruf kecil, tanpa akhiran salinan/.csv, tanda baca jadi spasi, "
              "huruf ganda diringkas (PHONCELL ~ PHONCEL).", "",
              "| Kunci nama | Berkas |", "| --- | --- |"]
    for key, names in groups.items():
        lines.append(f"| {_md(key)} | " + "<br>".join(_md(n + (' (baru)' if n in newset else '')) for n in names) + " |")

    old_pairs = pairs[~pairs["file_a"].isin(newset) & ~pairs["file_b"].isin(newset)
                      & (pairs[["pct_a_dalam_b", "pct_b_dalam_a"]].max(axis=1) >= AMBANG_PCT)]
    lines += ["", f"## Informasi: pasangan berkas LAMA dengan tumpang tindih >= {AMBANG_PCT:g}%", "",
              "Pasangan ini sudah ada di v3 dan tidak dikecualikan oleh aturan (tidak bernama salinan). "
              "Banyak di antaranya beda produk dalam toko yang sama (kemungkinan ulasan tingkat toko "
              "atau salah halaman saat scraping). Perlu keputusan pengguna.", "",
              f"Jumlah pasangan: {len(old_pairs)}; hash identik: {int(old_pairs['hash_identik'].sum())}.", "",
              "| Berkas A | Berkas B | Ulasan A | Ulasan B | Sama | % A di B | % B di A | Hash identik |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | :-: |"]
    for r in old_pairs.itertuples():
        lines.append("| " + " | ".join(_md(v) for v in (
            r.file_a, r.file_b, r.n_ulasan_a, r.n_ulasan_b, r.n_ulasan_sama, r.pct_a_dalam_b,
            r.pct_b_dalam_a, "ya" if r.hash_identik else "")) + " |")
    lines += ["", "## Keputusan pengecualian", "",
              f'Lihat "{ctx["out_pengecualian"]}". Kolom keputusan dapat diedit (kecualikan/pertahankan), '
              "lalu salinan pemetaan dibuat ulang dengan `python -m src.cleaning.pemetaan_v4` ke nama berkas baru.", ""]
    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    defaults = CleaningConfig()
    parser = argparse.ArgumentParser(description="Audit berkas baru + kandidat salinan (data mentah hanya dibaca)")
    parser.add_argument("--raw-dir", type=Path, default=defaults.raw_dir)
    parser.add_argument("--pemetaan-lama", type=Path, default=defaults.mapping_path)
    parser.add_argument("--manifest-lama", type=Path,
                        default=defaults.output_dir / "reports" / "file_manifest.csv")
    parser.add_argument("--out-md", type=Path, default=Path("reports/v4/audit_data_baru_v4.md"))
    parser.add_argument("--out-pasangan", type=Path, default=Path("reports/v4/tumpang_tindih_berkas_v4.csv"))
    parser.add_argument("--out-pengecualian", type=Path, default=Path("references/file_dikecualikan_v4.csv"))
    return parser


def run(args: argparse.Namespace) -> int:
    outputs = [args.out_md, args.out_pasangan, args.out_pengecualian]
    existing = [f'"{p}"' for p in outputs if p.exists()]
    if existing:
        print("BERHENTI: file keluaran sudah ada dan tidak akan ditimpa: " + ", ".join(existing))
        return 2
    config = CleaningConfig(raw_dir=args.raw_dir)
    raw = Path(args.raw_dir)
    mapping_name = Path(args.pemetaan_lama).name
    old_mapping = pd.read_csv(args.pemetaan_lama, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    manifest = pd.read_csv(args.manifest_lama, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    csvs = sorted(p.name for p in raw.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    data_files = [name for name in csvs if name != mapping_name]
    new = [name for name in data_files if name not in set(manifest["file"])]

    sha = {name: sha256_file(raw / name) for name in data_files}
    keys, n_reviews = {}, {}
    for name in data_files:
        reviews = ulasan_berkas(raw / name, config)
        keys[name] = [kunci_ulasan(t) for t in reviews["review"]]
        n_reviews[name] = len(reviews)
    pairs = tumpang_tindih(keys, sha)

    groups_all: dict[str, list[str]] = defaultdict(list)
    for name in data_files:
        groups_all[normalisasi_nama(name)].append(name)
    groups = {k: v for k, v in sorted(groups_all.items()) if len(v) > 1 and set(v) & set(new)}

    # Berkas yang dinilai: semua berkas baru + semua berkas pada pasangan >= ambang atau grup nama.
    involved = set(new)
    strong = pairs[(pairs[["pct_a_dalam_b", "pct_b_dalam_a"]].max(axis=1) >= AMBANG_PCT) | pairs["hash_identik"]]
    involved |= set(strong["file_a"]) | set(strong["file_b"])
    for names in groups.values():
        involved |= set(names)
    decisions = putuskan(sorted(involved), pairs)

    mapping_new = {name: turunkan_pemetaan(name) for name in new}
    old_keys = old_mapping.loc[old_mapping["status"].str.strip() == "ok", "product_key"].tolist()
    all_keys = Counter(old_keys + [m["product_key"] for m in mapping_new.values() if m["product_key"]])
    kunci_bentrok = {key for key, n in all_keys.items() if n > 1}

    ctx = {"raw_dir": raw, "manifest_lama": args.manifest_lama, "n_manifest_lama": len(manifest),
           "pemetaan_lama": args.pemetaan_lama, "n_pemetaan_lama": len(old_mapping), "n_csv": len(csvs),
           "baru": new, "pasangan": pairs, "keputusan": decisions, "pemetaan_baru": mapping_new,
           "n_ulasan": n_reviews, "grup_nama": groups, "kunci_bentrok": kunci_bentrok,
           "out_pengecualian": args.out_pengecualian}
    report = format_laporan(ctx)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pasangan, "x", encoding="utf-8-sig", newline="") as f:
        pairs.to_csv(f, index=False, lineterminator="\n")
    with open(args.out_pengecualian, "x", encoding="utf-8-sig", newline="") as f:
        decisions.to_csv(f, index=False, lineterminator="\n")
    with open(args.out_md, "x", encoding="utf-8") as f:
        f.write(report)
    print(f"CSV: {len(csvs)}; berkas baru: {len(new)}; pasangan: {len(pairs)}; "
          f"dikecualikan: {int((decisions['keputusan'] == 'kecualikan').sum())}")
    for path in outputs:
        print(f'  "{path}"')
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
