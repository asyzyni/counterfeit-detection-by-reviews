from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .build import BuildResult, count_date_only_duplicate_candidates
from .config import CleaningConfig
from .file_map import FileMap
from .schema import base_column_name
from .text_clean import normalize_fragment_key


# ============================================================
# CLEANING AUDIT (tiap aturan: sebelum / sesudah / alasan)
# ============================================================

def _audit_row(
    tahap: str,
    aturan: str,
    sebelum: int,
    sesudah: int,
    satuan: str,
    alasan: str,
) -> dict:

    return {
        "tahap": tahap,
        "aturan": aturan,
        "jumlah_sebelum": int(sebelum),
        "jumlah_sesudah": int(sesudah),
        "jumlah_terdampak": int(sebelum) - int(sesudah),
        "satuan": satuan,
        "alasan": alasan,
    }


def build_cleaning_audit(
    result: BuildResult,
    file_map: FileMap,
    config: CleaningConfig,
) -> pd.DataFrame:

    rows: list[dict] = []
    reviews = result.reviews
    manifest = result.manifest

    # --------------------------------------------------------
    # Tahap 1: file
    # --------------------------------------------------------

    n_processed = int((manifest["status"] == "diproses").sum())

    rows.append(_audit_row(
        "1_file", "pemetaan -> status ok",
        file_map.n_mapping_rows, len(file_map.ok), "baris pemetaan",
        "; ".join(f"{r.file}: {r.alasan}" for r in file_map.excluded.itertuples()),
    ))
    rows.append(_audit_row(
        "1_file", "file ok -> file diproses",
        len(file_map.ok), n_processed, "file",
        "dilewati: tanpa kolom review_text / duplikat isi bila opsi aktif",
    ))

    # --------------------------------------------------------
    # Tahap 2: baris sumber -> baris berteks
    # --------------------------------------------------------

    n_source = int(manifest["n_baris_sumber"].sum())
    n_empty_cells = int(manifest["n_kosong_semua_sel_teks_kosong"].fillna(0).sum())
    n_only_non_text = int(manifest["n_kosong_hanya_fragmen_non_teks"].fillna(0).sum())

    rows.append(_audit_row(
        "2_baris", "baris tanpa sel teks terisi",
        n_source, n_source - n_empty_cells, "baris sumber",
        "semua kolom review_text kosong (mis. ulasan bintang saja); tidak masuk tabel",
    ))
    rows.append(_audit_row(
        "2_baris", "baris yang selnya hanya non-teks",
        n_source - n_empty_cells, len(reviews), "baris sumber",
        "sel review_text hanya berisi tanggal/angka/url/label/nama tersamar; tidak masuk tabel",
    ))

    # --------------------------------------------------------
    # Tahap 3: fragmen sel
    # --------------------------------------------------------

    frag = result.fragment_counts
    in_review = frag[frag["peran_kolom"] == "review_text"]
    n_cells = int(in_review["jumlah_sel"].sum())
    n_kept = int(in_review.loc[in_review["jenis_nilai"] == "teks_masuk_review", "jumlah_sel"].sum())

    rows.append(_audit_row(
        "3_fragmen", "sel kolom review_text -> fragmen review",
        n_cells, n_kept, "sel",
        "disaring per nilai, rincian di baris 3_fragmen_rinci",
    ))

    reasons = {
        "tanggal": "nilai berupa tanggal (kolom bergeser saat scrape)",
        "tanggal_relatif": "nilai berupa tanggal relatif",
        "url": "nilai berupa URL",
        "angka": "nilai tanpa huruf, hanya angka (durasi video, +N foto, dsb.)",
        "label_aspek": "label aspek berakhiran ':' -> dipindah ke aspect_tags",
        "tag_aspek_kosakata": "cocok kosakata tag aspek yang disetujui -> aspect_tags",
        "nama_tersamar": "nama pengguna tersamar (x***y)",
        "penanda_penjual": "penanda balasan penjual",
        "simbol": "hanya simbol/tanda baca tanpa huruf/angka/emoji",
    }

    for row in in_review.itertuples():
        if row.jenis_nilai == "teks_masuk_review":
            continue
        rows.append(_audit_row(
            "3_fragmen_rinci", f"review_text: {row.jenis_nilai}",
            row.jumlah_sel, 0, "sel",
            reasons.get(row.jenis_nilai, row.jenis_nilai),
        ))

    other = frag[(frag["peran_kolom"] != "review_text") & (frag["jenis_nilai"] == "teks")]
    for row in other.itertuples():
        rows.append(_audit_row(
            "3_fragmen_rinci", f"sel berhuruf di kolom berperan {row.peran_kolom}",
            row.jumlah_sel, 0, "sel",
            "peran kolom bukan review_text (lihat column_map.csv); tidak masuk review",
        ))

    # --------------------------------------------------------
    # Tahap 4: text_light
    # --------------------------------------------------------

    for step, n_changed in result.light_changes.items():
        rows.append(_audit_row(
            "4_text_light", step,
            len(reviews), len(reviews) - n_changed, "baris (berubah = terdampak)",
            "pembersihan minimal; baris tetap ada",
        ))

    # --------------------------------------------------------
    # Tahap 5: flag -> include_for_inference
    # --------------------------------------------------------

    n = len(reviews)
    empty = int(reviews["is_empty"].sum())
    short = int(reviews["is_short"].sum())
    tech = int(reviews["is_technical_duplicate"].sum())

    rows.append(_audit_row(
        "5_flag", "is_empty", n, n - empty, "baris",
        "text_light kosong setelah dibersihkan (mis. hanya URL/HTML)",
    ))
    rows.append(_audit_row(
        "5_flag", "is_short", n - empty, n - empty - short, "baris",
        f"text_light < {config.min_review_length} karakter",
    ))
    rows.append(_audit_row(
        "5_flag", "is_technical_duplicate", n - empty - short,
        int(reviews["include_for_inference"].sum()), "baris",
        "product + text_light + timestamp berjam sama persis (scrape ganda)",
    ))
    rows.append(_audit_row(
        "5_flag", "is_text_duplicate (hanya ditandai)",
        n, n, "baris",
        f"{int(reviews['is_text_duplicate'].sum())} baris teks sama dalam produk, "
        f"timestamp beda; TIDAK dibuang",
    ))
    rows.append(_audit_row(
        "5_flag", "kandidat duplikat tanggal-saja (info)",
        n, n, "baris",
        f"{count_date_only_duplicate_candidates(reviews)} baris teks+tanggal sama "
        f"tetapi tanpa jam; tidak ditandai teknis (technical_dup_require_time="
        f"{config.technical_dup_require_time})",
    ))

    # --------------------------------------------------------
    # Tahap 6: timestamp
    # --------------------------------------------------------

    for status, count in reviews["timestamp_status"].value_counts().sort_index().items():
        rows.append(_audit_row(
            "6_timestamp", f"status {status}", n, n - (0 if status == "ok" else count),
            "baris",
            f"{count} baris; selain 'ok' -> NaT (tidak ditebak)",
        ))

    return pd.DataFrame(rows)


# ============================================================
# FORMAT TIMESTAMP PER KELUARGA SKEMA
# ============================================================

def timestamp_format_report(
    reviews: pd.DataFrame,
) -> pd.DataFrame:

    def pattern(raw: str) -> str:
        head = str(raw).split("|", 1)[0].strip()
        if not head:
            return "<kosong>"
        pattern_text = re.sub(r"\d+", "N", head) if re.search(r"[a-zA-Z]", head) \
            else re.sub(r"\d", "9", head)
        suffix = " | <teks>" if "|" in str(raw) else ""
        return pattern_text + suffix

    frame = reviews.assign(pola=reviews["timestamp_raw"].map(pattern))

    report = (
        frame.groupby(["schema_family", "pola", "timestamp_status"])
        .agg(
            n_baris=("review_id", "size"),
            n_file=("source_file", "nunique"),
            contoh=("timestamp_raw", "first"),
        )
        .reset_index()
        .sort_values(["schema_family", "n_baris"], ascending=[True, False], kind="mergesort")
    )

    return report.reset_index(drop=True)


# ============================================================
# KANDIDAT TAG ASPEK & FRASA TEMPLAT
# ============================================================

def aspect_tag_candidates(
    result: BuildResult,
    config: CleaningConfig,
) -> pd.DataFrame:

    frags = result.review_fragments
    columns = [
        "fragmen", "n_file", "n_produk", "n_kemunculan", "contoh_asli",
        "pernah_berakhiran_titik_dua", "kolom_asal", "keluarga_skema", "setuju",
    ]

    if frags.empty:
        return pd.DataFrame(columns=columns)

    family_of = dict(zip(result.manifest["file"], result.manifest["schema_family"]))
    product_of = dict(zip(result.manifest["file"], result.manifest["product_key"]))

    frags = frags.assign(
        fragmen=frags["fragment"].map(normalize_fragment_key),
        n_kata=frags["fragment"].str.split().str.len(),
        kolom=frags["column"].map(base_column_name),
        keluarga=frags["source_file"].map(family_of),
        produk=frags["source_file"].map(product_of),
    )
    frags = frags[(frags["n_kata"] <= config.aspect_candidate_max_words) & (frags["fragmen"] != "")]

    def ranked(series: pd.Series) -> pd.Series:
        # frekuensi menurun, seri dipecah alfabetis -> deterministik
        counts = series.value_counts()
        return counts.sort_index(kind="mergesort").sort_values(ascending=False, kind="mergesort")

    def top(series: pd.Series, k: int = 3) -> str:
        return "; ".join(f"{name} ({n})" for name, n in ranked(series).head(k).items())

    table = (
        frags.groupby("fragmen")
        .agg(
            n_file=("source_file", "nunique"),
            n_produk=("produk", "nunique"),
            n_kemunculan=("fragment", "size"),
            contoh_asli=("fragment", lambda s: ranked(s).index[0]),
            pernah_berakhiran_titik_dua=("fragment", lambda s: bool(s.str.strip().str.endswith(":").any())),
            kolom_asal=("kolom", top),
            keluarga_skema=("keluarga", top),
        )
        .reset_index()
    )

    table = table[table["n_file"] >= config.aspect_candidate_min_files]
    table["setuju"] = ""

    return table.sort_values(
        ["n_file", "n_kemunculan", "fragmen"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)[columns]


def preset_phrase_candidates(
    result: BuildResult,
    config: CleaningConfig,
) -> pd.DataFrame:
    """Segmen dipisah koma yang muncul di banyak file (indikasi frasa pilihan siap pakai)."""

    frags = result.review_fragments
    columns = ["segmen", "n_file", "n_kemunculan", "contoh_fragmen"]

    if frags.empty:
        return pd.DataFrame(columns=columns)

    segments = frags.assign(segmen=frags["fragment"].str.split(",")).explode("segmen")
    segments["segmen"] = segments["segmen"].map(normalize_fragment_key)
    segments = segments[
        segments["segmen"].str.split().str.len() >= config.preset_phrase_min_words
    ]

    table = (
        segments.groupby("segmen")
        .agg(
            n_file=("source_file", "nunique"),
            n_kemunculan=("fragment", "size"),
            contoh_fragmen=("fragment", "first"),
        )
        .reset_index()
    )
    table = table[table["n_file"] >= config.preset_phrase_min_files]

    return table.sort_values(
        ["n_file", "n_kemunculan", "segmen"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)[columns]


# ============================================================
# RINGKASAN PRODUK (tanpa memfilter)
# ============================================================

def product_summary(
    reviews: pd.DataFrame,
    config: CleaningConfig,
) -> pd.DataFrame:

    grouped = reviews.groupby("product_id", sort=True)

    summary = grouped.agg(
        nama_toko=("nama_toko", "first"),
        nama_produk=("nama_produk", "first"),
        n_file=("source_file", "nunique"),
        file=("source_file", lambda s: " | ".join(sorted(s.unique()))),
        n_baris_tabel=("review_id", "size"),
        n_include=("include_for_inference", "sum"),
        n_technical_duplicate=("is_technical_duplicate", "sum"),
        n_text_duplicate=("is_text_duplicate", "sum"),
        n_is_short=("is_short", "sum"),
        n_is_empty=("is_empty", "sum"),
    ).reset_index()

    include_nat = (
        reviews[reviews["include_for_inference"]]
        .groupby("product_id")["timestamp"]
        .apply(lambda s: int(s.isna().sum()))
        .rename("n_include_timestamp_nat")
    )
    summary = summary.merge(include_nat, on="product_id", how="left")
    summary["n_include_timestamp_nat"] = summary["n_include_timestamp_nat"].fillna(0).astype("int64")

    for threshold in config.product_thresholds:
        summary[f"lolos_{threshold}"] = summary["n_include"] >= threshold

    return summary


# ============================================================
# VALIDASI OTOMATIS
# ============================================================

@dataclass
class Check:
    nama: str
    lulus: bool
    detail: str


def run_checks(
    result: BuildResult,
    file_map: FileMap,
    config: CleaningConfig,
) -> list[Check]:

    checks: list[Check] = []
    reviews = result.reviews
    manifest = result.manifest

    # --------------------------------------------------------
    # File
    # --------------------------------------------------------

    n_ok = len(file_map.ok)
    n_processed = int((manifest["status"] == "diproses").sum())
    n_skipped = len(result.skipped_no_review)
    n_dup_excluded = int(
        result.excluded_files["alasan"].str.startswith("isi identik").sum()
    )

    expected = config.expected_ok_files if config.expected_ok_files is not None else n_ok
    checks.append(Check(
        "jumlah file status ok = harapan",
        n_ok == expected,
        f"ok={n_ok}, harapan={expected}",
    ))
    checks.append(Check(
        "file diproses + dilewati + dikecualikan-duplikat = file ok",
        n_processed + n_skipped + n_dup_excluded == n_ok,
        f"diproses={n_processed}, tanpa review_text={n_skipped}, "
        f"duplikat-dikecualikan={n_dup_excluded}, ok={n_ok}",
    ))
    checks.append(Check(
        "file dikecualikan dilaporkan",
        len(result.excluded_files) >= len(file_map.excluded),
        "; ".join(f"{r.file} ({r.alasan})" for r in result.excluded_files.itertuples()),
    ))

    no_review_rate = n_skipped / max(n_ok, 1)
    checks.append(Check(
        f"file tanpa review_text <= {config.max_files_without_review_rate:.0%}",
        no_review_rate <= config.max_files_without_review_rate,
        f"{n_skipped} file ({no_review_rate:.1%}): {result.skipped_no_review}",
    ))

    # --------------------------------------------------------
    # Baris
    # --------------------------------------------------------

    unbalanced = manifest.loc[~manifest["cek_seimbang"], "file"].tolist()
    checks.append(Check(
        "tiap file: baris sumber = baris kosong + baris keluar",
        not unbalanced,
        f"{len(manifest)} file diperiksa; tidak seimbang: {unbalanced}",
    ))

    rows_per_file = reviews.groupby("source_file").size()
    mismatch = [
        name for name, n in zip(manifest["file"], manifest["n_baris_keluar"])
        if int(rows_per_file.get(name, 0)) != int(n)
    ]
    checks.append(Check(
        "baris keluar di manifest = baris di tabel",
        not mismatch,
        f"selisih: {mismatch}",
    ))

    # --------------------------------------------------------
    # ID & isi
    # --------------------------------------------------------

    checks.append(Check(
        "review_id unik & tanpa NaN",
        reviews["review_id"].notna().all() and reviews["review_id"].is_unique,
        f"{len(reviews):,} baris, {reviews['review_id'].nunique():,} id unik",
    ))

    included = reviews[reviews["include_for_inference"]]
    bad_included = int(
        (included["review"].str.strip() == "").sum()
        + (included["text_light"].str.strip() == "").sum()
    )
    checks.append(Check(
        "tidak ada review/text_light kosong pada include_for_inference",
        bad_included == 0,
        f"{len(included):,} baris include; kosong={bad_included}",
    ))

    unknown_products = set(reviews["product_id"]) - set(file_map.ok["product_key"])
    checks.append(Check(
        "product_id seluruhnya berasal dari pemetaan",
        not unknown_products,
        f"tidak dikenal: {sorted(unknown_products)}",
    ))

    column_map = result.column_map
    roles = dict(zip(zip(column_map["file"], column_map["column"]), column_map["role"]))
    leaked_roles = {
        roles.get((file, col), "?")
        for file, cols in zip(reviews["source_file"], reviews["fragment_cols"])
        for col in cols.split(" | ")
        if col
    }
    checks.append(Check(
        "fragmen review hanya dari kolom berperan review_text",
        leaked_roles <= {"review_text"},
        f"peran asal fragmen: {sorted(leaked_roles)}",
    ))

    seq_ok = (
        reviews.groupby("product_id")["seq_in_product"]
        .apply(lambda s: sorted(s.tolist()) == list(range(1, len(s) + 1)))
        .all()
    ) if len(reviews) else True
    checks.append(Check(
        "seq_in_product = 1..n per produk",
        bool(seq_ok),
        f"{reviews['product_id'].nunique()} produk",
    ))

    return checks


def checks_to_frame(
    checks: list[Check],
) -> pd.DataFrame:

    return pd.DataFrame([vars(check) for check in checks])
