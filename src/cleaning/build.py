from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CleaningConfig
from .file_map import (
    FileMap,
    build_duplicate_file_groups,
    detect_family,
    file_sha256,
    read_raw_csv,
)
from .schema import (
    DATE_REL_RE,
    SELLER_STYLE_RE,
    ROLE_ASPECT,
    ROLE_REVIEW,
    ROLE_TIMESTAMP,
    classify_value,
    map_file_columns,
    profiles_to_frame,
)
from .text_clean import (
    clean_light_series,
    join_for_light,
    join_review,
    normalize_fragment_key,
)


# ============================================================
# CONSTANTS
# ============================================================

OUTPUT_COLUMNS = [
    "review_id",
    "product_id",
    "nama_toko",
    "nama_produk",
    "review",
    "text_light",
    "aspect_tags",
    "timestamp",
    "timestamp_has_time",
    "timestamp_raw",
    "timestamp_status",
    "row_order",
    "source_file",
    "fragment_cols",
    "n_fragments",
    "n_chars",
    "n_words",
    "is_empty",
    "is_short",
    "is_technical_duplicate",
    "is_text_duplicate",
    "dup_group_size",
    "seller_reply_suspect",
    "include_for_inference",
    "seq_in_product",
    "file_dup_group",
    "schema_family",
]

# Jenis nilai (lihat schema.classify_value) yang tidak masuk `review`.
NON_TEXT_KINDS = (
    "tanggal",
    "tanggal_relatif",
    "url",
    "angka",
    "nama_tersamar",
    "penanda_penjual",
    "simbol",
)

TRUTHY = {"ya", "y", "1", "true", "x", "setuju"}


# ============================================================
# TIMESTAMP
# ============================================================

def parse_timestamp(
    raw: str,
    formats: tuple[str, ...],
) -> tuple[pd.Timestamp, bool, str]:
    """
    Kembalikan (timestamp, punya_jam, status).
    status: ok | kosong | relatif | tak_terparsing
    Bagian setelah '|' (mis. "| Variasi: Hitam") dibuang dulu.
    Tanggal relatif ("2 minggu lalu") TIDAK ditebak -> NaT.
    """

    value = str(raw).split("|", 1)[0].strip()

    if not value:
        return pd.NaT, False, "kosong"

    if DATE_REL_RE.match(value):
        return pd.NaT, False, "relatif"

    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt)
        except ValueError:
            continue

        has_time = any(code in fmt for code in ("%H", "%M", "%S"))
        return pd.Timestamp(parsed), has_time, "ok"

    return pd.NaT, False, "tak_terparsing"


# ============================================================
# KOSAKATA TAG ASPEK (yang sudah disetujui)
# ============================================================

def load_aspect_vocab(
    path: Path | None,
) -> set[str]:
    """
    .txt : satu fragmen per baris.
    .csv : kolom `fragmen` (atau kolom pertama); bila ada kolom `setuju`,
           hanya baris bernilai ya/y/1/true/x yang dipakai.
    """

    if path is None:
        return set()

    path = Path(path)

    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        column = "fragmen" if "fragmen" in table.columns else table.columns[0]
        if "setuju" in table.columns:
            table = table[table["setuju"].str.strip().str.casefold().isin(TRUTHY)]
        values = table[column].tolist()
    else:
        values = path.read_text(encoding="utf-8-sig").splitlines()

    return {
        normalize_fragment_key(value)
        for value in values
        if normalize_fragment_key(value)
    }


# ============================================================
# HASIL
# ============================================================

@dataclass
class BuildResult:
    reviews: pd.DataFrame
    column_map: pd.DataFrame
    manifest: pd.DataFrame
    duplicate_files: pd.DataFrame
    fragment_counts: pd.DataFrame
    light_changes: dict[str, int]
    review_fragments: pd.DataFrame
    excluded_files: pd.DataFrame
    skipped_no_review: list[str] = field(default_factory=list)


# ============================================================
# SATU FILE
# ============================================================

def _review_id(product_id: str, file_tag: str, row_order: int) -> str:
    return f"{product_id}__{file_tag}__{row_order}"


def file_tag(name: str) -> str:
    """Tag pendek & stabil dari nama file (sha256 nama, 8 heksa)."""

    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]


def build_file_rows(
    file_row: pd.Series,
    df: pd.DataFrame,
    column_profiles: pd.DataFrame,
    config: CleaningConfig,
    aspect_vocab: set[str],
) -> tuple[list[dict], dict, Counter, list[dict]]:
    """
    Ubah satu CSV menjadi baris review.
    Kembalikan (baris_output, statistik_file, hitungan_fragmen, fragmen_teks).
    """

    source_file = file_row["file"]
    product_id = file_row["product_key"]
    tag = file_tag(source_file)

    roles = dict(zip(column_profiles["column"], column_profiles["role"]))

    review_cols = [c for c in df.columns if roles.get(c) == ROLE_REVIEW]
    aspect_cols = [c for c in df.columns if roles.get(c) == ROLE_ASPECT]
    ts_cols = [c for c in df.columns if roles.get(c) == ROLE_TIMESTAMP]
    ts_col = ts_cols[0] if ts_cols else None
    other_cols = [
        c for c in df.columns
        if roles.get(c) not in (ROLE_REVIEW, ROLE_ASPECT, ROLE_TIMESTAMP)
    ]

    counts: Counter = Counter()
    rows: list[dict] = []
    fragments_log: list[dict] = []

    n_empty_cells = 0
    n_only_non_text = 0

    values = df.to_dict(orient="list")

    for row_order in range(len(df)):

        review_fragments: list[str] = []
        fragment_cols: list[str] = []
        aspect_tags: list[str] = []
        had_nonempty_review_cell = False

        for col in df.columns:

            role = roles.get(col)
            value = str(values[col][row_order]).strip()

            if role == ROLE_REVIEW:
                kind = classify_value(value)

                if kind == "kosong":
                    continue

                had_nonempty_review_cell = True

                if kind == "teks":
                    if aspect_vocab and normalize_fragment_key(value) in aspect_vocab:
                        aspect_tags.append(value)
                        counts[("review_text", "tag_aspek_kosakata")] += 1
                        continue

                    review_fragments.append(value)
                    fragment_cols.append(col)
                    counts[("review_text", "teks_masuk_review")] += 1
                    fragments_log.append(
                        {"source_file": source_file, "column": col, "fragment": value}
                    )

                elif kind == "label_aspek":
                    aspect_tags.append(value)
                    counts[("review_text", "label_aspek")] += 1

                else:
                    counts[("review_text", kind)] += 1

            elif role == ROLE_ASPECT:
                if value:
                    aspect_tags.append(value)
                    counts[("aspect_tag", classify_value(value))] += 1

            elif col in other_cols and value:
                counts[(role, classify_value(value))] += 1

        if not review_fragments:
            if had_nonempty_review_cell:
                n_only_non_text += 1
            else:
                n_empty_cells += 1
            continue

        ts_raw = str(values[ts_col][row_order]) if ts_col else ""
        if ts_col:
            ts_value, has_time, ts_status = parse_timestamp(ts_raw, config.timestamp_formats)
        else:
            ts_value, has_time, ts_status = pd.NaT, False, "tidak_ada_kolom"

        rows.append(
            {
                "review_id": _review_id(product_id, tag, row_order),
                "product_id": product_id,
                "nama_toko": file_row["nama_toko"],
                "nama_produk": file_row["nama_produk"],
                "review": join_review(review_fragments, config.fragment_separator),
                "_light_input": join_for_light(review_fragments),
                "aspect_tags": " | ".join(aspect_tags),
                "timestamp": ts_value,
                "timestamp_has_time": has_time,
                "timestamp_raw": ts_raw,
                "timestamp_status": ts_status,
                "row_order": row_order,
                "source_file": source_file,
                "fragment_cols": " | ".join(fragment_cols),
                "n_fragments": len(review_fragments),
                # hanya ditandai: sisa balasan penjual di kolom campuran
                "seller_reply_suspect": any(
                    SELLER_STYLE_RE.search(fragment) for fragment in review_fragments
                ),
            }
        )

    stats = {
        "n_baris_sumber": len(df),
        "n_baris_keluar": len(rows),
        "n_baris_kosong": n_empty_cells + n_only_non_text,
        "n_kosong_semua_sel_teks_kosong": n_empty_cells,
        "n_kosong_hanya_fragmen_non_teks": n_only_non_text,
        "timestamp_col": ts_col or "",
        "review_cols": " | ".join(review_cols),
        "n_review_cols": len(review_cols),
        "aspect_cols": " | ".join(aspect_cols),
    }

    return rows, stats, counts, fragments_log


# ============================================================
# FLAG TINGKAT TABEL
# ============================================================

def add_flags(
    reviews: pd.DataFrame,
    config: CleaningConfig,
) -> pd.DataFrame:

    df = reviews.sort_values(
        ["product_id", "source_file", "row_order"],
        kind="mergesort",
    ).reset_index(drop=True)

    df["n_chars"] = df["text_light"].str.len().astype("int64")
    df["n_words"] = df["text_light"].str.split().str.len().fillna(0).astype("int64")
    df["is_empty"] = df["text_light"] == ""
    df["is_short"] = ~df["is_empty"] & (df["n_chars"] < config.min_review_length)

    # --------------------------------------------------------
    # Duplikat teknis: product + teks + timestamp (dengan jam) sama.
    # Kemunculan pertama (urut source_file, row_order) bukan duplikat.
    # --------------------------------------------------------

    eligible = ~df["is_empty"] & df["timestamp"].notna()
    if config.technical_dup_require_time:
        eligible &= df["timestamp_has_time"]

    tech_key = ["product_id", "text_light", "timestamp"]
    df["is_technical_duplicate"] = False
    df.loc[eligible, "is_technical_duplicate"] = (
        df.loc[eligible].duplicated(subset=tech_key, keep="first")
    )

    # --------------------------------------------------------
    # Duplikat teks: teks sama dalam produk yang sama, setelah
    # duplikat teknis disisihkan, masih >= 2 baris -> ditandai saja.
    # --------------------------------------------------------

    text_key = ["product_id", "text_light"]
    non_empty = ~df["is_empty"]

    df["dup_group_size"] = 0
    df.loc[non_empty, "dup_group_size"] = (
        df.loc[non_empty].groupby(text_key)["review_id"].transform("size")
    )

    distinct = non_empty & ~df["is_technical_duplicate"]
    n_distinct = (
        df.loc[distinct].groupby(text_key).size().rename("_distinct_n")
    )
    df = df.join(n_distinct, on=text_key)
    df["is_text_duplicate"] = non_empty & (df["_distinct_n"].fillna(0) >= 2)
    df = df.drop(columns="_distinct_n")

    df["dup_group_size"] = df["dup_group_size"].astype("int64")

    df["include_for_inference"] = (
        ~df["is_empty"] & ~df["is_short"] & ~df["is_technical_duplicate"]
    )

    # --------------------------------------------------------
    # Urutan dalam produk: timestamp (NaT di akhir), lalu
    # source_file, lalu row_order -> deterministik.
    # --------------------------------------------------------

    df = df.sort_values(
        ["product_id", "timestamp", "source_file", "row_order"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    df["seq_in_product"] = (df.groupby("product_id").cumcount() + 1).astype("int64")

    return df


def count_date_only_duplicate_candidates(
    df: pd.DataFrame,
) -> int:
    """Info saja: teks+tanggal sama tetapi timestamp tanpa jam (tidak ditandai teknis)."""

    subset = df[~df["is_empty"] & df["timestamp"].notna() & ~df["timestamp_has_time"]]
    return int(subset.duplicated(subset=["product_id", "text_light", "timestamp"]).sum())


# ============================================================
# SELURUH DATA
# ============================================================

def build_all(
    config: CleaningConfig,
    file_map: FileMap,
    log=print,
) -> BuildResult:

    aspect_vocab = load_aspect_vocab(config.aspect_tag_vocab_path)
    raw_dir = Path(config.raw_dir)

    # --------------------------------------------------------
    # 1. Integritas file (hash)
    # --------------------------------------------------------

    hashes = pd.DataFrame(
        {
            "file": file_map.ok["file"],
            "product_key": file_map.ok["product_key"],
            "sha256": [file_sha256(raw_dir / name) for name in file_map.ok["file"]],
            "ukuran_byte": [(raw_dir / name).stat().st_size for name in file_map.ok["file"]],
        }
    )

    duplicate_files = build_duplicate_file_groups(hashes)
    dup_group_of = dict(zip(duplicate_files["file"], duplicate_files["file_dup_group"]))

    excluded_files = file_map.excluded[["file", "product_key", "status", "alasan"]].copy()

    files_to_process = file_map.ok.copy()

    if config.exclude_duplicate_file_groups and not duplicate_files.empty:
        keep_first = duplicate_files.groupby("file_dup_group")["file"].transform("min")
        drop = duplicate_files.loc[duplicate_files["file"] != keep_first, "file"]
        extra = file_map.ok[file_map.ok["file"].isin(drop)][["file", "product_key", "status"]].copy()
        extra["alasan"] = "isi identik dengan file lain (exclude_duplicate_file_groups=True)"
        excluded_files = pd.concat([excluded_files, extra], ignore_index=True)
        files_to_process = files_to_process[~files_to_process["file"].isin(drop)]

    # --------------------------------------------------------
    # 2. Peran kolom + baris per file
    # --------------------------------------------------------

    all_profiles = []
    all_rows: list[dict] = []
    manifest_rows: list[dict] = []
    fragment_counts: Counter = Counter()
    fragments_log: list[dict] = []
    skipped_no_review: list[str] = []

    sha_of = dict(zip(hashes["file"], hashes["sha256"]))

    for _, file_row in files_to_process.iterrows():

        name = file_row["file"]
        df, encoding = read_raw_csv(raw_dir / name)
        family = detect_family(list(df.columns))

        profiles = map_file_columns(name, df, config)
        profile_frame = profiles_to_frame(profiles)
        profile_frame.insert(1, "product_key", file_row["product_key"])
        profile_frame.insert(2, "schema_family", family)
        all_profiles.append(profile_frame)

        base = {
            "file": name,
            "product_key": file_row["product_key"],
            "nama_toko": file_row["nama_toko"],
            "nama_produk": file_row["nama_produk"],
            "schema_family": family,
            "encoding": encoding,
            "sha256": sha_of[name],
            "file_dup_group": dup_group_of.get(name, ""),
            "n_kolom": df.shape[1],
            "n_kolom_perlu_cek": int((profile_frame["role"] == "perlu_cek").sum()),
        }

        if not (profile_frame["role"] == ROLE_REVIEW).any():
            skipped_no_review.append(name)
            manifest_rows.append(
                {
                    **base,
                    "status": "dilewati: tidak ada kolom review_text",
                    "n_baris_sumber": len(df),
                    "n_baris_keluar": 0,
                    "n_baris_kosong": len(df),
                }
            )
            continue

        rows, stats, counts, frag_log = build_file_rows(
            file_row, df, profile_frame, config, aspect_vocab
        )

        for row in rows:
            row["file_dup_group"] = base["file_dup_group"]
            row["schema_family"] = family

        all_rows.extend(rows)
        fragment_counts.update(counts)
        fragments_log.extend(frag_log)
        manifest_rows.append({**base, "status": "diproses", **stats})

    column_map = pd.concat(all_profiles, ignore_index=True)

    # --------------------------------------------------------
    # 3. Tabel review + text_light + flag
    # --------------------------------------------------------

    reviews = pd.DataFrame(
        all_rows,
        columns=[
            c for c in OUTPUT_COLUMNS
            if c not in ("text_light", "n_chars", "n_words", "is_empty", "is_short",
                         "is_technical_duplicate", "is_text_duplicate",
                         "dup_group_size", "include_for_inference", "seq_in_product")
        ] + ["_light_input"],
    )

    text_light, light_changes = clean_light_series(
        reviews["_light_input"], nfkc=config.unicode_nfkc
    )
    reviews["text_light"] = text_light
    reviews = reviews.drop(columns="_light_input")

    reviews["timestamp"] = pd.to_datetime(reviews["timestamp"]).astype("datetime64[ns]")

    reviews = add_flags(reviews, config)
    reviews = reviews[OUTPUT_COLUMNS]

    # --------------------------------------------------------
    # 4. Manifest per file (flag & % NaT)
    # --------------------------------------------------------

    manifest = pd.DataFrame(manifest_rows)

    per_file = reviews.groupby("source_file").agg(
        n_is_empty=("is_empty", "sum"),
        n_is_short=("is_short", "sum"),
        n_technical_duplicate=("is_technical_duplicate", "sum"),
        n_text_duplicate=("is_text_duplicate", "sum"),
        n_seller_reply_suspect=("seller_reply_suspect", "sum"),
        n_include=("include_for_inference", "sum"),
        n_timestamp_nat=("timestamp", lambda s: int(s.isna().sum())),
        n_ts_relatif=("timestamp_status", lambda s: int((s == "relatif").sum())),
        n_ts_tak_terparsing=("timestamp_status", lambda s: int((s == "tak_terparsing").sum())),
        n_ts_dengan_jam=("timestamp_has_time", "sum"),
    )

    manifest = manifest.merge(per_file, left_on="file", right_index=True, how="left")

    count_cols = list(per_file.columns)
    manifest[count_cols] = manifest[count_cols].fillna(0).astype("int64")

    manifest["pct_timestamp_nat"] = np.where(
        manifest["n_baris_keluar"] > 0,
        (100 * manifest["n_timestamp_nat"] / manifest["n_baris_keluar"].clip(lower=1)).round(2),
        0.0,
    )
    manifest["cek_seimbang"] = (
        manifest["n_baris_sumber"] == manifest["n_baris_kosong"] + manifest["n_baris_keluar"]
    )

    manifest = manifest.sort_values("file", kind="mergesort").reset_index(drop=True)

    fragment_frame = pd.DataFrame(
        [
            {"peran_kolom": role, "jenis_nilai": kind, "jumlah_sel": n}
            for (role, kind), n in fragment_counts.items()
        ]
    ).sort_values(["peran_kolom", "jenis_nilai"], kind="mergesort").reset_index(drop=True)

    log(
        f"[build] file diproses={len(files_to_process) - len(skipped_no_review)}, "
        f"baris output={len(reviews):,}"
    )

    return BuildResult(
        reviews=reviews,
        column_map=column_map,
        manifest=manifest,
        duplicate_files=duplicate_files,
        fragment_counts=fragment_frame,
        light_changes=light_changes,
        review_fragments=pd.DataFrame(fragments_log),
        excluded_files=excluded_files.sort_values("file", kind="mergesort").reset_index(drop=True),
        skipped_no_review=skipped_no_review,
    )
