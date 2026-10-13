"""Audit lokal parquet proposal dan penggantian slang; tidak mengubah tabel masukan.

Konteks contoh hanya berasal dari kolom review pengguna, bukan konteks kamus asal.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from .config import PROTECTED_TERMS
from .proposal import _phrase_pattern, _proposal_transformer, load_slang_dictionary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_value(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def trace_proposal(series: pd.Series, slang: dict[str, str]) -> tuple[pd.Series, dict, Counter, dict]:
    """Jalankan transformasi yang sama, hitung kejadian dan posisi ulasan per pasangan."""

    frequencies: Counter = Counter()
    positions: dict = defaultdict(set)
    position = -1

    def on_slang(key: str, baku: str, original: str, replacement: str) -> None:
        frequencies[(key, baku)] += 1
        positions[(key, baku)].add(position)

    transform = _proposal_transformer(slang, PROTECTED_TERMS, on_slang=on_slang)
    counts: Counter = Counter()
    texts = []
    for position, value in enumerate(series):
        result, changes = transform(value)
        texts.append(result)
        counts.update(changes)
    result = pd.Series(texts, index=series.index, name="text_proposal", dtype="object")
    replacements = counts.pop("jumlah_slang_dinormalkan", 0)
    if replacements != sum(frequencies.values()):
        raise ValueError("Jumlah jejak slang berbeda dari hitungan transformasi")
    stats = {
        "n_baris": len(series),
        "n_berubah": int((result != series.fillna("").astype(str)).sum()),
        "n_kosong": int(result.eq("").sum()),
        "slang_dijalankan": True,
        "jumlah_slang_dinormalkan": replacements,
        "jumlah_pasangan_slang_terpakai": len(frequencies),
        "baris_berubah_per_aturan": dict(counts),
    }
    return result, stats, frequencies, dict(positions)


def count_protected(original: pd.Series, updated: pd.Series) -> list[dict]:
    """Hitung tiap istilah sendiri: ori juga dihitung dalam frasa bukan ori."""

    rows = []
    before_texts = original.map(text_value)
    after_texts = updated.map(text_value)
    for term in PROTECTED_TERMS:
        pattern = _phrase_pattern([term])
        before = before_texts.map(lambda text: sum(1 for _ in pattern.finditer(text)))
        after = after_texts.map(lambda text: sum(1 for _ in pattern.finditer(text)))
        lost = (before - after).clip(lower=0)
        gained = (after - before).clip(lower=0)
        rows.append({
            "istilah": term, "sebelum": int(before.sum()), "sesudah": int(after.sum()),
            "kemunculan_hilang_per_baris": int(lost.sum()),
            "baris_kehilangan": int(lost.gt(0).sum()),
            "kemunculan_bertambah_per_baris": int(gained.sum()),
            "baris_bertambah": int(gained.gt(0).sum()),
        })
    return rows


def make_top_replacements(original: pd.DataFrame, updated: pd.DataFrame,
                          frequencies: Counter, positions: dict) -> pd.DataFrame:
    """Top 100; 3 posisi ulasan unik tanpa pengembalian, seed pandas 42 per pasangan."""

    columns = ["slang", "baku", "frekuensi_penggantian", "jumlah_ulasan_terdampak", "jumlah_contoh"]
    for number in range(1, 4):
        columns.extend(f"contoh_{number}_{field}" for field in
                       ("posisi_baris", "review_id", "seq_in_product", "review", "text_proposal"))
    records = []
    for pair, count in sorted(frequencies.items(), key=lambda item: (-item[1], *item[0]))[:100]:
        affected = sorted(positions[pair])
        chosen = pd.Series(affected, dtype="int64").sample(n=min(3, len(affected)), random_state=42)
        record = dict.fromkeys(columns, "")
        record.update(slang=pair[0], baku=pair[1], frekuensi_penggantian=count,
                      jumlah_ulasan_terdampak=len(affected), jumlah_contoh=len(chosen))
        for number, position in enumerate(chosen.tolist(), start=1):
            source = original.iloc[position]
            record[f"contoh_{number}_posisi_baris"] = position
            for field in ("review_id", "seq_in_product", "review"):
                record[f"contoh_{number}_{field}"] = text_value(source[field])
            record[f"contoh_{number}_text_proposal"] = text_value(updated.iloc[position]["text_proposal"])
        records.append(record)
    return pd.DataFrame.from_records(records, columns=columns)


def verify_tables(original: pd.DataFrame, updated: pd.DataFrame, slang: dict[str, str],
                  expected_rows: int | None = None, expected_columns: int | None = None
                  ) -> tuple[dict, pd.DataFrame]:
    if expected_rows is not None and len(original) != expected_rows:
        raise ValueError(f"Jumlah baris input {len(original)} != {expected_rows}")
    if expected_columns is not None and len(original.columns) != expected_columns:
        raise ValueError(f"Jumlah kolom input {len(original.columns)} != {expected_columns}")
    expected_names = list(original.columns) + ["text_proposal", "text_proposal_kosong"]
    if list(updated.columns) != expected_names:
        raise ValueError("Nama/urutan kolom v2 tidak sama dengan seluruh kolom lama + dua kolom proposal")
    pd.testing.assert_frame_equal(original, updated[list(original.columns)], check_exact=True)
    if "text_light" not in original:
        raise ValueError("Kolom audit wajib tidak tersedia: text_light")
    pd.testing.assert_series_equal(original["text_light"], updated["text_light"], check_exact=True)
    text_lex_available = "text_lex" in original
    if text_lex_available:
        pd.testing.assert_series_equal(original["text_lex"], updated["text_lex"], check_exact=True)
    pd.testing.assert_frame_equal(original[["review_id", "seq_in_product"]],
                                  updated[["review_id", "seq_in_product"]], check_exact=True)
    regenerated, stats, frequencies, positions = trace_proposal(original["review"], slang)
    pd.testing.assert_series_equal(regenerated, updated["text_proposal"], check_exact=True)
    expected_empty = regenerated.eq("").rename("text_proposal_kosong")
    pd.testing.assert_series_equal(expected_empty, updated["text_proposal_kosong"], check_exact=True)
    protected = count_protected(original["review"], updated["text_proposal"])
    lost = [row for row in protected if row["kemunculan_hilang_per_baris"]]
    if lost:
        raise ValueError("PROTECTED_TERMS kehilangan kemunculan pada baris asal: "
                         + json.dumps(lost, ensure_ascii=False))
    included = original["include_for_inference"].eq(True).fillna(False)
    technical = original["is_technical_duplicate"].eq(True).fillna(False)
    duplicate = original["is_text_duplicate"].eq(True).fillna(False)
    stats.update({
        "n_kolom_lama": len(original.columns), "n_kolom_v2": len(updated.columns),
        "jumlah_baris_identik": len(original) == len(updated),
        "urutan_review_id_seq_identik": True, "seluruh_kolom_lama_identik": True,
        "dtype_kolom_lama_identik": True, "text_light_identik": True,
        "text_lex_tersedia": text_lex_available,
        "text_lex_identik": True if text_lex_available else None,
        "regenerasi_text_proposal_identik": True, "flag_kosong_identik": True,
        "n_include_for_inference": int(included.sum()),
        "n_technical_duplicate": int(technical.sum()),
        "n_technical_duplicate_masih_include": int((technical & included).sum()),
        "n_text_duplicate": int(duplicate.sum()),
        "n_text_duplicate_masih_include": int((duplicate & included).sum()),
        "n_kosong_masih_include": int((expected_empty & included).sum()),
        "kolom_lama_diverifikasi": list(original.columns), "protected_terms": protected,
    })
    return stats, make_top_replacements(original, updated, frequencies, positions)


def format_report(stats: dict, top: pd.DataFrame, paths: dict, hashes: dict) -> str:
    lines = ["# Verifikasi parquet proposal v2 — kamus alay", "", "## Artefak dan SHA-256", ""]
    for label, path in paths.items():
        lines.append(f'- {label}: "{path}"' + (f'; SHA-256 `{hashes[label]}`' if label in hashes else ""))
    lines += ["", "## Terverifikasi", "",
              "Nilai, dtype, indeks, nama serta urutan seluruh kolom lama dibandingkan persis. "
              "Keluaran proposal diregenerasi memakai fungsi preprocessing yang sama dengan jejak "
              "setiap penggantian slang aktual, lalu dibandingkan dengan parquet v2.", ""]
    for key, value in stats.items():
        if key not in ("protected_terms", "baris_berubah_per_aturan", "kolom_lama_diverifikasi"):
            lines.append(f"- {key}: {value}")
    lines += ["", "Kolom lama yang diperiksa: " + ", ".join(f"`{name}`" for name in stats["kolom_lama_diverifikasi"])
              + "."]
    if not stats["text_lex_tersedia"]:
        lines += ["", "`text_lex` tidak tersedia pada input maupun v2. Tidak ada kolom yang dibuat "
                  "untuk menggantikannya, dan perbandingan nilainya tidak dapat dilakukan "
                  "(`text_lex_identik=null`). Seluruh kolom lama yang benar-benar tersedia tetap "
                  "dibandingkan persis."]
    lines += ["", "## Per aturan", "", "| Aturan | Baris berubah |", "| --- | ---: |"]
    lines.extend(f"| {rule} | {count} |" for rule, count in stats["baris_berubah_per_aturan"].items())
    lines += ["", "Satu ulasan dapat berubah pada beberapa aturan; hitungan baris antartahap tidak dijumlahkan.",
              "", "## PROTECTED_TERMS", "",
              "Pencocokan masing-masing istilah tidak peka kapital, pada batas kata, dan menerima spasi "
              "fleksibel. Setiap istilah dihitung sendiri: `ori` dalam `bukan ori` ikut dihitung. "
              "Kehilangan diperiksa per baris agar kenaikan pada ulasan lain tidak menutup kehilangan. "
              "Kenaikan dapat terjadi ketika pembersihan simbol/typo membentuk batas kata atau ejaan "
              "yang cocok; kenaikan tidak berarti istilah asal dihapus.", "",
              "| Istilah | Sebelum | Sesudah | Hilang per baris | Baris kehilangan | Bertambah per baris |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in stats["protected_terms"]:
        lines.append("| " + " | ".join(str(row[key]) for key in (
            "istilah", "sebelum", "sesudah", "kemunculan_hilang_per_baris", "baris_kehilangan",
            "kemunculan_bertambah_per_baris")) + " |")
    lines += ["", "## Tinjauan manual penggantian slang", "",
              f'- CSV: "{paths["top_csv"]}"; {len(top)} pasangan dari '
              f'{stats["jumlah_pasangan_slang_terpakai"]} pasangan yang terpakai.',
              "- Peringkat berdasarkan frekuensi penggantian aktual menurun, lalu slang dan baku menaik.",
              "- Contoh berasal hanya dari review pengguna. Tiga posisi ulasan berbeda dipilih tanpa "
              "pengembalian memakai `random_state=42` per pasangan, setelah posisi diurutkan.",
              "- Jika pasangan muncul pada kurang dari tiga ulasan, kolom contoh sisanya kosong; "
              "tidak ada konteks buatan. Posisi baris menggunakan indeks posisi mulai nol.",
              f'- Pasangan top-100 dengan kurang dari tiga ulasan: '
              f'{int(top["jumlah_contoh"].lt(3).sum()) if len(top) else 0}.',
              "- Frekuensi menghitung kejadian; satu ulasan dapat menyumbang beberapa kejadian pasangan sama.",
              "", "## Sumber dan batasan", "",
              'Kamus lokal mengikuti "references/slang_PROVENANCE.md" dan profil '
              '"reports/profil_kamus_alay.md". Sumber: Salsabila dkk. (2018), ekspor loader SEACrowd; '
              'lisensi dinyatakan Unknown. Audit ini tidak mengunduh sumber atau memuat kolom context kamus. '
              "Audit kesetaraan data tidak membuktikan kualitas linguistik normalisasi; contoh perlu ditinjau.", ""]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verifikasi v2 dan top 100 penggantian slang secara lokal")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Parquet v2 yang dibaca, tidak ditulis")
    parser.add_argument("--slang-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("reports/verifikasi_proposal_v2_alay.md"))
    parser.add_argument("--top-csv", type=Path, default=Path("reports/top_penggantian_slang.csv"))
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-columns", type=int)
    return parser


def run(args: argparse.Namespace) -> int:
    paths = {"input": args.input.expanduser(), "output": args.output.expanduser(),
             "slang_csv": args.slang_csv.expanduser(), "report": args.report.expanduser(),
             "top_csv": args.top_csv.expanduser()}
    try:
        if len({path.resolve() for path in paths.values()}) != len(paths):
            raise ValueError("Semua path input/output/kamus/laporan/top CSV harus berbeda")
        for key in ("report", "top_csv"):
            if paths[key].exists():
                raise ValueError(f'File sudah ada, tidak boleh ditimpa: "{paths[key]}"')
        hashes = {key: sha256(paths[key]) for key in ("input", "output", "slang_csv")}
        if args.expected_input_sha256 and hashes["input"] != args.expected_input_sha256:
            raise ValueError("SHA-256 input tidak cocok dengan nilai wajib")
        original, updated = pd.read_parquet(paths["input"]), pd.read_parquet(paths["output"])
        stats, top = verify_tables(original, updated, load_slang_dictionary(paths["slang_csv"]),
                                   args.expected_rows, args.expected_columns)
        after = sha256(paths["input"])
        if after != hashes["input"]:
            raise ValueError("SHA-256 input berubah selama audit")
        stats["sha256_input_sebelum"] = hashes["input"]
        stats["sha256_input_sesudah"] = after
        stats["sha256_input_tidak_berubah"] = True
        report = format_report(stats, top, paths, hashes)
        for key in ("report", "top_csv"):
            paths[key].parent.mkdir(parents=True, exist_ok=True)
        with paths["top_csv"].open("x", encoding="utf-8", newline="") as stream:
            top.to_csv(stream, index=False, lineterminator="\n")
        with paths["report"].open("x", encoding="utf-8") as stream:
            stream.write(report)
    except (OSError, ValueError, AssertionError, KeyError) as exc:
        print(f"BERHENTI: {exc}")
        return 2
    print(json.dumps({"verifikasi": stats, "sha256": hashes,
                      "paths": {key: str(value) for key, value in paths.items()}}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
