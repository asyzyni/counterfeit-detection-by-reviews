"""Konversi ekspor SEACrowd lokal menjadi slang,baku; tanpa jaringan.

CSV context tidak pernah disalin. File keluaran yang sudah ada ditolak.
Exit 2 setelah menulis bahan tinjauan bila >30% kunci unik dikeluarkan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .config import PROTECTED_TERMS


SOURCE_HEADER = ["slang", "formal", "in_dictionary", "context", "categories"]
REASONS = ("kosong", "duplikat_pasangan", "slang_sama_baku", "protected_terms",
           "multi_baku", "kunci_satu_huruf", "kunci_dua_huruf")
MOJIBAKE_MARKERS = ("\ufffd", "Ã", "Â", "â€", "ðŸ", "ï»¿")


def normalize_space(value: str) -> str:
    return " ".join(value.split())


def normalized_pairs(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    return [(normalize_space(row["slang"]).casefold(), normalize_space(row["formal"]))
            for row in rows]


def alternatives(pairs: list[tuple[str, str]]) -> dict[str, set[str]]:
    choices: dict[str, set[str]] = defaultdict(set)
    for key, value in pairs:
        if key and value:
            choices[key].add(value)
    return choices


def convert_rows(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict], dict]:
    """Satu alasan utama per baris; konflik dideteksi sebelum filter apa pun.

    Baku dirapikan spasinya tanpa casefold paksa. Pemeriksaan identitas dan
    substring terlindungi tidak peka kapital. Panjang kunci dihitung karakter.
    Duplikat = pasangan setelah normalisasi, bukan duplikat komentar/konteks.
    """
    pairs = normalized_pairs(rows)
    choices = alternatives(pairs)
    ambiguous = {key for key, values in choices.items() if len(values) > 1}
    keys = {key for key, _ in pairs if key}
    protected = tuple(normalize_space(term).casefold() for term in PROTECTED_TERMS)
    seen: set[tuple[str, str]] = set()
    accepted, excluded = [], []
    counts = dict.fromkeys(REASONS, 0)
    for key, value in pairs:
        reason = None
        if not key or not value:
            reason = "kosong"
        elif (key, value) in seen:
            reason = "duplikat_pasangan"
        else:
            seen.add((key, value))
            if key == value.casefold():
                reason = "slang_sama_baku"
            elif any(term in key or term in value.casefold() for term in protected):
                reason = "protected_terms"
            elif key in ambiguous:
                reason = "multi_baku"
            elif len(key) == 1:
                reason = "kunci_satu_huruf"
            elif len(key) == 2:
                reason = "kunci_dua_huruf"
        if reason:
            counts[reason] += 1
            excluded.append({"slang": key, "baku": value, "alasan": reason})
        else:
            accepted.append({"slang": key, "baku": value})
    accepted.sort(key=lambda row: (row["slang"], row["baku"]))
    excluded.sort(key=lambda row: (row["slang"], row["baku"], row["alasan"]))
    lost = keys - {row["slang"] for row in accepted}
    stats = {
        "n_baris": len(rows), "n_kunci_unik": len(keys),
        "n_pasangan_unik": len(set(pairs)), "n_kunci_multi_baku": len(ambiguous),
        "n_diterima": len(accepted), "n_baris_dikecualikan": len(excluded),
        "n_kunci_dikeluarkan": len(lost),
        "persen_kunci_dikeluarkan": 100 * len(lost) / len(keys) if keys else 0.0,
        # Perbandingan bilangan bulat: tepat 30% tidak memicu gerbang.
        "gerbang_30_persen": 10 * len(lost) > 3 * len(keys),
        "alasan": counts,
    }
    assert len(accepted) + len(excluded) == len(rows)
    assert len({row["slang"] for row in accepted}) == len(accepted)
    return accepted, excluded, stats


def read_source(path: Path) -> tuple[bytes, list[dict[str, str]]]:
    data = path.read_bytes()
    # Pertahankan newline di dalam field berkutip; jangan gunakan splitlines().
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""), strict=True)
    if reader.fieldnames != SOURCE_HEADER:
        raise ValueError(f"Header sumber harus {SOURCE_HEADER}")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Jumlah kolom record CSV tidak sesuai header")
    if not rows:
        raise ValueError("CSV sumber kosong")
    return data, rows


def csv_text(rows: list[dict], fields: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def cell(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(
        ">", "&gt;").replace("|", "&#124;").replace("\n", "<br>").replace("\r", "")


def format_profile(data: bytes, rows: list[dict], accepted: list[dict], excluded: list[dict],
                   stats: dict, input_path: Path, output_path: Path, excluded_path: Path) -> str:
    pairs = normalized_pairs(rows)
    choices = alternatives(pairs)
    ambiguous = {key: sorted(values) for key, values in sorted(choices.items()) if len(values) > 1}
    raw_pairs = {(row["slang"], row["formal"]) for row in rows}
    keys_trim = {row["slang"].strip().casefold() for row in rows if row["slang"].strip()}
    keys = {key for key, _ in pairs if key}
    one, two = sorted(key for key in keys if len(key) == 1), sorted(key for key in keys if len(key) == 2)
    key_counts = Counter(key for key, _ in pairs)
    contexts: dict[str, set[str]] = defaultdict(set)
    for (key, _), row in zip(pairs, rows):
        contexts[key].add(row["context"])
    non_ascii = sorted({(key, value) for key, value in raw_pairs if any(ord(c) > 127 for c in key + value)})
    suspicious = sorted({(key, value) for key, value in raw_pairs
                         if any(marker in key + value for marker in MOJIBAKE_MARKERS)})
    identities = {key for key, value in pairs if key and key == value.casefold()}
    lines = ["# Profil dan penyaringan kamus alay", "",
             f'- Sumber lokal: "{input_path}"; SHA-256: `{hashlib.sha256(data).hexdigest()}`.',
             "- Encoding terverifikasi: UTF-8 strict; parser CSV strict dengan newline dipertahankan.",
             "- Header terverifikasi: `slang,formal,in_dictionary,context,categories`.",
             f'- Record CSV: **{len(rows)}**; karakter LF: **{data.count(bytes([10]))}**.',
             "  Hitungan LF bukan jumlah record: field komentar berkutip dapat memuat newline.",
             f'- Slang unik casefold+trim: **{len(keys_trim)}**; setelah rapikan seluruh spasi: **{len(keys)}**.',
             f'- Pasangan unik mentah: **{len(raw_pairs)}**; setelah normalisasi: **{len(set(pairs))}**.',
             f'- Kunci multi-baku: **{len(ambiguous)}**; pasangan terkait: **{sum(map(len, ambiguous.values()))}**.',
             f'- Kunci sama dengan baku (tidak peka kapital): **{len(identities)}**.',
             f'- Kunci 1 karakter: **{len(one)}**; 2 karakter: **{len(two)}** (sebelum filter lain).',
             f'- Baris slang/formal kosong setelah normalisasi: **{sum(not k or not v for k, v in pairs)}**.',
             f'- Kunci yang muncul >1 record: **{sum(n > 1 for n in key_counts.values())}**; '
             f'kunci dengan >1 context berbeda: **{sum(len(v) > 1 for v in contexts.values())}**.',
             "  Ini memverifikasi adanya pengulangan kunci dalam berbagai konteks; tidak membuktikan satu record "
             "selalu satu konteks unik. Isi context tidak disalin.",
             "- Angka 3.592 kata dari halaman HF (informasi pengguna) tidak sama dengan ekspor lokal ini; "
             "penyebab perbedaan tidak disimpulkan tanpa bukti sumber lain.", "",
             "## Encoding dan penanda mojibake", "",
             f'- Pasangan slang/formal non-ASCII: {len(non_ascii)}.',
             f'- Pasangan dengan penanda {MOJIBAKE_MARKERS!r}: {len(suspicious)}.',
             "- Pemeriksaan penanda bersifat heuristik. Untuk ekspor ini, bila seluruh slang/formal ASCII, "
             "tidak ditemukan persoalan decoding non-ASCII pada dua kolom tersebut.", "",
             "## Nilai in_dictionary yang teramati", "",
             "Makna kolom belum dibuktikan oleh isi CSV; nilai tidak digunakan menyaring.", "",
             "| Nilai literal | Record |", "| --- | ---: |"]
    for value, count in sorted(Counter(row["in_dictionary"] for row in rows).items()):
        lines.append(f"| {cell(value)} | {count} |")
    category_counts = Counter(row["categories"] for row in rows)
    labels = sorted({label for value in category_counts for label in re.findall(r"'([^']*)'", value)})
    lines += ["", "## Categories yang teramati", "",
              f"{len(category_counts)} nilai teks berbeda; {len(labels)} label di antara tanda petik tunggal: "
              + ", ".join(f"`{cell(label)}`" for label in labels) + ".",
              "Format berisi representasi daftar/array, misalnya "
              "`['afiksasi' 'elongasi']`; tidak diparsing sebagai Python dan tidak digunakan menyaring. "
              "Ejaan label berasal dari berkas; arti kategori tidak ditebak.", "",
              "| 10 nilai paling sering | Record |", "| --- | ---: |"]
    for value, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))[:10]:
        lines.append(f"| {cell(value)} | {count} |")
    lines += ["", "## 15 contoh kunci multi-baku (urut alfabet)", "",
              "| Slang | Semua baku berbeda |", "| --- | --- |"]
    for key, values in list(ambiguous.items())[:15]:
        lines.append(f"| {cell(key)} | {cell('; '.join(values))} |")
    lines += ["", "## Aturan operasional (asumsi pengguna)", "",
              "1. Kunci casefold dan spasi dirapikan; baku hanya dirapikan spasinya (kapital dipertahankan).",
              "2. Baris kosong dibuang; pasangan identik setelah normalisasi dideduplikasi. "
              "Duplikat tetap dicatat sebagai satu baris pengecualian per record yang dibuang.",
              "3. Identitas slang/baku dibandingkan tanpa peka kapital dan dibuang.",
              "4. PROTECTED_TERMS dicocokkan sebagai substring literal tanpa peka kapital pada kedua sisi. "
              "Ini juga membuang substring insidental seperti ori dalam favorit/tutorial; bukan pencocokan batas kata.",
              "5. Konflik baku dihitung dari SEMUA pasangan nonkosong sebelum filter lain. "
              "Seluruh pemetaan kunci konflik dibuang; frekuensi tidak memilih pemenang.",
              "6. Kunci panjang satu dan dua karakter dibuang secara default; tidak ada whitelist diam-diam.",
              "7. Satu alasan utama per record, prioritas: " + " → ".join(REASONS) + ". "
              "Karena alasan dapat tumpang tindih, hitungan alasan pendek berbeda dari profil awal.",
              "8. Keluaran diurutkan deterministik; tidak memakai in_dictionary, categories, atau context untuk filter.",
              "PROTECTED_TERMS: " + ", ".join(f"`{t}`" for t in PROTECTED_TERMS) + ".", "",
              "## Rekonsiliasi penyaringan", "",
              "| Alasan utama | Record dibuang |", "| --- | ---: |"]
    for reason, count in stats["alasan"].items():
        lines.append(f"| {reason} | {count} |")
    lines += [f'| **Total dibuang** | **{len(excluded)}** |',
              f'| **Diterima (pasangan/kunci)** | **{len(accepted)}** |', "",
              f'Rekonsiliasi: {len(rows)} = {len(excluded)} + {len(accepted)}. '
              f'Pasangan unik = {len(set(pairs))}; setelah deduplikasi ada '
              f'{len(set(pairs)) - len(accepted)} pasangan unik tidak diterima.', "",
              f'Gerbang: **{stats["n_kunci_dikeluarkan"]}/{len(keys)} = '
              f'{stats["persen_kunci_dikeluarkan"]:.6f}% kunci unik dikeluarkan**. '
              f'Status >30%: **{stats["gerbang_30_persen"]}**.',
              "Penyebut adalah kunci nonkosong unik sebelum filter; pembilang adalah kunci yang tidak memiliki "
              "pemetaan diterima. Duplikasi konteks tidak dihitung sebagai kehilangan kunci.",
              f'Kamus: "{output_path}" (slang,baku); pengecualian: "{excluded_path}" (slang,baku,alasan). '
              "Kedua CSV tidak memuat context.", "",
              "## Seluruh kunci dua karakter untuk tinjauan", "",
              "Termasuk kunci yang alasan utamanya multi-baku/protected_terms. Semua ditolak pada run ini.", "",
              "| Slang | Baku | Alasan utama pasangan nonduplikat |", "| --- | --- | --- |"]
    by_pair = {(r["slang"], r["baku"]): r["alasan"] for r in excluded if r["alasan"] != "duplikat_pasangan"}
    for key in two:
        for value in sorted(choices.get(key, set())):
            lines.append(f"| {cell(key)} | {cell(value)} | {by_pair.get((key, value), '')} |")
    lines += ["", "## Kunci satu karakter", "", ", ".join(f"`{cell(k)}`" for k in one), "",
              "## Pasangan terkena PROTECTED_TERMS", "", "| Slang | Baku |", "| --- | --- |"]
    for row in excluded:
        if row["alasan"] == "protected_terms":
            lines.append(f'| {cell(row["slang"])} | {cell(row["baku"])} |')
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("references/sumber/colloquial-indonesian-lexicon.csv"))
    parser.add_argument("--output", type=Path, default=Path("references/slang.csv"))
    parser.add_argument("--excluded", type=Path, default=Path("references/slang_dikecualikan.csv"))
    parser.add_argument("--profile", type=Path, default=Path("reports/profil_kamus_alay.md"))
    parser.add_argument("--expected-sha256", help="Hash ekspor yang diwajibkan sebelum konversi")
    return parser


def run(args: argparse.Namespace) -> int:
    source, target, excluded_path, profile = (
        p.expanduser() for p in (args.input, args.output, args.excluded, args.profile))
    try:
        if len({p.resolve() for p in (source, target, excluded_path, profile)}) != 4:
            raise ValueError("Path input dan ketiga keluaran harus berbeda")
        for path in (target, excluded_path, profile):
            if path.exists():
                raise ValueError(f'File sudah ada, tidak boleh ditimpa: "{path}"')
        data, rows = read_source(source)
        digest = hashlib.sha256(data).hexdigest()
        if args.expected_sha256 and digest != args.expected_sha256.lower():
            raise ValueError(f'SHA-256 sumber berbeda: "{source}"; teramati {digest}')
        accepted, excluded, stats = convert_rows(rows)
        contents = [csv_text(accepted, ["slang", "baku"]),
                    csv_text(excluded, ["slang", "baku", "alasan"]),
                    format_profile(data, rows, accepted, excluded, stats, source, target, excluded_path)]
        for path, content in zip((target, excluded_path, profile), contents):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(content)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        print(f"BERHENTI: {exc}")
        return 2
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f'SHA-256 ekspor: {digest}')
    for path in (target, excluded_path, profile):
        print(f'Keluaran: "{path}"; SHA-256: {hashlib.sha256(path.read_bytes()).hexdigest()}')
    if stats["gerbang_30_persen"]:
        print("BERHENTI: >30% kunci unik dikeluarkan; tinjau keluaran dan tunggu keputusan pengguna sebelum v2.")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
