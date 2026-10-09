"""Fixture data mentah sintetis kecil untuk tiga keluarga skema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.cleaning.config import CleaningConfig


MAPPING_NAME = "pemetaan_file_toko_produk.csv"


# ============================================================
# CSV SINTETIS
# ============================================================

def _shopee_webscraper() -> pd.DataFrame:
    """Keluarga A: kolom `data` (teks tanpa angka), data3 = tanggal, data4 = label+teks."""

    n = 8
    return pd.DataFrame(
        {
            "web_scraper_order": [f"1785000000-{i + 1}" for i in range(n)],
            "web_scraper_start_url": ["https://shopee.co.id/hp-murah-i.1.2"] * n,
            "pagination": [""] * n,
            "data": [
                "barang bagus sesuai pesanan",
                "pengiriman cepat",
                "",
                "mantap <b>original</b> https://contoh.id/x",
                "bagussss​ banget 👍👍",
                "pengiriman cepat",
                "",
                "pengiriman cepat",
            ],
            "timestamp": [
                "2026-03-07 10:00 | Variasi: Hitam,8/256",
                "2025-09-08 11:53",
                "2025-09-09 08:00",
                "2026-01-01",
                "2 minggu lalu",
                "2025-09-08 11:53",
                "2025-09-10 09:00",
                "2025-10-01 12:00",
            ],
            "data4": [
                "Kualitas:",
                "",
                "",
                "baterai 5000mAh awet",
                "",
                "",
                "",
                "",
            ],
            "data3": [
                "a*****b",
                "Budi",
                "c*****d",
                "",
                "e*****f",
                "Budi",
                "g*****h",
                "Sari",
            ],
        }
    )


def _lazada() -> pd.DataFrame:
    """Keluarga B: time, item-content*, review-attribute (label), img-more-text."""

    return pd.DataFrame(
        {
            "time": [
                "2026-06-10",
                "3 minggu lalu",
                "2026-06-12",
                "07/03/2026",
                "2026-06-14",
                "2026-06-15",
            ],
            "item-content-main-content-reviews-item": [
                "hp nya bagus sekali",
                "barang ori, seller amanah",
                "",
                "kurang puas baterai boros",
                "harga",
                "",
            ],
            "img-more-text": ["+1", "", "+3", "", "", ""],
            "review-attribute": ["Harga:", "", "Durability:", "", "⚡Kinerja:", ""],
            "item-content-main-content-reviews-item (3)": [
                "murah",
                "",
                "awet",
                "",
                "",
                "",
            ],
        }
    )


def _shopee_random() -> pd.DataFrame:
    """Keluarga C: kolom acak + balasan penjual (QSiE2A) dengan penanda rFzVcr."""

    return pd.DataFrame(
        {
            "XYk98l": [
                "2026-07-12 18:51 | Variasi: NAVY",
                "2026-07-15 15:03",
                "2026-06-21 14:40 | Variasi: Blue",
                "2026-04-04 11:49",
                "2026-04-05 11:49",
                "2026-04-06 11:49",
            ],
            "meQyXP": ["aman", "baik", "", "oke", "", "x"],
            "f35Wh2": ["ori", "berfungsi dengan baik", "", "", "", ""],
            "K5v3lN (3)": ["Produk:", "", "", "Warna:", "", ""],
            "YNedDV": [
                "Pengiriman cepat dan aman",
                "",
                "",
                "Packing rapi",
                "",
                "",
            ],
            "rFzVcr": ["Respon Penjual:", "", "", "", "", ""],
            "QSiE2A": ["Hai kak, terima kasih atas penilaiannya", "", "", "", "", ""],
            "HcSdrS src (2)": [
                "https://down-id.img.susercontent.com/file/a",
                "",
                "https://down-id.img.susercontent.com/file/b",
                "",
                "",
                "",
            ],
        }
    )


# ============================================================
# FOLDER MENTAH + PEMETAAN
# ============================================================

def write_raw_dir(
    raw_dir: Path,
    extra_duplicate: bool = False,
) -> None:

    raw_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "Toko A_HP Satu.csv": ("Toko A", "HP Satu", _shopee_webscraper()),
        "Toko B_HP Dua.csv": ("Toko B", "HP Dua", _lazada()),
        "Toko C_HP Tiga.csv": ("Toko C", "HP Tiga", _shopee_random()),
    }

    if extra_duplicate:
        files["Toko C_HP Empat.csv"] = ("Toko C", "HP Empat", _shopee_random())

    rows = []
    for name, (store, product, frame) in files.items():
        frame.to_csv(raw_dir / name, index=False)
        rows.append(
            {
                "file": name,
                "nama_toko": store,
                "nama_produk": product,
                "product_key": f"{store.lower()}|{product.lower()}",
                "status": "ok",
                "sumber_nama": "nama file",
                "catatan": "",
                "ukuran_byte": (raw_dir / name).stat().st_size,
            }
        )

    # file yang dikecualikan
    pd.DataFrame({"x": ["abc"]}).to_csv(raw_dir / "TOKO SAJA.csv", index=False)
    rows.append(
        {
            "file": "TOKO SAJA.csv",
            "nama_toko": "",
            "nama_produk": "",
            "product_key": "",
            "status": "dikecualikan: hanya nama toko",
            "sumber_nama": "aturan pengguna",
            "catatan": "",
            "ukuran_byte": 0,
        }
    )

    # baris yang merujuk file pemetaan itu sendiri (kasus nyata 2026-10-09)
    rows.append(
        {
            "file": MAPPING_NAME,
            "nama_toko": "pemetaan_file_toko",
            "nama_produk": "produk",
            "product_key": "pemetaan_file_toko|produk",
            "status": "ok",
            "sumber_nama": "underscore terakhir",
            "catatan": "",
            "ukuran_byte": 0,
        }
    )

    pd.DataFrame(rows).to_csv(raw_dir / MAPPING_NAME, index=False, encoding="utf-8-sig")


def make_config(
    raw_dir: Path,
    output_dir: Path,
    **overrides,
) -> CleaningConfig:

    n_ok = len(
        [
            name for name in raw_dir.glob("*.csv")
            if name.name not in (MAPPING_NAME, "TOKO SAJA.csv")
        ]
    )

    params = {
        "raw_dir": raw_dir,
        "output_dir": output_dir,
        "expected_mapping_rows": n_ok + 1,
        "expected_ok_files": n_ok,
    }
    params.update(overrides)

    return CleaningConfig(**params)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    path = tmp_path / "raw"
    write_raw_dir(path)
    return path
