from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


_REPO_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass
class CleaningConfig:

    # =========================================================
    # PATH INPUT / OUTPUT
    # =========================================================

    # Folder CSV mentah (HANYA-BACA).
    raw_dir: Path = Path("/Users/asyzyni/Desktop/TA/Data Beneran")

    # Nama file pemetaan file -> toko/produk di dalam raw_dir.
    mapping_filename: str = "pemetaan_file_toko_produk.csv"
    mapping_encoding: str = "utf-8-sig"

    # Folder output BARU. Run akan berhenti bila folder ini sudah ada.
    output_dir: Path = _REPO_DIR / "outputs" / "cleaning_beneran"

    # Jumlah baris pemetaan & status "ok" yang diharapkan.
    # None = tidak diperiksa (dipakai tes / data scrape ulang).
    # Baris yang merujuk file pemetaan itu sendiri tidak dihitung.
    expected_mapping_rows: int | None = 404
    expected_ok_files: int | None = 403

    # =========================================================
    # TEKS
    # =========================================================

    min_review_length: int = 2

    # Pemisah antar-fragmen pada kolom `review`.
    fragment_separator: str = " | "

    # Normalisasi Unicode NFKC pada text_light
    # (mis. huruf "𝙗𝙖𝙜𝙪𝙨" -> "bagus"). Emoji tidak terpengaruh.
    unicode_nfkc: bool = True

    # Daftar tag aspek yang SUDAH DISETUJUI (satu fragmen per baris).
    # None = tidak ada fragmen yang dipisah berdasarkan kosakata.
    aspect_tag_vocab_path: Path | None = None

    # Kandidat tag aspek: fragmen <= N kata yang muncul di >= M file.
    aspect_candidate_max_words: int = 3
    aspect_candidate_min_files: int = 3

    # Kandidat frasa templat (segmen dipisah koma, >= N kata, >= M file).
    preset_phrase_min_words: int = 3
    preset_phrase_min_files: int = 10

    # =========================================================
    # TIMESTAMP
    # =========================================================

    # Format absolut yang diterima (dicoba berurutan, setelah
    # memotong bagian "| Variasi: ..."). Format lain -> NaT.
    timestamp_formats: Tuple[str, ...] = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    )

    # =========================================================
    # DETEKSI PERAN KOLOM (ambang profil nilai)
    # =========================================================

    role_strong_rate: float = 0.90
    role_weak_rate: float = 0.30
    masked_name_rate: float = 0.05
    seller_reply_cooccurrence: float = 0.80

    # Bila lebih dari proporsi ini file tidak punya kolom review_text,
    # run berhenti.
    max_files_without_review_rate: float = 0.05

    # =========================================================
    # DUPLIKAT
    # =========================================================

    # Duplikat teknis hanya bila timestamp memuat jam (sesuai spesifikasi).
    technical_dup_require_time: bool = True

    # Bila True: dari tiap grup file ber-hash sama, hanya file pertama
    # (urut nama) yang diproses; sisanya dicatat sebagai dikecualikan.
    exclude_duplicate_file_groups: bool = False

    # =========================================================
    # LAIN-LAIN
    # =========================================================

    product_thresholds: Tuple[int, ...] = (3, 5, 10)
    parquet_compression: str = "snappy"
    n_profile_samples: int = 5

    # =========================================================
    # PATHS
    # =========================================================

    @property
    def mapping_path(self) -> Path:
        return Path(self.raw_dir) / self.mapping_filename

    @property
    def reports_dir(self) -> Path:
        return Path(self.output_dir) / "reports"

    @property
    def reviews_path(self) -> Path:
        return Path(self.output_dir) / "reviews_clean.parquet"

    # =========================================================
    # VALIDASI
    # =========================================================

    def validate(self) -> None:

        if not Path(self.raw_dir).is_dir():
            raise FileNotFoundError(
                f"Folder data mentah tidak ditemukan: {self.raw_dir}"
            )

        if not self.mapping_path.is_file():
            raise FileNotFoundError(
                f"File pemetaan tidak ditemukan: {self.mapping_path}"
            )

        if self.min_review_length < 1:
            raise ValueError("min_review_length minimal 1.")

        if not self.timestamp_formats:
            raise ValueError("timestamp_formats tidak boleh kosong.")

        for name in (
            "role_strong_rate",
            "role_weak_rate",
            "masked_name_rate",
            "seller_reply_cooccurrence",
            "max_files_without_review_rate",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} harus di antara 0 dan 1.")

        if (
            self.aspect_tag_vocab_path is not None
            and not Path(self.aspect_tag_vocab_path).is_file()
        ):
            raise FileNotFoundError(
                f"aspect_tag_vocab_path tidak ditemukan: "
                f"{self.aspect_tag_vocab_path}"
            )
