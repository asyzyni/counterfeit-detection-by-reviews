"""Konfigurasi RM 2 (HMM). Semua path dan nama kolom diatur di sini, bukan di kode."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPRESENTATIONS = ("prob", "logit", "alr", "clr")
BACKENDS = ("auto", "hmmlearn", "numpy")


@dataclass
class HMMConfig:
    # ---------------------------------------------------------
    # KOLOM MASUKAN (tabel keluaran RM 1; satu baris per ulasan)
    # ---------------------------------------------------------
    product_col: str = "product_id"
    timestamp_col: str = "timestamp"
    # Penentu urutan bila timestamp kembar. Dipakai berurutan; kolom yang tidak ada dilewati.
    # Harus sama dengan aturan urutan di cleaning/RM 1 (seq_in_product).
    tiebreak_cols: tuple[str, ...] = ("source_file", "row_order", "seq_in_product")
    # Urutan observasi o_t = [S+, S~, S-] seperti proposal (BUKAN urutan kolom RM 1).
    prob_cols: tuple[str, str, str] = ("p_positif", "p_netral", "p_negatif")
    logit_cols: tuple[str, str, str] = ("logit_positif", "logit_netral", "logit_negatif")
    include_col: str = "include_for_inference"  # bila ada: hanya baris True yang dipakai
    synthetic_col: str = "is_synthetic"  # opsional, level produk

    # ---------------------------------------------------------
    # REPRESENTASI OBSERVASI
    # ---------------------------------------------------------
    representation: str = "prob"  # prob (proposal) | logit | alr | clr
    alr_ref_index: int = 1  # indeks di prob_cols yang jadi pembagi ALR (default: netral)
    prob_eps: float = 1e-6  # pemotong sebelum log pada alr/clr
    prob_sum_tol: float = 1e-4  # toleransi |sum(p) - 1|

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------
    n_states: int = 2
    covariance_type: str = "diag"  # hanya "diag" yang didukung
    n_iter: int = 200
    tol: float = 1e-4  # berhenti bila kenaikan total log-likelihood < tol
    min_covar: float = 1e-3  # lantai varians (sama dengan bawaan hmmlearn)
    init_variance_prob: float = 0.04  # sigma^2 awal untuk representasi "prob" (proposal)
    n_init: int = 10  # restart ke-0 = inisialisasi proposal; sisanya acak
    seed: int = 42
    backend: str = "auto"  # auto: hmmlearn bila terpasang, kalau tidak numpy

    # ---------------------------------------------------------
    # SEKUENS
    # ---------------------------------------------------------
    min_reviews: int = 3  # proposal: 3. Untuk HMM stabil, 5-10 lebih aman (keputusan terbuka)

    # ---------------------------------------------------------
    # KELUARAN
    # ---------------------------------------------------------
    model_path: Path | None = None
    meta_path: Path | None = None
    features_path: Path | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        self.prob_cols = tuple(self.prob_cols)  # type: ignore[assignment]
        self.logit_cols = tuple(self.logit_cols)  # type: ignore[assignment]
        self.tiebreak_cols = tuple(self.tiebreak_cols)
        self.validate()

    # ---------------------------------------------------------
    def validate(self) -> None:
        if self.representation not in REPRESENTATIONS:
            raise ValueError(f"representation harus salah satu {REPRESENTATIONS}, didapat {self.representation!r}")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend harus salah satu {BACKENDS}, didapat {self.backend!r}")
        if self.covariance_type != "diag":
            raise ValueError("Hanya covariance_type='diag' yang didukung (sesuai proposal).")
        if self.n_states < 2:
            raise ValueError("n_states minimal 2.")
        if self.n_iter < 1 or self.n_init < 1:
            raise ValueError("n_iter dan n_init minimal 1.")
        if self.min_reviews < 2:
            raise ValueError("min_reviews minimal 2 (pos_transisi butuh selisih antar waktu).")
        if len(self.prob_cols) != 3 or len(self.logit_cols) != 3:
            raise ValueError("prob_cols dan logit_cols harus berisi tepat 3 kolom.")
        if not 0 <= self.alr_ref_index < 3:
            raise ValueError("alr_ref_index harus 0, 1, atau 2.")
        if self.min_covar <= 0 or self.tol < 0 or self.prob_eps <= 0:
            raise ValueError("min_covar dan prob_eps harus > 0, tol >= 0.")

    @property
    def negative_index(self) -> int:
        """Indeks kelas negatif di prob_cols (kolom terakhir, o_t = [S+, S~, S-])."""
        return 2

    @property
    def obs_dim(self) -> int:
        return 2 if self.representation == "alr" else 3

    @classmethod
    def from_experiment(cls, experiment: Any, **overrides: Any) -> "HMMConfig":
        """Ambil nilai dari ExperimentConfig (src/config.py) dan tentukan path keluaran bawaan."""
        values: dict[str, Any] = {
            "n_states": experiment.hmm_components,
            "covariance_type": experiment.hmm_covariance_type,
            "n_iter": experiment.hmm_n_iter,
            "min_reviews": experiment.min_reviews_per_product,
            "seed": experiment.random_state,
            "model_path": experiment.models_dir / "hmm_model.pkl",
            "meta_path": experiment.models_dir / "hmm_meta.json",
            "features_path": experiment.features_dir / "hmm_features.parquet",
        }
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, Path):
                out[key] = str(value)
        return out
