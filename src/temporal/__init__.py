"""RM 2: Hidden Markov Model untuk pola temporal sekuens sentimen per produk.

Alur singkat (lihat docs/hmm_rm2.md):

    tabel keluaran RM 1 -> build_sequences -> fit (hanya Train) -> beku -> fitur per produk

Modul ini tidak menerima atau membaca label counterfeit/genuine.
"""

from .config import HMMConfig
from .features import extract_features
from .fitting import fit_hmm
from .gaussian_hmm import GaussianDiagHMM
from .pipeline import run_rm2
from .sequences import SequenceSet, build_sequences
from .splits import load_split_ids, temporary_product_split

__all__ = [
    "GaussianDiagHMM",
    "HMMConfig",
    "SequenceSet",
    "build_sequences",
    "extract_features",
    "fit_hmm",
    "load_split_ids",
    "run_rm2",
    "temporary_product_split",
]
