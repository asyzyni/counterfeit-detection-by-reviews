"""Fitur temporal per produk dari HMM yang sudah dibeku.

Tiga fitur proposal (hlm. 41-42) + kolom pendukung:

log_lik       log P(O | lambda) / T                  (terminasi forward, dinormalisasi panjang)
prop_state1   rata-rata gamma_t(1) sepanjang sekuens (proporsi waktu di state paling negatif)
pos_transisi  max_t |gamma_{t+1}(1) - gamma_t(1)|     (nilai selisih maksimum, sesuai Tabel 3.6)
idx_transisi  (opsional) indeks 0-based titik waktu SESUDAH lonjakan terbesar
occ_state1    (opsional) proporsi t dengan argmax posterior = state 1

"state 1" = state terakhir setelah pengurutan (rata-rata p_negatif tertinggi). Nama kolom
mengikuti proposal; maknanya adalah "state negatif tinggi", bukan "counterfeit".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HMMConfig
from .gaussian_hmm import GaussianDiagHMM
from .sequences import SequenceSet

FEATURE_COLUMNS = ("log_lik", "prop_state1", "pos_transisi")
EXTRA_COLUMNS = ("idx_transisi", "occ_state1")


def extract_features(model: GaussianDiagHMM, seqs: SequenceSet, cfg: HMMConfig) -> pd.DataFrame:
    """Satu baris per produk. Model tidak diubah (hanya inferensi)."""
    if len(seqs) == 0:
        return pd.DataFrame(columns=[cfg.product_col, "n_reviews", *FEATURE_COLUMNS, *EXTRA_COLUMNS])
    gamma, ll, _ = model.posteriors(seqs.X, seqs.lengths)
    if not np.allclose(gamma.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Posterior gamma_t tidak berjumlah 1 pada sebagian titik waktu.")
    if not (np.all(np.isfinite(gamma)) and np.all(np.isfinite(ll))):
        raise ValueError("Posterior atau log-likelihood memuat NaN/Inf.")

    gamma = np.clip(gamma, 0.0, 1.0)  # buang galat pembulatan ~1e-14
    last = model.n_states - 1
    rows = []
    for i, pid in enumerate(seqs.product_ids):
        g = gamma[seqs.offsets[i] : seqs.offsets[i + 1]]
        g1 = g[:, last]
        t_len = len(g1)
        delta = np.abs(np.diff(g1))
        rows.append({
            cfg.product_col: pid,
            "n_reviews": t_len,
            "log_lik": float(ll[i]) / t_len,
            "prop_state1": float(g1.mean()),
            "pos_transisi": float(delta.max()),
            "idx_transisi": int(delta.argmax()) + 1,
            "occ_state1": float((g.argmax(axis=1) == last).mean()),
        })
    return pd.DataFrame(rows)
