"""Pembentukan sekuens per produk dari tabel keluaran RM 1.

Satu sekuens = ulasan satu produk, urut timestamp naik; o_t = [S+, S~, S-].
Tanpa padding atau truncation. Hanya urutan relatif yang dipakai (bukan rentang waktu absolut).

Modul ini hanya membaca kolom yang ada di daftar putih (id produk, timestamp, kolom pengurut,
probabilitas/logit, penanda include/sintetis). Kolom lain, termasuk kolom label bila ada,
tidak pernah disentuh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .config import HMMConfig


# ----------------------------------------------------------------------
# REPRESENTASI OBSERVASI
# ----------------------------------------------------------------------
def to_observation(probs: np.ndarray, cfg: HMMConfig, logits: np.ndarray | None = None) -> np.ndarray:
    """Ubah probabilitas (N, 3) ber-urutan [S+, S~, S-] menjadi matriks observasi HMM.

    prob  : probabilitas mentah (bawaan, seperti proposal)
    logit : logit mentah dari RM 1 (tanpa pemusatan)
    alr   : log(p_k / p_ref) untuk k != ref (2 dimensi)
    clr   : log p_k - rata-rata log p (3 dimensi, rangkum nol)
    """
    probs = np.asarray(probs, dtype=float)
    if cfg.representation == "prob":
        return probs.copy()
    if cfg.representation == "logit":
        if logits is None:
            raise ValueError("representation='logit' membutuhkan kolom logit dari RM 1.")
        return np.asarray(logits, dtype=float).copy()
    log_p = np.log(np.clip(probs, cfg.prob_eps, None))
    if cfg.representation == "alr":
        keep = [i for i in range(3) if i != cfg.alr_ref_index]
        return log_p[:, keep] - log_p[:, [cfg.alr_ref_index]]
    return log_p - log_p.mean(axis=1, keepdims=True)  # clr


# ----------------------------------------------------------------------
# WADAH SEKUENS
# ----------------------------------------------------------------------
@dataclass
class SequenceSet:
    product_ids: list[Any]
    lengths: np.ndarray  # (S,)
    X: np.ndarray  # (N, d) observasi, ditumpuk
    P: np.ndarray  # (N, 3) probabilitas [S+, S~, S-] (dipakai untuk mengurutkan state)
    offsets: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.lengths = np.asarray(self.lengths, dtype=int)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        if self.offsets[-1] != len(self.X) or len(self.X) != len(self.P):
            raise ValueError("lengths tidak cocok dengan jumlah baris X/P.")

    def __len__(self) -> int:
        return len(self.product_ids)

    def __iter__(self) -> Iterator[tuple[Any, np.ndarray]]:
        for i, pid in enumerate(self.product_ids):
            yield pid, self.X[self.offsets[i] : self.offsets[i + 1]]

    def subset(self, ids: Any) -> "SequenceSet":
        """Sekuens untuk id tertentu (urutan mengikuti SequenceSet ini)."""
        wanted = set(ids)
        keep = [i for i, pid in enumerate(self.product_ids) if pid in wanted]
        return self._take(keep)

    def _take(self, keep: list[int]) -> "SequenceSet":
        if not keep:
            return SequenceSet([], np.array([], dtype=int), self.X[:0], self.P[:0])
        rows = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in keep])
        return SequenceSet(
            [self.product_ids[i] for i in keep], self.lengths[keep], self.X[rows], self.P[rows]
        )


# ----------------------------------------------------------------------
# PEMBENTUKAN SEKUENS
# ----------------------------------------------------------------------
def _required_columns(cfg: HMMConfig) -> list[str]:
    cols = [cfg.product_col, cfg.timestamp_col, *cfg.prob_cols]
    if cfg.representation == "logit":
        cols += list(cfg.logit_cols)
    return cols


def build_sequences(df: pd.DataFrame, cfg: HMMConfig) -> tuple[SequenceSet, dict[str, Any], pd.Series]:
    """Bentuk sekuens dari tabel RM 1.

    Mengembalikan ``(SequenceSet, laporan, is_synthetic_per_produk)``. Laporan memuat jumlah
    baris yang dibuang, produk yang dikeluarkan karena T < min_reviews, dan sebaran panjang.
    """
    missing = [c for c in _required_columns(cfg) if c not in df.columns]
    if missing:
        raise KeyError(f"Kolom wajib tidak ada di tabel masukan: {missing}. Kolom tersedia: {list(df.columns)}")

    tie_cols = [c for c in cfg.tiebreak_cols if c in df.columns]
    use_cols = list(dict.fromkeys(_required_columns(cfg) + tie_cols))
    for optional in (cfg.include_col, cfg.synthetic_col):
        if optional in df.columns:
            use_cols.append(optional)
    data = df.loc[:, use_cols].copy()  # daftar putih kolom: label tidak pernah ikut
    report: dict[str, Any] = {"n_rows_input": int(len(data))}

    if cfg.include_col in data.columns:
        keep = data[cfg.include_col].eq(True)
        report["n_rows_not_included"] = int((~keep).sum())
        data = data.loc[keep]
    else:
        report["n_rows_not_included"] = 0

    data[cfg.timestamp_col] = pd.to_datetime(data[cfg.timestamp_col], errors="coerce")
    value_cols = list(cfg.prob_cols) + (list(cfg.logit_cols) if cfg.representation == "logit" else [])
    bad = data[value_cols].isna().any(axis=1) | data[cfg.timestamp_col].isna() | data[cfg.product_col].isna()
    report["n_rows_dropped_nan"] = int(bad.sum())
    data = data.loc[~bad]
    if data.empty:
        raise ValueError("Tidak ada baris valid setelah penyaringan.")

    probs = data[list(cfg.prob_cols)].to_numpy(dtype=float)
    if not np.all(np.isfinite(probs)) or probs.min() < -1e-9 or probs.max() > 1 + 1e-9:
        raise ValueError("Probabilitas di luar [0, 1] atau tidak berhingga.")
    deviation = float(np.abs(probs.sum(axis=1) - 1.0).max())
    report["max_prob_sum_deviation"] = deviation
    if deviation > cfg.prob_sum_tol:
        raise ValueError(f"Probabilitas tidak berjumlah 1 (deviasi maks {deviation:.3g} > {cfg.prob_sum_tol}).")
    if cfg.representation == "logit" and not np.all(np.isfinite(data[list(cfg.logit_cols)].to_numpy(dtype=float))):
        raise ValueError("Logit memuat NaN/Inf.")

    # Urutan deterministik: produk, timestamp, lalu kolom pengurut (stabil).
    order_cols = [cfg.product_col, cfg.timestamp_col, *tie_cols]
    data = data.sort_values(order_cols, kind="mergesort")
    report["order_rule"] = order_cols
    report["n_unresolved_ties"] = int(data.duplicated(order_cols).sum())
    report["n_rows_tied_timestamp"] = int(data.duplicated([cfg.product_col, cfg.timestamp_col], keep=False).sum())

    sizes = data.groupby(cfg.product_col, sort=True).size()
    report["n_products_total"] = int(len(sizes))
    too_short = sizes[sizes < cfg.min_reviews]
    report["min_reviews"] = cfg.min_reviews
    report["n_products_excluded_short"] = int(len(too_short))
    report["excluded_short_ids"] = [str(i) for i in too_short.index]
    kept_ids = sizes.index[sizes >= cfg.min_reviews]
    if len(kept_ids) == 0:
        raise ValueError(f"Tidak ada produk dengan >= {cfg.min_reviews} ulasan.")
    data = data[data[cfg.product_col].isin(kept_ids)]
    sizes = sizes.loc[kept_ids]

    p = data[list(cfg.prob_cols)].to_numpy(dtype=float)
    logits = data[list(cfg.logit_cols)].to_numpy(dtype=float) if cfg.representation == "logit" else None
    X = to_observation(p, cfg, logits)
    if not np.all(np.isfinite(X)):
        raise ValueError("Observasi memuat NaN/Inf setelah transformasi.")

    seqs = SequenceSet(list(sizes.index), sizes.to_numpy(dtype=int), X, p)

    is_syn = pd.Series(False, index=sizes.index, name=cfg.synthetic_col)
    if cfg.synthetic_col in data.columns:
        per_product = data.groupby(cfg.product_col)[cfg.synthetic_col].agg(lambda s: bool(s.astype(bool).any()))
        is_syn = per_product.reindex(sizes.index).fillna(False).astype(bool)

    lens = sizes.to_numpy()
    report["n_products_used"] = int(len(sizes))
    report["n_reviews_used"] = int(lens.sum())
    report["length_min"] = int(lens.min())
    report["length_median"] = float(np.median(lens))
    report["length_max"] = int(lens.max())
    report["length_quantiles"] = {str(q): float(np.quantile(lens, q)) for q in (0.05, 0.25, 0.75, 0.95)}
    return seqs, report, is_syn
