"""HMM emisi Gaussian multivariat berkovarians diagonal: inferensi (log-space) dan Baum-Welch.

Semua perhitungan forward/backward dilakukan di ruang log dengan numpy, jadi hasil fitur
identik apa pun pustaka yang dipakai untuk fitting (hmmlearn atau cadangan numpy di sini).

Notasi (Rabiner 1989): lambda = (pi, A, B); alpha_t(i), beta_t(i), gamma_t(i), xi_t(i,j).
Multi-sekuens: sekuens-sekuens produk TIDAK disambung. Data masuk sebagai matriks X yang
ditumpuk dan vektor ``lengths`` (sama dengan konvensi hmmlearn).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

_LOG_2PI = float(np.log(2.0 * np.pi))


def _safe_log(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(x)


@dataclass
class GaussianDiagHMM:
    startprob: np.ndarray  # (K,)
    transmat: np.ndarray  # (K, K)
    means: np.ndarray  # (K, d)
    variances: np.ndarray  # (K, d)

    def __post_init__(self) -> None:
        self.startprob = np.asarray(self.startprob, dtype=float)
        self.transmat = np.asarray(self.transmat, dtype=float)
        self.means = np.asarray(self.means, dtype=float)
        self.variances = np.asarray(self.variances, dtype=float)
        k, d = self.means.shape
        if self.startprob.shape != (k,) or self.transmat.shape != (k, k) or self.variances.shape != (k, d):
            raise ValueError("Bentuk parameter HMM tidak konsisten.")

    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        return self.means.shape[0]

    @property
    def n_features(self) -> int:
        return self.means.shape[1]

    def copy(self) -> "GaussianDiagHMM":
        return GaussianDiagHMM(self.startprob.copy(), self.transmat.copy(), self.means.copy(), self.variances.copy())

    def reordered(self, order: np.ndarray) -> "GaussianDiagHMM":
        """Susun ulang state: state baru i = state lama order[i]."""
        order = np.asarray(order, dtype=int)
        return GaussianDiagHMM(
            self.startprob[order],
            self.transmat[np.ix_(order, order)],
            self.means[order],
            self.variances[order],
        )

    def check_valid(self, atol: float = 1e-8) -> None:
        """Baris stokastik, varians positif, tanpa NaN/Inf."""
        for name in ("startprob", "transmat", "means", "variances"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} memuat NaN/Inf.")
        if not np.isclose(self.startprob.sum(), 1.0, atol=atol):
            raise ValueError(f"startprob berjumlah {self.startprob.sum()}, bukan 1.")
        if not np.allclose(self.transmat.sum(axis=1), 1.0, atol=atol):
            raise ValueError("Ada baris transmat yang tidak berjumlah 1.")
        if np.any(self.variances <= 0):
            raise ValueError("Ada varians <= 0.")

    # ------------------------------------------------------------------
    # EMISI
    # ------------------------------------------------------------------
    def log_emission(self, X: np.ndarray) -> np.ndarray:
        """log b_i(o_t) untuk Gaussian diagonal: (N, K)."""
        X = np.asarray(X, dtype=float)
        diff = X[:, None, :] - self.means[None, :, :]
        return -0.5 * (
            self.n_features * _LOG_2PI
            + np.log(self.variances).sum(axis=1)[None, :]
            + (diff**2 / self.variances[None, :, :]).sum(axis=2)
        )

    # ------------------------------------------------------------------
    # FORWARD / BACKWARD (satu sekuens; implementasi acuan yang mudah dibaca)
    # ------------------------------------------------------------------
    def forward_log(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward algorithm di ruang log. Mengembalikan log alpha (T, K) dan log P(O | lambda)."""
        log_b = self.log_emission(X)
        log_a = _safe_log(self.transmat)
        t_len = log_b.shape[0]
        log_alpha = np.empty_like(log_b)
        log_alpha[0] = _safe_log(self.startprob) + log_b[0]
        for t in range(1, t_len):
            log_alpha[t] = log_b[t] + logsumexp(log_alpha[t - 1][:, None] + log_a, axis=0)
        return log_alpha, float(logsumexp(log_alpha[-1]))

    def loglik(self, X: np.ndarray) -> float:
        """log P(O | lambda) satu sekuens (terminasi forward)."""
        return self.forward_log(X)[1]

    def posterior(self, X: np.ndarray) -> np.ndarray:
        """gamma_t(i) satu sekuens: (T, K)."""
        gamma, _, _ = self.posteriors(X, np.array([len(X)]))
        return gamma

    # ------------------------------------------------------------------
    # FORWARD-BACKWARD MULTI-SEKUENS (dipadatkan agar vektorisasi lintas sekuens)
    # ------------------------------------------------------------------
    def _padded(self, X: np.ndarray, lengths: np.ndarray):
        lengths = np.asarray(lengths, dtype=int)
        if lengths.ndim != 1 or lengths.min() < 1 or lengths.sum() != len(X):
            raise ValueError("lengths harus bilangan >= 1 dan berjumlah sama dengan jumlah baris X.")
        n_seq, t_max = len(lengths), int(lengths.max())
        seq_idx = np.repeat(np.arange(n_seq), lengths)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        t_idx = np.arange(len(X)) - np.repeat(starts, lengths)
        log_b = np.zeros((n_seq, t_max, self.n_states))
        log_b[seq_idx, t_idx] = self.log_emission(X)
        return lengths, seq_idx, t_idx, log_b

    def _forward_backward(self, log_b: np.ndarray, lengths: np.ndarray):
        n_seq, t_max, k = log_b.shape
        log_pi = _safe_log(self.startprob)
        log_a = _safe_log(self.transmat)

        log_alpha = np.zeros((n_seq, t_max, k))
        log_alpha[:, 0] = log_pi[None, :] + log_b[:, 0]
        for t in range(1, t_max):
            step = logsumexp(log_alpha[:, t - 1, :, None] + log_a[None], axis=1) + log_b[:, t]
            valid = (t < lengths)[:, None]
            log_alpha[:, t] = np.where(valid, step, log_alpha[:, t - 1])
        last = log_alpha[np.arange(n_seq), lengths - 1]
        ll = logsumexp(last, axis=1)

        log_beta = np.zeros((n_seq, t_max, k))
        for t in range(t_max - 2, -1, -1):
            step = logsumexp(log_a[None] + (log_b[:, t + 1] + log_beta[:, t + 1])[:, None, :], axis=2)
            valid = ((t + 1) < lengths)[:, None]
            log_beta[:, t] = np.where(valid, step, 0.0)
        return log_alpha, log_beta, ll

    def posteriors(self, X: np.ndarray, lengths: np.ndarray):
        """gamma (N, K) untuk data ditumpuk, log-likelihood per sekuens (S,), dan ringkasan xi.

        Mengembalikan ``(gamma, ll, xi_sum)`` dengan xi_sum = sum_t xi_t(i,j) lintas semua sekuens (K, K).
        """
        lengths, seq_idx, t_idx, log_b = self._padded(X, lengths)
        log_alpha, log_beta, ll = self._forward_backward(log_b, lengths)
        log_gamma = log_alpha + log_beta - ll[:, None, None]
        gamma_pad = np.exp(log_gamma)
        gamma = gamma_pad[seq_idx, t_idx]

        log_a = _safe_log(self.transmat)
        xi_sum = np.zeros((self.n_states, self.n_states))
        for t in range(log_b.shape[1] - 1):
            valid = (t + 1) < lengths
            if not valid.any():
                break
            log_xi = (
                log_alpha[valid, t, :, None]
                + log_a[None]
                + (log_b[valid, t + 1] + log_beta[valid, t + 1])[:, None, :]
                - ll[valid, None, None]
            )
            xi_sum += np.exp(log_xi).sum(axis=0)
        return gamma, ll, xi_sum

    def score_each(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """log P(O_s | lambda) per sekuens."""
        lengths, _, _, log_b = self._padded(X, lengths)
        log_alpha = np.zeros_like(log_b)
        log_alpha[:, 0] = _safe_log(self.startprob)[None, :] + log_b[:, 0]
        log_a = _safe_log(self.transmat)
        for t in range(1, log_b.shape[1]):
            step = logsumexp(log_alpha[:, t - 1, :, None] + log_a[None], axis=1) + log_b[:, t]
            log_alpha[:, t] = np.where((t < lengths)[:, None], step, log_alpha[:, t - 1])
        return logsumexp(log_alpha[np.arange(len(lengths)), lengths - 1], axis=1)


# ----------------------------------------------------------------------
# BAUM-WELCH (cadangan numpy)
# ----------------------------------------------------------------------
@dataclass
class EMResult:
    model: GaussianDiagHMM
    loglik_history: list[float]
    n_iter: int
    converged: bool
    n_decreases: int  # berapa kali log-likelihood turun lebih dari 1e-6 (mestinya 0)


def baum_welch(
    init: GaussianDiagHMM,
    X: np.ndarray,
    lengths: np.ndarray,
    n_iter: int = 200,
    tol: float = 1e-4,
    min_covar: float = 1e-3,
) -> EMResult:
    """EM untuk HMM Gaussian diagonal, multi-sekuens. Memakai semua sekuens secara bersamaan."""
    X = np.asarray(X, dtype=float)
    lengths = np.asarray(lengths, dtype=int)
    model = init.copy()
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    history: list[float] = []
    converged = False
    n_decreases = 0

    for it in range(1, n_iter + 1):
        gamma, ll, xi_sum = model.posteriors(X, lengths)
        total = float(ll.sum())
        if history:
            if total < history[-1] - 1e-6:
                n_decreases += 1
            if abs(total - history[-1]) < tol:
                history.append(total)
                converged = True
                break
        history.append(total)

        # --- M-step ---
        new = model.copy()
        start = gamma[starts].sum(axis=0)
        new.startprob = start / start.sum()
        row = xi_sum.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            trans = np.where(row > 0, xi_sum / row, 1.0 / model.n_states)
        new.transmat = trans
        post = gamma.sum(axis=0)  # (K,)
        for i in range(model.n_states):
            if post[i] < 1e-12:  # state mati: pertahankan parameter lama
                continue
            mu = gamma[:, i] @ X / post[i]
            var = gamma[:, i] @ (X**2) / post[i] - mu**2
            new.means[i] = mu
            new.variances[i] = np.maximum(var, min_covar)
        model = new

    # log-likelihood akhir dari parameter terakhir
    final = float(model.score_each(X, lengths).sum())
    if not history or final != history[-1]:
        history.append(final)
    return EMResult(model, history, len(history), converged, n_decreases)
