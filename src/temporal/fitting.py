"""Pelatihan HMM (Baum-Welch) dengan inisialisasi proposal + restart acak, dan pengurutan state.

HMM tanpa label (unsupervised). Fungsi di sini hanya menerima observasi, tidak ada label.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import HMMConfig
from .gaussian_hmm import GaussianDiagHMM, baum_welch
from .sequences import SequenceSet, to_observation

LOGGER = logging.getLogger(__name__)

# Inisialisasi dari proposal (hlm. 37-38), urutan o_t = [S+, S~, S-].
PROPOSAL_PI = np.array([0.6, 0.4])
PROPOSAL_A = np.array([[0.8, 0.2], [0.3, 0.7]])
PROPOSAL_MU0 = np.array([0.6, 0.3, 0.1])  # state 0: dominan positif
PROPOSAL_MU1 = np.array([0.1, 0.3, 0.6])  # state 1: dominan negatif


def hmmlearn_available() -> bool:
    return importlib.util.find_spec("hmmlearn") is not None


def resolve_backend(cfg: HMMConfig) -> str:
    if cfg.backend == "numpy":
        return "numpy"
    if cfg.backend == "hmmlearn":
        if not hmmlearn_available():
            raise ImportError("backend='hmmlearn' diminta, tetapi hmmlearn belum terpasang (pip install hmmlearn).")
        return "hmmlearn"
    return "hmmlearn" if hmmlearn_available() else "numpy"


def state_name(index: int, n_states: int) -> str:
    """Nama netral. State yang negatif belum tentu counterfeit (produk asli yang cacat pun bisa negatif)."""
    if index == n_states - 1:
        return "state_negatif_tinggi"
    if index == 0:
        return "state_negatif_rendah"
    return f"state_antara_{index}"


# ----------------------------------------------------------------------
# INISIALISASI
# ----------------------------------------------------------------------
def _proposal_means(n_states: int) -> np.ndarray:
    weights = np.linspace(0.0, 1.0, n_states)
    return np.array([(1 - w) * PROPOSAL_MU0 + w * PROPOSAL_MU1 for w in weights])


def _init_variances(cfg: HMMConfig, X: np.ndarray, n_states: int) -> np.ndarray:
    if cfg.representation == "prob":
        return np.full((n_states, X.shape[1]), cfg.init_variance_prob)
    return np.tile(np.maximum(X.var(axis=0), cfg.min_covar), (n_states, 1))


def make_init(cfg: HMMConfig, seqs: SequenceSet, restart: int, rng: np.random.Generator) -> GaussianDiagHMM:
    """restart 0 = inisialisasi proposal; restart >= 1 = acak (bentuk tetap, nilai dari rng)."""
    k = cfg.n_states
    if restart == 0:
        if k == 2:
            pi, a = PROPOSAL_PI.copy(), PROPOSAL_A.copy()
        else:
            pi = np.full(k, 1.0 / k)
            a = np.full((k, k), 0.2 / (k - 1))
            np.fill_diagonal(a, 0.8)
        mean_prob = _proposal_means(k)
        if cfg.representation == "logit":
            # logit = log p + konstanta per baris; konstanta diperkirakan dari data latih.
            offset = float(np.mean(seqs.X[:, 0] - np.log(np.clip(seqs.P[:, 0], cfg.prob_eps, None))))
            means = np.log(np.clip(mean_prob, cfg.prob_eps, None)) + offset
        else:
            means = to_observation(mean_prob, cfg)
    else:
        pi = rng.dirichlet(np.ones(k))
        a = rng.dirichlet(np.ones(k), size=k) + 2.0 * np.eye(k)
        a = a / a.sum(axis=1, keepdims=True)
        picks = rng.choice(len(seqs.X), size=k, replace=False)
        means = seqs.X[picks].copy() + rng.normal(0.0, 1e-3, size=(k, seqs.X.shape[1]))
    return GaussianDiagHMM(pi, a, means, _init_variances(cfg, seqs.X, k))


# ----------------------------------------------------------------------
# SATU FIT
# ----------------------------------------------------------------------
def _fit_numpy(init: GaussianDiagHMM, seqs: SequenceSet, cfg: HMMConfig) -> dict[str, Any]:
    res = baum_welch(init, seqs.X, seqs.lengths, n_iter=cfg.n_iter, tol=cfg.tol, min_covar=cfg.min_covar)
    return {"model": res.model, "n_iter": res.n_iter, "converged": res.converged, "n_decreases": res.n_decreases}


def _fit_hmmlearn(init: GaussianDiagHMM, seqs: SequenceSet, cfg: HMMConfig, seed: int) -> dict[str, Any]:
    from hmmlearn.hmm import GaussianHMM  # impor lambat: modul tetap bisa dipakai tanpa hmmlearn

    m = GaussianHMM(
        n_components=cfg.n_states,
        covariance_type="diag",
        n_iter=cfg.n_iter,
        tol=cfg.tol,
        min_covar=cfg.min_covar,
        init_params="",  # parameter awal dari kita, bukan dari hmmlearn
        params="stmc",
        random_state=seed,
    )
    m.startprob_ = init.startprob.copy()
    m.transmat_ = init.transmat.copy()
    m.means_ = init.means.copy()
    m.covars_ = init.variances.copy()
    m.fit(seqs.X, lengths=seqs.lengths.tolist())  # multi-sekuens: jangan disambung
    k, d = cfg.n_states, seqs.X.shape[1]
    covars = np.asarray(m.covars_)
    if covars.shape != (k, d):
        raise RuntimeError(f"Bentuk covars_ hmmlearn tak terduga: {covars.shape}, diharapkan {(k, d)}.")
    model = GaussianDiagHMM(m.startprob_, m.transmat_, m.means_, covars)
    history = list(getattr(m.monitor_, "history", []))
    return {
        "model": model,
        "n_iter": len(history),
        "converged": bool(getattr(m.monitor_, "converged", False)),
        "n_decreases": int(sum(1 for a, b in zip(history[:-1], history[1:]) if b < a - 1e-6)),
    }


# ----------------------------------------------------------------------
# PENGURUTAN STATE
# ----------------------------------------------------------------------
def state_negative_mass(model: GaussianDiagHMM, seqs: SequenceSet, neg_index: int) -> np.ndarray:
    """Rata-rata p_negatif berbobot posterior untuk tiap state, dihitung di data latih."""
    gamma, _, _ = model.posteriors(seqs.X, seqs.lengths)
    weight = gamma.sum(axis=0)
    return (gamma * seqs.P[:, [neg_index]]).sum(axis=0) / np.maximum(weight, 1e-12)


def order_states(model: GaussianDiagHMM, seqs: SequenceSet, neg_index: int) -> tuple[GaussianDiagHMM, np.ndarray, np.ndarray]:
    """Urutkan state menurut rata-rata p_negatif naik: state terakhir = paling negatif.

    Mengembalikan (model terurut, permutasi order, rata-rata p_negatif tiap state terurut).
    """
    mass = state_negative_mass(model, seqs, neg_index)
    order = np.argsort(mass, kind="mergesort")
    return model.reordered(order), order, mass[order]


# ----------------------------------------------------------------------
# FIT DENGAN RESTART
# ----------------------------------------------------------------------
@dataclass
class FitResult:
    model: GaussianDiagHMM  # terurut, beku
    backend: str
    state_order: list[int]  # state_baru[i] = state_lama[state_order[i]] pada model terbaik
    state_neg_mass: list[float]
    state_names: list[str]
    restarts: list[dict[str, Any]] = field(default_factory=list)
    best_restart: int = 0
    train_loglik: float = 0.0
    train_loglik_per_review: float = 0.0
    converged: bool = False
    warnings: list[str] = field(default_factory=list)


def fit_hmm(train: SequenceSet, cfg: HMMConfig) -> FitResult:
    """Fit pada sekuens Train. Pilih restart dengan log-likelihood latih terbaik, lalu urutkan state."""
    if len(train) < 2:
        raise ValueError("Butuh minimal 2 produk latih untuk fit HMM.")
    backend = resolve_backend(cfg)
    root = np.random.SeedSequence(cfg.seed)
    child_seeds = root.spawn(cfg.n_init)

    fitted: list[tuple[GaussianDiagHMM, dict[str, Any]]] = []
    restarts: list[dict[str, Any]] = []
    for r in range(cfg.n_init):
        rng = np.random.default_rng(child_seeds[r])
        seed_int = int(child_seeds[r].generate_state(1)[0])
        init = make_init(cfg, train, r, rng)
        try:
            out = _fit_hmmlearn(init, train, cfg, seed_int) if backend == "hmmlearn" else _fit_numpy(init, train, cfg)
            model = out["model"]
            model.check_valid()
            total = float(model.score_each(train.X, train.lengths).sum())
            if not np.isfinite(total):
                raise FloatingPointError("log-likelihood tidak berhingga")
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            LOGGER.warning("restart %d gagal: %s", r, exc)
            restarts.append({"restart": r, "seed": seed_int, "init": "proposal" if r == 0 else "acak",
                             "ok": False, "error": str(exc)})
            fitted.append((None, {}))  # type: ignore[arg-type]
            continue
        ordered, order, mass = order_states(model, train, cfg.negative_index)
        fitted.append((ordered, out))
        restarts.append({
            "restart": r, "seed": seed_int, "init": "proposal" if r == 0 else "acak", "ok": True,
            "train_loglik": total, "n_iter": out["n_iter"], "converged": out["converged"],
            "n_decreases": out["n_decreases"], "state_neg_mass": [float(v) for v in mass],
            "state_order": [int(v) for v in order],
        })

    ok = [i for i, (m, _) in enumerate(fitted) if m is not None]
    if not ok:
        raise RuntimeError("Semua restart gagal; periksa data dan konfigurasi.")
    best = max(ok, key=lambda i: restarts[i]["train_loglik"])
    best_model, best_out = fitted[best]

    mass = state_negative_mass(best_model, train, cfg.negative_index)  # model sudah terurut

    warns: list[str] = []
    if cfg.n_states == 2 and abs(mass[1] - mass[0]) < 0.05:
        warns.append("Kedua state hampir tidak berbeda dalam rata-rata p_negatif (<0.05): state belum terpisah jelas.")
    if not best_out["converged"]:
        warns.append(f"Restart terbaik belum konvergen dalam {cfg.n_iter} iterasi (tol={cfg.tol}).")
    lls = [restarts[i]["train_loglik"] for i in ok]
    spread = max(lls) - min(lls)
    if spread > 1e-3 * abs(max(lls)):
        warns.append(f"Log-likelihood latih berbeda antar restart (rentang {spread:.4g}): ada optimum lokal.")

    k = cfg.n_states
    return FitResult(
        model=best_model,
        backend=backend,
        state_order=restarts[best]["state_order"],
        state_neg_mass=[float(v) for v in mass],
        state_names=[state_name(i, k) for i in range(k)],
        restarts=restarts,
        best_restart=best,
        train_loglik=float(restarts[best]["train_loglik"]),
        train_loglik_per_review=float(restarts[best]["train_loglik"]) / float(train.lengths.sum()),
        converged=bool(best_out["converged"]),
        warnings=warns,
    )
