"""Orkestrasi RM 2: sekuens -> fit (hanya Train) -> beku -> fitur Train/Test/sintetis -> simpan.

Kontrak:
  masukan : tabel keluaran RM 1 (satu baris per ulasan) + id produk Train dan Test
  keluaran: tabel fitur satu baris per produk, hmm_model.pkl, hmm_meta.json
Modul ini tidak menerima label counterfeit/genuine.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from .config import HMMConfig
from .features import EXTRA_COLUMNS, FEATURE_COLUMNS, extract_features
from .fitting import FitResult, fit_hmm
from .gaussian_hmm import GaussianDiagHMM
from .sequences import SequenceSet, build_sequences

SCHEMA_VERSION = 1


@dataclass
class RM2Result:
    features: pd.DataFrame
    fit: FitResult
    meta: dict[str, Any]
    seqs: SequenceSet
    paths: dict[str, Path] = field(default_factory=dict)


# ----------------------------------------------------------------------
def ids_sha256(ids: Iterable[Any]) -> str:
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode("utf-8")).hexdigest()


def assert_no_leakage(fit_ids: set, train_ids: set, test_ids: set, synthetic_ids: set) -> None:
    """Pagar kebocoran. Dipanggil sebelum fit."""
    if not fit_ids <= train_ids:
        raise AssertionError(f"Id untuk fit bukan bagian dari Train: {sorted(map(str, fit_ids - train_ids))[:5]}")
    if fit_ids & test_ids:
        raise AssertionError(f"Id untuk fit beririsan dengan Test: {sorted(map(str, fit_ids & test_ids))[:5]}")
    if train_ids & test_ids:
        raise AssertionError(f"Train dan Test beririsan: {sorted(map(str, train_ids & test_ids))[:5]}")
    if synthetic_ids & fit_ids:
        raise AssertionError("Produk sintetis tidak boleh ikut fit.")
    if synthetic_ids & test_ids:
        raise AssertionError("Produk sintetis tidak boleh masuk Test.")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def _check_free(paths: list[Path | None], overwrite: bool) -> None:
    existing = [str(p) for p in paths if p is not None and p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "File keluaran sudah ada dan tidak ditimpa (set overwrite=True / --overwrite):\n  " + "\n  ".join(existing)
        )


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    for name in ("scipy", "hmmlearn", "sklearn", "joblib"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "?")
        except ImportError:
            out[name] = "tidak terpasang"
    return out


def _describe(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, float]]:
    if df.empty:
        return {}
    d = df[cols].describe().T
    return {c: {k: float(v) for k, v in d.loc[c].items()} for c in cols}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ----------------------------------------------------------------------
def run_rm2(
    df: pd.DataFrame,
    cfg: HMMConfig,
    train_ids: Iterable[Any],
    test_ids: Iterable[Any],
    split_is_temporary: bool = False,
    stability_seeds: Iterable[int] | None = None,
    save: bool = True,
) -> RM2Result:
    """Jalankan RM 2 end-to-end pada tabel RM 1 ``df``."""
    train_set, test_set = set(train_ids), set(test_ids)
    if save:
        _check_free([cfg.model_path, cfg.meta_path, cfg.features_path], cfg.overwrite)

    seqs, seq_report, is_syn = build_sequences(df, cfg)
    syn_set = set(is_syn[is_syn].index)
    available = set(seqs.product_ids)

    fit_ids = (train_set & available) - syn_set
    assert_no_leakage(fit_ids, train_set, test_set, syn_set)
    test_used = [p for p in seqs.product_ids if p in test_set]
    train_used = [p for p in seqs.product_ids if p in fit_ids]
    syn_used = [p for p in seqs.product_ids if p in syn_set]
    unassigned = available - set(train_used) - set(test_used) - set(syn_used)
    if len(train_used) < 2:
        raise ValueError(f"Produk Train yang bisa dipakai hanya {len(train_used)}; periksa daftar split dan min_reviews.")
    if not test_used:
        raise ValueError("Tidak ada produk Test dengan sekuens valid.")

    train_seqs, test_seqs, syn_seqs = seqs.subset(train_used), seqs.subset(test_used), seqs.subset(syn_used)

    fit = fit_hmm(train_seqs, cfg)
    frozen = fit.model.copy()  # bukti beku: parameter tidak boleh berubah oleh transform

    parts = []
    for name, subset in (("train", train_seqs), ("test", test_seqs), ("synthetic", syn_seqs)):
        feats = extract_features(fit.model, subset, cfg)
        feats.insert(1, "split", name)
        feats.insert(2, "is_synthetic", name == "synthetic")
        parts.append(feats)
    features = pd.concat([p for p in parts if len(p)], ignore_index=True)
    for attr in ("startprob", "transmat", "means", "variances"):
        if not np.array_equal(getattr(frozen, attr), getattr(fit.model, attr)):
            raise AssertionError(f"Parameter {attr} berubah saat transform: model tidak beku.")
    if not np.all(np.isfinite(features[list(FEATURE_COLUMNS)].to_numpy())):
        raise ValueError("Fitur memuat NaN/Inf.")

    cols = list(FEATURE_COLUMNS)
    tr, te = features[features.split == "train"], features[features.split == "test"]
    corr = tr[cols + list(EXTRA_COLUMNS)].corr().round(4)

    stability = None
    seeds = list(stability_seeds or [])
    if seeds:
        stability = _seed_stability(train_seqs, cfg, fit, tr, seeds)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "versions": _versions(),
        "backend": fit.backend,
        "config": cfg.to_dict(),
        "split_is_temporary": bool(split_is_temporary),
        "split_note": "TEMPORARY, bukan untuk dilaporkan" if split_is_temporary else "daftar id dari luar modul",
        "train_ids_sha256": ids_sha256(train_used),
        "n_products": {"train": len(train_used), "test": len(test_used), "synthetic": len(syn_used),
                       "unassigned_not_transformed": len(unassigned)},
        "sequences": seq_report,
        "state_order_applied": fit.state_order,
        "state_names": fit.state_names,
        "state_neg_mass_train": fit.state_neg_mass,
        "state_naming_note": "State terakhir = rata-rata p_negatif tertinggi. Itu belum tentu counterfeit; "
                             "produk asli yang cacat juga bisa negatif (keterbatasan untuk naskah).",
        "best_restart": fit.best_restart,
        "train_loglik": fit.train_loglik,
        "train_loglik_per_review": fit.train_loglik_per_review,
        "converged": fit.converged,
        "restarts": fit.restarts,
        "warnings": fit.warnings,
        "params": {"startprob": fit.model.startprob, "transmat": fit.model.transmat,
                   "means": fit.model.means, "variances": fit.model.variances},
        "feature_summary": {"train": _describe(tr, cols), "test": _describe(te, cols)},
        "feature_correlation_train": corr.to_dict(),
        "seed_stability": stability,
        "runtime": {"argv": sys.argv[:1]},
    }

    result = RM2Result(features=features, fit=fit, meta=_json_safe(meta), seqs=seqs)
    if save:
        result.paths = save_outputs(result, cfg)
    return result


def _seed_stability(train_seqs: SequenceSet, cfg: HMMConfig, main_fit: FitResult, main_train: pd.DataFrame, seeds: list[int]) -> dict:
    """Fit ulang dengan seed lain; bandingkan urutan state dan korelasi fitur Train terhadap fit utama."""
    from dataclasses import replace

    rows = []
    for s in seeds:
        alt_cfg = replace(cfg, seed=int(s))
        alt = fit_hmm(train_seqs, alt_cfg)
        feats = extract_features(alt.model, train_seqs, alt_cfg).set_index(cfg.product_col)
        base = main_train.set_index(cfg.product_col).loc[feats.index]
        row = {"seed": int(s), "train_loglik": alt.train_loglik, "state_neg_mass": alt.state_neg_mass,
               "state_order_consistent": bool(alt.state_neg_mass[-1] > alt.state_neg_mass[0])}
        for c in FEATURE_COLUMNS:
            row[f"corr_{c}"] = float(np.corrcoef(feats[c], base[c])[0, 1])
        rows.append(row)
    return {"seeds": rows}


def save_outputs(result: RM2Result, cfg: HMMConfig) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if cfg.features_path is not None:
        _write_table(result.features, Path(cfg.features_path))
        paths["features"] = Path(cfg.features_path)
    if cfg.model_path is not None:
        Path(cfg.model_path).parent.mkdir(parents=True, exist_ok=True)
        m = result.fit.model
        joblib.dump({
            "schema_version": SCHEMA_VERSION,
            "startprob": m.startprob, "transmat": m.transmat, "means": m.means, "variances": m.variances,
            "state_names": result.fit.state_names,
            "representation": cfg.representation,
            "config": cfg.to_dict(),
            "train_ids_sha256": result.meta["train_ids_sha256"],
        }, cfg.model_path)
        paths["model"] = Path(cfg.model_path)
    if cfg.meta_path is not None:
        Path(cfg.meta_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.meta_path, "w", encoding="utf-8") as stream:
            json.dump(result.meta, stream, ensure_ascii=False, indent=2)
        paths["meta"] = Path(cfg.meta_path)
    return paths


def load_hmm_model(path: str | Path) -> tuple[GaussianDiagHMM, dict[str, Any]]:
    """Muat model beku dari hmm_model.pkl. File ini dibuat sendiri oleh run_rm2; jangan muat pickle dari sumber tak dikenal."""
    blob = joblib.load(path)
    model = GaussianDiagHMM(blob["startprob"], blob["transmat"], blob["means"], blob["variances"])
    model.check_valid()
    return model, blob


def transform_new_data(df: pd.DataFrame, model_path: str | Path, cfg: HMMConfig | None = None) -> pd.DataFrame:
    """Terapkan model beku ke data baru (tanpa fit ulang). Konfigurasi diambil dari file model bila cfg=None."""
    model, blob = load_hmm_model(model_path)
    if cfg is None:
        raw = dict(blob["config"])
        raw.pop("model_path", None), raw.pop("meta_path", None), raw.pop("features_path", None)
        cfg = HMMConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()})
    seqs, _, _ = build_sequences(df, cfg)
    return extract_features(model, seqs, cfg)
