"""Inferensi sentimen RM 1: IndoBERT (SmSA, 3 kelas) -> p_negatif, p_netral, p_positif.

Model dimuat dari repo privat Hugging Face dengan revision (hash commit) yang dipin,
disimpan di cache lokal, lalu dimuat ulang dengan ``local_files_only=True``.

Pemakaian singkat:

    from src.sentiment_inference import SentimentModelConfig, load_sentiment_model, predict_sentiment_probs

    bundle = load_sentiment_model(SentimentModelConfig(revision="<hash 40 karakter>"))
    probs = predict_sentiment_probs(["barangnya bagus", "kecewa, rusak"], bundle=bundle)

Catatan keamanan dan desain:
- Token Hugging Face hanya dibaca dari variabel lingkungan (default ``HF_TOKEN``) dan
  tidak pernah dicetak atau dicatat ke log.
- ``training_args.bin`` tidak diunduh dan tidak dimuat (tidak diperlukan untuk inferensi,
  dan berformat pickle).
- Bobot dimuat dengan ``use_safetensors=True`` dalam fp32 dan mode eval.
- Urutan kolom probabilitas diturunkan dari ``config.id2label``, bukan diasumsikan.
- Modul ini sengaja tidak mengimpor torch/transformers saat di-import; keduanya baru
  dimuat di ``load_sentiment_model`` agar logika lainnya bisa diuji tanpa torch.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# ============================================================
# KONSTANTA
# ============================================================

REPO_ID = "asyzyni/indobert-nlu-smsa-counterfeit-ta"
REVISION_ENV = "INDOBERT_SENTIMENT_REVISION"
TOKEN_ENV = "HF_TOKEN"
MAX_LENGTH = 128

PROB_COLUMNS = ("p_negatif", "p_netral", "p_positif")
LOGIT_COLUMNS = ("logit_negatif", "logit_netral", "logit_positif")
LABEL_NAMES = ("negatif", "netral", "positif")

_CANONICAL = ("negative", "neutral", "positive")
EXPECTED_INDICES = {"negative": 0, "neutral": 1, "positive": 2}

_ALIASES = {
    "negative": "negative",
    "negatif": "negative",
    "neutral": "neutral",
    "netral": "neutral",
    "positive": "positive",
    "positif": "positive",
}

# Hanya file yang dibutuhkan untuk inferensi. training_args.bin sengaja TIDAK ada di sini.
DOWNLOAD_PATTERNS = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "token_type_ids")


class ModelCheckError(RuntimeError):
    """Model/tokenizer tidak memenuhi syarat yang diharapkan untuk inferensi."""


# ============================================================
# PEMERIKSAAN (tanpa torch)
# ============================================================

def validate_revision(revision: str | None) -> str:
    """Wajibkan hash commit penuh (40 heksadesimal huruf kecil)."""

    if not revision:
        raise ValueError(
            "revision model belum diisi. Isi dengan hash commit lengkap (40 karakter), "
            f"lewat SentimentModelConfig.revision, --revision, atau variabel {REVISION_ENV}."
        )
    revision = str(revision).strip()
    if not _FULL_SHA.fullmatch(revision):
        raise ValueError(
            "revision harus hash commit lengkap (40 karakter heksadesimal huruf kecil); "
            f"didapat {revision!r} ({len(revision)} karakter). Hash singkat tidak dipin."
        )
    return revision


def _canonical_label(label: object) -> str:
    key = re.sub(r"[^a-z]", "", str(label).lower())
    if key not in _ALIASES:
        raise ModelCheckError(
            f"Label model {label!r} tidak dikenali sebagai negatif/netral/positif"
        )
    return _ALIASES[key]


def check_label_mapping(
    id2label: dict,
    require_expected_order: bool = True,
) -> dict[str, int]:
    """Kembalikan peta sentimen -> indeks logit berdasarkan ``config.id2label``.

    Dengan ``require_expected_order`` model harus berurutan 0=negatif, 1=netral, 2=positif.
    """

    parsed = {int(index): _canonical_label(label) for index, label in id2label.items()}
    if sorted(parsed) != [0, 1, 2]:
        raise ModelCheckError(f"Indeks kelas harus tepat 0, 1, 2; didapat {sorted(parsed)}")
    if sorted(parsed.values()) != sorted(_CANONICAL):
        raise ModelCheckError(
            "id2label harus memetakan tepat ke negatif, netral, positif; "
            f"didapat {parsed}"
        )
    indices = {label: index for index, label in parsed.items()}
    if require_expected_order and indices != EXPECTED_INDICES:
        raise ModelCheckError(
            "id2label tidak sesuai yang diharapkan {0: negatif, 1: netral, 2: positif}; "
            f"didapat {dict(sorted(parsed.items()))}"
        )
    return indices


def check_tokenizer(tokenizer: Any, max_length: int = MAX_LENGTH, strict: bool = True) -> bool:
    """Periksa ``tokenizer.model_max_length == max_length``.

    Inferensi tetap memakai ``max_length`` eksplisit, tetapi nilai default tokenizer yang
    sangat besar menandakan tokenizer tersimpan tidak sama dengan saat training.
    """

    actual = getattr(tokenizer, "model_max_length", None)
    if actual == max_length:
        return True
    message = (
        f"tokenizer.model_max_length = {actual}, diharapkan {max_length}. "
        "Inferensi tetap memakai max_length eksplisit."
    )
    if strict:
        raise ModelCheckError(message)
    LOGGER.warning(message)
    return False


def pick_device(requested: str, cuda_available: bool, mps_available: bool) -> str:
    """Pilih device tanpa fallback diam-diam ke CPU."""

    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("device='cuda' diminta, tetapi CUDA tidak tersedia")
        return "cuda"
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("device='mps' diminta, tetapi MPS tidak tersedia")
        return "mps"
    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        raise RuntimeError(
            "Tidak ada GPU (CUDA/MPS) yang terdeteksi. Set device='cpu' secara eksplisit "
            "bila memang ingin memakai CPU."
        )
    raise ValueError(f"device tidak dikenal: {requested!r} (pilih auto, mps, cuda, atau cpu)")


# ============================================================
# KONFIGURASI & BUNDLE
# ============================================================

@dataclass
class SentimentModelConfig:
    repo_id: str = REPO_ID
    revision: str | None = field(default_factory=lambda: os.environ.get(REVISION_ENV))
    cache_dir: str | None = None
    model_dir: str | None = None        # bila diisi: muat dari folder lokal (tanpa jaringan)
    device: str = "auto"                # auto | mps | cuda | cpu
    max_length: int = MAX_LENGTH
    batch_size: int = 16
    token_env: str = TOKEN_ENV
    strict_tokenizer_check: bool = True
    allow_download: bool = True


@dataclass
class SentimentBundle:
    tokenizer: Any
    forward: Callable[[dict], np.ndarray]    # batch numpy -> logits numpy (n, 3)
    sentiment_indices: dict[str, int]        # negative/neutral/positive -> indeks logit
    device: str
    max_length: int = MAX_LENGTH
    batch_size: int = 16
    meta: dict = field(default_factory=dict)


def _has_required_files(path: Path) -> list[str]:
    missing = []
    if not (path / "config.json").is_file():
        missing.append("config.json")
    if not (path / "model.safetensors").is_file():
        missing.append("model.safetensors")
    if not ((path / "tokenizer.json").is_file() or (path / "vocab.txt").is_file()):
        missing.append("tokenizer.json atau vocab.txt")
    return missing


def resolve_model_path(config: SentimentModelConfig) -> tuple[str, str | None]:
    """Kembalikan (folder model lokal, revision). Cache lokal dicoba dulu, lalu unduh."""

    if config.model_dir:
        path = Path(config.model_dir).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"Folder model tidak ditemukan: {path}")
        missing = _has_required_files(path)
        if missing:
            raise FileNotFoundError(f"Folder model kurang: {', '.join(missing)}")
        return str(path), None

    revision = validate_revision(config.revision)

    from huggingface_hub import snapshot_download

    token = os.environ.get(config.token_env) or None
    kwargs = dict(
        repo_id=config.repo_id,
        revision=revision,
        cache_dir=config.cache_dir,
        allow_patterns=list(DOWNLOAD_PATTERNS),
        token=token,
    )

    try:
        path = snapshot_download(local_files_only=True, **kwargs)
        missing = _has_required_files(Path(path))
        if missing:
            raise FileNotFoundError(f"cache lokal kurang: {', '.join(missing)}")
        LOGGER.info("Model dimuat dari cache lokal (revision %s)", revision)
        return str(path), revision
    except Exception as exc:  # cache belum ada/tidak lengkap
        if not config.allow_download:
            raise
        LOGGER.info("Cache lokal belum lengkap (%s); mengunduh revision %s", type(exc).__name__, revision)

    path = snapshot_download(**kwargs)
    missing = _has_required_files(Path(path))
    if missing:
        raise FileNotFoundError(f"Hasil unduhan kurang: {', '.join(missing)}")
    return str(path), revision


def load_sentiment_model(config: SentimentModelConfig | None = None) -> SentimentBundle:
    """Muat tokenizer + model (fp32, eval) dan kembalikan bundle siap pakai."""

    cfg = config or SentimentModelConfig()

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    path, revision = resolve_model_path(cfg)

    model_config = AutoConfig.from_pretrained(path, local_files_only=True)
    if int(getattr(model_config, "num_labels", 0)) != 3:
        raise ModelCheckError(f"Model harus punya 3 kelas; num_labels={model_config.num_labels}")
    sentiment_indices = check_label_mapping(model_config.id2label)

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    check_tokenizer(tokenizer, cfg.max_length, cfg.strict_tokenizer_check)

    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    device = pick_device(cfg.device, torch.cuda.is_available(), bool(mps_ok))

    model = AutoModelForSequenceClassification.from_pretrained(
        path, config=model_config, local_files_only=True, use_safetensors=True
    )
    model = model.float().to(device)
    model.eval()

    def forward(batch: dict) -> np.ndarray:
        tensors = {
            key: torch.as_tensor(value).to(device)
            for key, value in batch.items()
            if key in _MODEL_INPUT_KEYS
        }
        with torch.inference_mode():
            logits = model(**tensors).logits
        return logits.detach().float().cpu().numpy()

    meta = {
        "repo_id": None if cfg.model_dir else cfg.repo_id,
        "revision": revision,
        "model_path": path,
        "device": device,
        "dtype": "float32",
        "max_length": cfg.max_length,
        "id2label": {int(k): str(v) for k, v in model_config.id2label.items()},
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    return SentimentBundle(
        tokenizer=tokenizer,
        forward=forward,
        sentiment_indices=sentiment_indices,
        device=device,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        meta=meta,
    )


# ============================================================
# PREDIKSI
# ============================================================

def _token_lengths(tokenizer: Any, texts: list[str], chunk: int = 2048) -> np.ndarray:
    """Panjang token tanpa pemotongan (termasuk token khusus [CLS] dan [SEP])."""

    lengths: list[int] = []
    for start in range(0, len(texts), chunk):
        encoded = tokenizer(
            texts[start : start + chunk],
            truncation=False,
            padding=False,
            add_special_tokens=True,
            verbose=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return np.asarray(lengths, dtype=np.int64)


def predict_sentiment_probs(
    texts: Sequence[object] | pd.Series,
    bundle: SentimentBundle | None = None,
    config: SentimentModelConfig | None = None,
    *,
    batch_size: int | None = None,
    return_diagnostics: bool = False,
    return_logits: bool = False,
    log_every: int = 100,
) -> pd.DataFrame:
    """Softmax sentimen per teks -> DataFrame ``p_negatif, p_netral, p_positif``.

    - Indeks hasil sama dengan indeks ``texts`` bila berupa Series.
    - Teks kosong/NaN tidak dibuang: barisnya tetap ada dengan probabilitas NaN.
    - ``return_diagnostics`` menambah ``n_token`` (tanpa pemotongan), ``terpotong`` (> max_length),
      ``kosong``. ``return_logits`` menambah ``logit_*`` mentah (urutan sama dengan probabilitas).
    - Tidak ada gradien; urutan kolom mengikuti ``config.id2label`` model.
    """

    if bundle is None:
        bundle = load_sentiment_model(config)
    size = bundle.batch_size if batch_size is None else batch_size
    if size <= 0:
        raise ValueError("batch_size harus lebih besar dari 0")

    series = texts if isinstance(texts, pd.Series) else pd.Series(list(texts), dtype="object")
    values = series.tolist()
    total = len(values)

    valid_positions: list[int] = []
    valid_texts: list[str] = []
    for position, value in enumerate(values):
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text:
            valid_positions.append(position)
            valid_texts.append(text)

    lengths = _token_lengths(bundle.tokenizer, valid_texts)
    order = np.argsort(lengths, kind="stable")          # kurangi padding; hasil dikembalikan ke urutan asal

    logits_model = np.full((len(valid_texts), 3), np.nan, dtype=np.float64)
    n_batches = (len(valid_texts) + size - 1) // size
    for batch_number, start in enumerate(range(0, len(valid_texts), size), start=1):
        batch_idx = order[start : start + size]
        encoded = bundle.tokenizer(
            [valid_texts[i] for i in batch_idx],
            padding=True,
            truncation=True,
            max_length=bundle.max_length,
            return_tensors="np",
        )
        logits = np.asarray(bundle.forward(dict(encoded)), dtype=np.float64)
        if logits.shape != (len(batch_idx), 3):
            raise ModelCheckError(
                f"Bentuk logit {logits.shape} tidak sesuai ({len(batch_idx)}, 3)"
            )
        logits_model[batch_idx] = logits
        if log_every and (batch_number % log_every == 0 or batch_number == n_batches):
            LOGGER.info("Batch %d/%d selesai", batch_number, n_batches)

    finite = np.isfinite(logits_model).all(axis=1)
    if not finite.all():
        bad = [valid_positions[i] for i in np.flatnonzero(~finite)[:10]]
        raise FloatingPointError(
            f"Logit tidak finite (NaN/Inf) pada {int((~finite).sum())} teks; posisi awal: {bad}"
        )

    shifted = logits_model - logits_model.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs_model = exp / exp.sum(axis=1, keepdims=True)

    order_idx = [bundle.sentiment_indices[label] for label in _CANONICAL]
    probs = probs_model[:, order_idx]
    logits_ordered = logits_model[:, order_idx]

    if len(probs) and np.abs(probs.sum(axis=1) - 1.0).max() > 1e-6:
        raise FloatingPointError("Probabilitas tidak berjumlah 1 (toleransi 1e-6)")

    prob_values = np.full((total, 3), np.nan)
    prob_values[valid_positions] = probs
    result = pd.DataFrame(prob_values, index=series.index, columns=list(PROB_COLUMNS))

    if return_logits:
        logit_values = np.full((total, 3), np.nan)
        logit_values[valid_positions] = logits_ordered
        for j, column in enumerate(LOGIT_COLUMNS):
            result[column] = logit_values[:, j]

    if return_diagnostics:
        n_token = np.full(total, -1, dtype=np.int64)
        n_token[valid_positions] = lengths
        empty = n_token < 0
        result["n_token"] = pd.array(np.where(empty, 0, n_token), dtype="Int64")
        result.loc[empty, "n_token"] = pd.NA
        result["terpotong"] = (n_token > bundle.max_length)
        result["kosong"] = empty

    return result


def argmax_label(probs: pd.DataFrame) -> pd.Series:
    """Label argmax sebagai kolom bantu (NaN bila probabilitas NaN). Bukan pengganti probabilitas."""

    values = probs[list(PROB_COLUMNS)].to_numpy(dtype=float)
    ok = ~np.isnan(values).any(axis=1)
    labels = np.full(len(values), None, dtype=object)
    labels[ok] = np.array(LABEL_NAMES, dtype=object)[values[ok].argmax(axis=1)]
    return pd.Series(labels, index=probs.index, name="label_argmax")


# ============================================================
# PEMERIKSAAN MODEL (butuh bundle)
# ============================================================

def run_model_checks(
    bundle: SentimentBundle,
    sample_texts: Sequence[str],
    tolerance: float = 1e-4,
) -> list[dict]:
    """Jalankan pemeriksaan yang diminta; tidak melempar, mengembalikan daftar hasil."""

    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"nama": name, "lulus": bool(ok), "detail": detail})

    id2label = bundle.meta.get("id2label")
    if id2label is not None:
        try:
            check_label_mapping(id2label)
            add("id2label = {0:negatif, 1:netral, 2:positif}", True, str(id2label))
        except ModelCheckError as exc:
            add("id2label = {0:negatif, 1:netral, 2:positif}", False, str(exc))

    actual = getattr(bundle.tokenizer, "model_max_length", None)
    add(
        f"tokenizer.model_max_length = {bundle.max_length}",
        actual == bundle.max_length,
        f"nilai: {actual}",
    )

    sample = [str(text) for text in sample_texts if str(text).strip()]
    if not sample:
        add("sampel teks tersedia", False, "tidak ada teks valid untuk diuji")
        return checks

    try:
        batch = predict_sentiment_probs(sample, bundle=bundle, batch_size=len(sample), log_every=0)
        single = pd.concat(
            [predict_sentiment_probs([text], bundle=bundle, batch_size=1, log_every=0) for text in sample],
            ignore_index=True,
        )
    except Exception as exc:  # tampilkan sebagai hasil, bukan crash
        add("prediksi sampel berjalan", False, f"{type(exc).__name__}: {exc}")
        return checks

    values = batch.to_numpy(dtype=float)
    add("tidak ada NaN/Inf pada probabilitas", bool(np.isfinite(values).all()), f"{len(sample)} teks diuji")
    deviation = float(np.abs(values.sum(axis=1) - 1.0).max())
    add("probabilitas berjumlah 1 (<=1e-6)", deviation <= 1e-6, f"deviasi maks {deviation:.2e}")
    diff = float(np.abs(values - single.to_numpy(dtype=float)).max())
    add(f"hasil batch = hasil satu-satu (<= {tolerance:g})", diff <= tolerance, f"selisih maks {diff:.2e}")
    return checks


# ============================================================
# RINGKASAN SANITY CHECK (tanpa torch)
# ============================================================

def summarize_predictions(
    texts: pd.Series,
    probs: pd.DataFrame,
    max_length: int = MAX_LENGTH,
    short_words: int = 2,
) -> dict:
    """Ringkasan: distribusi argmax, rata-rata p_netral, terpotong, kosong/sangat pendek."""

    if "n_token" not in probs.columns:
        raise ValueError("probs harus dihasilkan dengan return_diagnostics=True")

    n_total = int(len(probs))
    empty = probs["kosong"].to_numpy(dtype=bool)
    inferred = ~empty
    n_inferred = int(inferred.sum())

    words = (
        texts.reset_index(drop=True)
        .where(texts.reset_index(drop=True).notna(), "")
        .astype(str)
        .str.split()
        .str.len()
        .to_numpy()
    )
    very_short = inferred & (words <= short_words)

    labels = argmax_label(probs)
    counts = labels.dropna().value_counts()
    distribution = {
        name: {
            "n": int(counts.get(name, 0)),
            "persen": round(100.0 * int(counts.get(name, 0)) / n_inferred, 2) if n_inferred else None,
        }
        for name in LABEL_NAMES
    }

    prob_values = probs.loc[inferred, list(PROB_COLUMNS)].to_numpy(dtype=float)
    means = {
        column: (float(prob_values[:, j].mean()) if n_inferred else None)
        for j, column in enumerate(PROB_COLUMNS)
    }
    deviation = float(np.abs(prob_values.sum(axis=1) - 1.0).max()) if n_inferred else 0.0

    tokens = probs.loc[inferred, "n_token"].astype(float).to_numpy()
    truncated = int(probs["terpotong"].to_numpy(dtype=bool).sum())

    return {
        "n_total": n_total,
        "n_kosong": int(empty.sum()),
        "n_diinferensi": n_inferred,
        "n_sangat_pendek": int(very_short.sum()),
        "ambang_sangat_pendek_kata": short_words,
        "n_terpotong": truncated,
        "persen_terpotong": round(100.0 * truncated / n_inferred, 2) if n_inferred else None,
        "max_length": max_length,
        "distribusi_argmax": distribution,
        "rata2_probabilitas": means,
        "rata2_p_netral": means["p_netral"],
        "n_nan_pada_diinferensi": int(np.isnan(prob_values).sum()),
        "deviasi_maks_jumlah_prob": deviation,
        "token_median": float(np.median(tokens)) if n_inferred else None,
        "token_p95": float(np.percentile(tokens, 95)) if n_inferred else None,
        "token_maks": int(tokens.max()) if n_inferred else None,
    }


KETERBATASAN = (
    "Model difine-tune pada SmSA (ulasan umum), bukan ulasan marketplace. Kualitas pada domain "
    "ulasan ponsel BELUM terukur; evaluasi domain memerlukan sampel berlabel manual (rencana 150 baris, "
    "bukan dari rating bintang). Angka di sini adalah distribusi keluaran model, bukan akurasi. "
    "Jangan mengklaim akurasi di domain marketplace."
)


def format_summary_md(summary: dict, meta: dict, checks: list[dict] | None = None) -> str:
    """Ringkasan sanity check dalam Markdown (Bahasa Indonesia)."""

    lines = ["# Ringkasan inferensi sentimen (RM 1)", ""]
    lines += ["## Model dan lingkungan", ""]
    for key in ("repo_id", "revision", "model_path", "device", "dtype", "max_length", "torch", "transformers"):
        if key in meta:
            lines.append(f"- {key}: `{meta[key]}`")
    for key in ("input", "output", "jumlah_baris_input", "waktu_mulai", "durasi_detik"):
        if key in meta:
            lines.append(f"- {key}: `{meta[key]}`")
    lines += ["", "## Sanity check", ""]
    lines += [
        f"- Baris total: {summary['n_total']:,}",
        f"- Diinferensi: {summary['n_diinferensi']:,}",
        f"- Teks kosong/NaN (probabilitas NaN): {summary['n_kosong']:,}",
        f"- Sangat pendek (<= {summary['ambang_sangat_pendek_kata']} kata): {summary['n_sangat_pendek']:,}",
        f"- Terpotong (> {summary['max_length']} token): {summary['n_terpotong']:,} ({summary['persen_terpotong']}%)",
        f"- Rata-rata p_netral: {summary['rata2_p_netral']}",
        f"- Deviasi maks jumlah probabilitas dari 1: {summary['deviasi_maks_jumlah_prob']:.2e}",
        f"- NaN pada baris yang diinferensi: {summary['n_nan_pada_diinferensi']}",
        f"- Token: median {summary['token_median']}, p95 {summary['token_p95']}, maks {summary['token_maks']}",
        "",
        "Distribusi argmax (kolom bantu, bukan pengganti probabilitas):",
        "",
        "| Kelas | n | % |",
        "| --- | ---: | ---: |",
    ]
    for name in LABEL_NAMES:
        item = summary["distribusi_argmax"][name]
        lines.append(f"| {name} | {item['n']:,} | {item['persen']} |")
    if checks:
        lines += ["", "## Pemeriksaan model", "", "| Pemeriksaan | Lulus | Detail |", "| --- | :-: | --- |"]
        for check in checks:
            lines.append(f"| {check['nama']} | {'ya' if check['lulus'] else 'TIDAK'} | {check['detail']} |")
    lines += ["", "## Keterbatasan", "", KETERBATASAN, ""]
    return "\n".join(lines)
