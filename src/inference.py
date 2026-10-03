"""Inference helpers for saved three-class IndoBERT sentiment models."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


LOGGER = logging.getLogger(__name__)
SENTIMENT_COLUMNS = ("p_negative", "p_neutral", "p_positive")
SENTIMENT_LABELS = ("negative", "neutral", "positive")


def _label_to_sentiment(label: object) -> str:
    """Normalize a model label and reject labels with unknown meaning."""
    normalized = re.sub(r"[^a-z]", "", str(label).lower())
    aliases = {
        "negative": "negative",
        "negatif": "negative",
        "neutral": "neutral",
        "netral": "neutral",
        "positive": "positive",
        "positif": "positive",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Label model {label!r} tidak dikenali sebagai negative, neutral, atau positive"
        )
    return aliases[normalized]


def resolve_sentiment_indices(id2label: dict) -> dict[str, int]:
    """Return sentiment -> logit index, independent of the model's class order."""
    parsed = {int(index): _label_to_sentiment(label) for index, label in id2label.items()}
    assert len(parsed) == 3, f"Model harus punya tepat 3 kelas sentimen; didapat {parsed}"
    assert set(parsed.values()) == set(SENTIMENT_LABELS), (
        "id2label harus memetakan tepat ke negative, neutral, positive; "
        f"didapat {parsed}"
    )
    assert sorted(parsed) == [0, 1, 2], f"Indeks kelas harus 0, 1, 2; didapat {parsed}"
    return {sentiment: next(i for i, label in parsed.items() if label == sentiment)
            for sentiment in SENTIMENT_LABELS}


def predict_sentiment_probs(
    df: pd.DataFrame,
    model_dir: str,
    text_col: str = "text_light",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Append negative/neutral/positive softmax probabilities to unlabeled rows.

    Blank and null text rows are retained in their original positions with NaN
    probability values. Model artifacts are loaded locally from ``model_dir``.
    """
    if text_col not in df.columns:
        raise KeyError(f"Kolom teks {text_col!r} tidak ditemukan")
    if batch_size <= 0:
        raise ValueError("batch_size harus lebih besar dari 0")
    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Direktori model tidak ditemukan: {model_path}")

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    id2label = getattr(config, "id2label", None) or {}
    sentiment_indices = resolve_sentiment_indices(id2label)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=config, local_files_only=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    result = df.copy()
    for column in SENTIMENT_COLUMNS:
        result[column] = np.nan

    text_values = result[text_col]
    valid_positions: list[int] = []
    valid_texts: list[str] = []
    skipped = 0
    for position, value in enumerate(text_values.tolist()):
        if pd.isna(value) or not str(value).strip():
            skipped += 1
            continue
        valid_positions.append(position)
        valid_texts.append(str(value))

    if skipped:
        LOGGER.warning("Melewati %d baris dengan teks kosong/NaN", skipped)

    for start in range(0, len(valid_texts), batch_size):
        batch_texts = valid_texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        assert probabilities.shape[1] == 3, (
            f"Model menghasilkan {probabilities.shape[1]} logit, seharusnya 3"
        )
        ordered = np.column_stack(
            [probabilities[:, sentiment_indices[label]] for label in SENTIMENT_LABELS]
        )
        positions = valid_positions[start : start + len(batch_texts)]
        result.iloc[positions, result.columns.get_loc(SENTIMENT_COLUMNS[0])] = ordered[:, 0]
        result.iloc[positions, result.columns.get_loc(SENTIMENT_COLUMNS[1])] = ordered[:, 1]
        result.iloc[positions, result.columns.get_loc(SENTIMENT_COLUMNS[2])] = ordered[:, 2]

    return result
