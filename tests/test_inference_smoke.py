"""Ten-example smoke test for a locally saved, fine-tuned IndoBERT model."""

import os
from pathlib import Path

import pandas as pd
import pytest

from src.inference import predict_sentiment_probs, resolve_sentiment_indices


SENTENCES = [
    "Barangnya bagus banget, sesuai foto, recommended!",
    "Kecewa, barang rusak, tidak sesuai deskripsi.",
    "Barang sampai, belum dicoba.",
    "Kualitasnya mantap, pengiriman cepat dan aman.",
    "Produk cacat, layar tidak menyala sejak dibuka.",
    "Paket diterima oleh saya pagi ini.",
    "Sangat puas, berfungsi dengan baik dan harganya murah.",
    "Buruk sekali, cepat panas dan baterainya boros.",
    "Warna hitam, kapasitas memori 128 GB.",
    "Tidak rekomendasi, sering mati sendiri setelah dipakai.",
]


def test_model_label_mapping_is_explicit_and_order_independent():
    assert resolve_sentiment_indices({0: "neutral", 1: "negative", 2: "positive"}) == {
        "negative": 1,
        "neutral": 0,
        "positive": 2,
    }


def test_ten_sentence_model_smoke():
    model_dir = os.environ.get("INDOBERT_SENTIMENT_MODEL_DIR")
    if not model_dir or not Path(model_dir).is_dir():
        pytest.skip("Set INDOBERT_SENTIMENT_MODEL_DIR ke folder model fine-tuning tersimpan")

    output = predict_sentiment_probs(
        pd.DataFrame({"text_light": SENTENCES}), model_dir=model_dir, batch_size=4
    )
    assert len(output) == 10
    assert output[["p_negative", "p_neutral", "p_positive"]].notna().all().all()
    assert ((output[["p_negative", "p_neutral", "p_positive"]].sum(axis=1) - 1).abs() < 1e-5).all()
    print(output.to_string(index=False))
