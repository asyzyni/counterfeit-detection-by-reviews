import numpy as np
import pandas as pd
import pytest

from src.cleaning.run_timestamps import merge_sentiment_timestamps, preserve_original_columns, temporal_sequences
from src import export_sekuens as es


def fixture():
    old = pd.DataFrame({
        "review_id": ["a", "b", "c"], "product_id": ["P"] * 3,
        "source_file": ["b.csv", "a.csv", "a.csv"], "row_order": [0, 0, 1],
        "seq_in_product": [1, 2, 3], "timestamp": pd.to_datetime(["2026-07-29", None, None]),
        "timestamp_has_time": [False] * 3, "timestamp_status": ["ok", "relatif", "relatif"],
        "timestamp_raw": ["2026-07-29", "12 jam lalu", "hari ini"],
        "review": ["baik"] * 3, "text_light": ["baik"] * 3, "include_for_inference": [True] * 3,
        "text_proposal": ["baik"] * 3,
    })
    new = old.drop(columns="text_proposal").copy()
    new["timestamp"] = pd.Series(pd.Timestamp("2026-07-29"), index=new.index, dtype="datetime64[ns]")
    new["timestamp_status"] = ["ok", "relatif_dikonversi", "relatif_dikonversi"]
    new["timestamp_is_estimated"] = [False, True, True]
    new["timestamp_anchor"] = pd.to_datetime([None, "2026-07-29", "2026-07-29"])
    new["timestamp_anchor_source"] = ["tidak_ada", "mtime", "mtime"]
    new["timestamp_relative_raw"] = ["", "12 jam lalu", "hari ini"]
    new["seq_in_product"] = [3, 1, 2]
    return old, new.iloc[[1, 2, 0]].reset_index(drop=True)


def test_migration_joins_by_id_preserves_text_and_exact_probability_strings():
    old, new = fixture()
    complete = preserve_original_columns(old, new)
    sentiment = es.read_input_bytes(es.to_csv_bytes(old))
    sentiment["p_positif"] = ["0.70000000000000001", "0.6", "0.5"]
    sentiment["p_netral"] = ["0.2", "0.2", "0.2"]
    sentiment["p_negatif"] = ["0.1", "0.2", "0.3"]
    updated = merge_sentiment_timestamps(sentiment, complete)
    assert updated.review_id.tolist() == old.review_id.tolist()
    assert updated.seq_in_product.tolist() == ["3", "1", "2"]
    assert updated.p_positif.tolist() == sentiment.p_positif.tolist()
    sequences, report = temporal_sequences(updated)
    np.testing.assert_allclose(sequences.P[:, 0], [0.6, 0.5, 0.7])
    assert report["n_reviews_used"] == 3
    assert report["order_rule"][:4] == ["product_id", "timestamp", "source_file", "row_order"]


def test_migration_rejects_text_id_and_absolute_timestamp_changes():
    old, new = fixture()
    with pytest.raises(AssertionError):
        preserve_original_columns(old, new.assign(text_light="diubah"))
    with pytest.raises(ValueError, match="review_id"):
        preserve_original_columns(old, new.iloc[:2])
    new.loc[new.review_id.eq("a"), "timestamp"] = pd.Timestamp("2026-07-30")
    with pytest.raises(AssertionError):
        preserve_original_columns(old, new)


def test_join_rejects_duplicate_or_missing_ids():
    old, new = fixture()
    sentiment = es.read_input_bytes(es.to_csv_bytes(old))
    with pytest.raises(ValueError, match="unik"):
        merge_sentiment_timestamps(pd.concat([sentiment, sentiment.iloc[[0]]]), new)
    with pytest.raises(ValueError, match="berbeda"):
        merge_sentiment_timestamps(sentiment.iloc[:2], new)
