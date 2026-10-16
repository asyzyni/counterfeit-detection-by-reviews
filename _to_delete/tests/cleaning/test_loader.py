"""Tes pembacaan format tunggal dan pengaman pipeline lama dengan data sintetis."""

from pathlib import Path

import pandas as pd
import pytest

from src.cleaning import load_clean_reviews
from src.cleaning.config import CleaningConfig
from src.config import ExperimentConfig
from src.preprocessing import DataPreprocessor, inspect_cleaned_data


@pytest.fixture
def reviews_path(tmp_path: Path) -> Path:
    path = tmp_path / "hasil cleaning" / "reviews_clean.parquet"
    path.parent.mkdir()
    pd.DataFrame(
        {
            "review_id": ["b2", "a3", "a1", "b1", "a2"],
            "product_id": ["b", "a", "a", "b", "a"],
            "text_light": ["bagus", "pendek", "awet", "cepat", "aman"],
            "timestamp": pd.to_datetime(["2026-01-02", None, "2026-01-01", None, None]),
            "include_for_inference": [True, False, True, pd.NA, True],
            "seq_in_product": [2, 3, 1, 1, 2],
            "nama_toko": ["Toko B", "Toko A", "Toko A", "Toko B", "Toko A"],
        }
    ).astype({"include_for_inference": "boolean"}).to_parquet(path, index=False)
    return path


@pytest.mark.parametrize(
    ("only_included", "expected_ids"),
    [(True, ["a1", "a2", "b2"]), (False, ["a1", "a2", "a3", "b1", "b2"])],
)
def test_loader_filters_sorts_and_preserves_types(reviews_path, only_included, expected_ids):
    source = pd.read_parquet(reviews_path)
    result = load_clean_reviews(path=reviews_path, only_included=only_included)

    expected = source.set_index("review_id").loc[expected_ids].reset_index()
    pd.testing.assert_frame_equal(result, expected)
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
    assert result["timestamp"].isna().any()
    for column in ["review_id", "product_id", "text_light"]:
        assert result[column].map(lambda value: isinstance(value, str)).all()


def test_loader_uses_config_and_default_without_raw_data(reviews_path, monkeypatch):
    config = CleaningConfig(output_dir=reviews_path.parent, raw_dir=reviews_path.parent / "absent")
    expected = load_clean_reviews(path=reviews_path)

    pd.testing.assert_frame_equal(load_clean_reviews(config=config), expected)
    monkeypatch.setattr("src.cleaning.loader.CleaningConfig", lambda: config)
    pd.testing.assert_frame_equal(load_clean_reviews(), expected)
    assert not config.raw_dir.exists()


def test_explicit_path_overrides_config(reviews_path, tmp_path):
    config = CleaningConfig(output_dir=tmp_path / "absent")
    assert len(load_clean_reviews(config=config, path=str(reviews_path))) == 3
    assert not config.output_dir.exists()


def test_missing_file_has_actionable_error(tmp_path):
    path = tmp_path / "absent" / "reviews_clean.parquet"
    with pytest.raises(FileNotFoundError, match="python -m src.cleaning.run") as error:
        load_clean_reviews(path=path)
    assert str(path) in str(error.value)
    assert not path.parent.exists()


@pytest.mark.parametrize("columns", [["nama_toko", "text_light"], []])
def test_column_selection_keeps_inference_contract(reviews_path, columns):
    result = load_clean_reviews(path=reviews_path, columns=columns)
    assert list(result.columns) == list(
        dict.fromkeys([*columns, "review_id", "product_id", "text_light", "timestamp"])
    )
    assert result["review_id"].tolist() == ["a1", "a2", "b2"]


def test_unknown_requested_column_has_clear_error(reviews_path):
    with pytest.raises(ValueError, match="Kolom yang diminta.*tidak_ada"):
        load_clean_reviews(path=reviews_path, columns=["tidak_ada"])


@pytest.mark.parametrize("missing", ["text_light", "seq_in_product", "include_for_inference"])
def test_missing_required_column_has_clear_error(reviews_path, missing):
    frame = pd.read_parquet(reviews_path).drop(columns=missing)
    frame.to_parquet(reviews_path, index=False)
    with pytest.raises(ValueError, match=f"Kolom wajib.*{missing}"):
        load_clean_reviews(path=reviews_path)


def test_all_excluded_returns_empty_frame_with_columns(reviews_path):
    frame = pd.read_parquet(reviews_path).assign(include_for_inference=False)
    frame.to_parquet(reviews_path, index=False)
    result = load_clean_reviews(path=reviews_path)
    assert result.empty
    assert list(result.columns) == list(frame.columns)


@pytest.fixture
def legacy_preprocessor(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "produk.csv").write_text("review\nbarang bagus\n", encoding="utf-8")
    # Seluruh direktori konfigurasi lama berada di folder tes sementara.
    return DataPreprocessor(ExperimentConfig(base_dir=tmp_path, raw_data_dir=raw))


@pytest.mark.parametrize("method", ["clean_all", "clean_single_file", "get_cleaned_files", "inspect_cleaned_data"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_legacy_rejects_new_format_without_writes(legacy_preprocessor, reviews_path, method, overwrite):
    preprocessor = legacy_preprocessor
    config = preprocessor.config
    config.overwrite = overwrite
    target = config.cleaned_dir / "reviews_clean.parquet"
    target.write_bytes(reviews_path.read_bytes())
    before = {p.relative_to(config.output_dir): p.read_bytes()
              for p in config.output_dir.rglob("*") if p.is_file()}

    message = "pakai python -m src.cleaning.run" if method.startswith("clean_") else "load_clean_reviews"
    with pytest.raises(RuntimeError, match=message):
        if method == "inspect_cleaned_data":
            inspect_cleaned_data(config.cleaned_dir)
        elif method == "clean_single_file":
            preprocessor.clean_single_file(config.data_dir / "produk.csv")
        else:
            getattr(preprocessor, method)()

    after = {p.relative_to(config.output_dir): p.read_bytes()
             for p in config.output_dir.rglob("*") if p.is_file()}
    assert before == after
    assert list(config.cleaned_dir.iterdir()) == [target]
    assert list(config.cleaning_done_dir.iterdir()) == []


def test_legacy_per_product_format_still_works(legacy_preprocessor):
    preprocessor = legacy_preprocessor
    config = preprocessor.config
    manifest = preprocessor.clean_all()
    assert manifest["status"].tolist() == ["success"]
    assert preprocessor.get_done_marker("produk").is_file()
    assert (config.cleaned_dir / "produk" / "_meta.json").is_file()
    assert preprocessor.get_cleaned_files()["product_id"].tolist() == ["produk"]
    report, sample = inspect_cleaned_data(config.cleaned_dir)
    assert report["total_parquet_files"] == 1
    assert sample["text_light"].tolist() == ["barang bagus"]
    assert preprocessor.clean_single_file(config.data_dir / "produk.csv")["status"] == "skipped"
