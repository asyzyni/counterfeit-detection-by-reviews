from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class ExperimentConfig:

    # =========================================================
    # PROJECT
    # =========================================================

    base_dir: Path("/Users/asyzyni/Desktop/TA /Kode dan Eksperimen/Kode TA/Lagi")

    # Folder data mentah boleh berada di luar project
    raw_data_dir: Path | None = Path("/Users/asyzyni/Desktop/TA /Kode dan Eksperimen/Kode TA/data")

    experiment_name: str = "Percobaan I"

    random_state: int = 42


    # =========================================================
    # PREPROCESSING
    # =========================================================

    chunk_size: int = 50_000
    sample_rows: int = 500
    min_review_length: int = 2
    overwrite: bool = False
    parquet_compression: str = "snappy"


    # =========================================================
    # TF-IDF
    # =========================================================

    tfidf_max_features: int = 30_000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_sublinear_tf: bool = True


    # =========================================================
    # SENTIMENT
    # =========================================================

    sentiment_model_name: str | None = None
    sentiment_batch_size: int = 32
    sentiment_max_length: int = 512


    # =========================================================
    # HMM
    # =========================================================

    hmm_components: int = 2
    hmm_covariance_type: str = "diag"
    hmm_n_iter: int = 200


    # =========================================================
    # TRAIN TEST
    # =========================================================

    test_size: float = 0.20
    min_reviews_per_product: int = 3


    # =========================================================
    # PATHS
    # =========================================================

    @property
    def data_dir(self) -> Path:
        """
        Folder raw data.

        Jika raw_data_dir diberikan, gunakan folder tersebut.
        Jika tidak, fallback ke base_dir / data.
        """

        if self.raw_data_dir is not None:
            return Path(self.raw_data_dir)

        return self.base_dir / "data"


    @property
    def reference_dir(self) -> Path:
        return self.base_dir / "references"


    @property
    def output_dir(self) -> Path:
        return self.base_dir / "outputs"


    @property
    def cleaned_dir(self) -> Path:
        return self.output_dir / "cleaned"


    @property
    def labels_dir(self) -> Path:
        return self.output_dir / "labels"


    @property
    def sentiment_dir(self) -> Path:
        return self.output_dir / "sentiment"


    @property
    def features_dir(self) -> Path:
        return self.output_dir / "features"


    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"


    @property
    def predictions_dir(self) -> Path:
        return self.output_dir / "predictions"


    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"


    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"


    @property
    def cleaning_done_dir(self) -> Path:
        return self.checkpoint_dir / "cleaning"


    @property
    def sentiment_done_dir(self) -> Path:
        return self.checkpoint_dir / "sentiment"


    @property
    def cleaning_manifest_path(self) -> Path:
        return self.logs_dir / "cleaning_manifest.csv"


    @property
    def product_labels_path(self) -> Path:
        return self.labels_dir / "product_labels.parquet"


    @property
    def product_features_path(self) -> Path:
        return self.features_dir / "product_features.parquet"


    @property
    def predictions_path(self) -> Path:
        return self.predictions_dir / "predictions.parquet"


    # =========================================================
    # SETUP
    # =========================================================

    def create_directories(self) -> None:

        directories = [
            self.reference_dir,
            self.output_dir,
            self.cleaned_dir,
            self.labels_dir,
            self.sentiment_dir,
            self.features_dir,
            self.models_dir,
            self.predictions_dir,
            self.logs_dir,
            self.checkpoint_dir,
            self.cleaning_done_dir,
            self.sentiment_done_dir,
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True
            )


    def validate(self) -> None:

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Folder raw data tidak ditemukan: "
                f"{self.data_dir}"
            )

        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size harus lebih besar dari 0."
            )

        if self.sample_rows <= 0:
            raise ValueError(
                "sample_rows harus lebih besar dari 0."
            )

        if self.min_review_length < 1:
            raise ValueError(
                "min_review_length minimal 1."
            )

        if not 0 < self.test_size < 1:
            raise ValueError(
                "test_size harus berada antara 0 dan 1."
            )

        if self.min_reviews_per_product < 1:
            raise ValueError(
                "min_reviews_per_product minimal 1."
            )

        if self.hmm_components < 2:
            raise ValueError(
                "hmm_components minimal 2."
            )

        if self.sentiment_batch_size <= 0:
            raise ValueError(
                "sentiment_batch_size harus lebih besar dari 0."
            )

        if self.sentiment_max_length <= 0:
            raise ValueError(
                "sentiment_max_length harus lebih besar dari 0."
            )

    def setup(self) -> None:

        self.validate()
        self.create_directories()