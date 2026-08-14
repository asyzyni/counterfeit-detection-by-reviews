from __future__ import annotations 

from dataclasses import dataclass, asdict 
from pathlib import Path 
from typing import Tuple 

@dataclass
class ExperimentConfig: 
    # =========================================================
    # PROJECT
    # =========================================================
    base_dir: Path 
    experiment_name: str = "Pipeline -1"
    random_state: int = 42 

    # =========================================================
    # Preprocessing
    # =========================================================
    chunk_size: int = 50_000 # jumlah baris yang dibaca per chunk csv 
    sample_rows: int = 500 # jumlah baris awal untuk deteksi struktur awal csv 
    min_review_length: int = 2 # review lebih pendek dari ini akn dibuang 
    overwrite: bool = False 
    parquet_compression: str = "snappy"

    # =========================================================
    # TF-IDF (weak labels)
    # =========================================================

    tfidf_max_features: int = 30_000
    tfidf_ngram_range: Tupe[int, int] = (1, 2)
    tf_idf_sublinear_tf: bool = True 

    # =========================================================
    # INDOBERT
    # =========================================================
    sentiment_model_name: str | None = None 
    sentiment_batch_size: int = 32 
    sentiment_max_length: int = 512 

    # =========================================================
    # HMM
    # =========================================================

    hmm_components: int = 2 
    hmm_covariance_type: str = "diag" ## ? 
    hmm_n_iter: int = 200 

    # =========================================================
    # Train - test
    # =========================================================
    test_size: float = 0.20 
    min_reviews_per_product: int = 3 

    # =========================================================
    # path
    # =========================================================
    @property 
    def data_dir(self) -> Path: 
        return self,base_dir / "data"
    
    @property 
    def reference_dir(self) -> Path: 
        return self.base_dir / "references"

    @property 
    def output_dir(self) -> Path: 
        return self.base_dir / "outputs"
    
    @property 
    def cleaned_dir(self) -> Path: 
        return.self.output_dir / "cleaned"

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
        return self.output_dir / "model"
    
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
        return (
            self.checkpoint_dir
            / "cleaning"
        )

    @property
    def sentiment_done_dir(self) -> Path:
        return (
            self.checkpoint_dir
            / "sentiment"
        )

    

    @property
    def cleaning_manifest_path(self) -> Path:
        return (
            self.logs_dir
            / "cleaning_manifest.csv"
        )


    @property
    def product_labels_path(self) -> Path:
        return (
            self.labels_dir
            / "product_labels.parquet"
        )


    @property
    def product_features_path(self) -> Path:
        return (
            self.features_dir
            / "product_features.parquet"
        )


    @property
    def predictions_path(self) -> Path:
        return (
            self.predictions_dir
            / "predictions.parquet"
        )
    
    # =========================================================
    # DIRECTORY PREPARATION
    # =========================================================

    def create_directories(self) -> None:
        """
        Membuat seluruh folder yang diperlukan pipeline.
        """

        directories = [
            self.data_dir,
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
    
    # =========================================================
    # VALIDATION
    # =========================================================

    def validate(self) -> None: 
        if self.chunk_size <= 0: 
            raise ValueError (
                "chunk harus lebih gede daeri 0"
            )
        
        if self.sample_rows <= 0:
            raise ValueError (
                "sample hrus lebih gde dri 0"
            )
        
        if self.min_review_length <= 1:
            raise ValueError(
                "minimal 1"
            )

        if not 0 < self.test_size < 1:
            raise ValueError(
                "hrus diantara 0 dan 1"
            )
        
        if self.min_reviews_per_product < 1:
            raise ValueError(
                mnimal 1
            )
        
