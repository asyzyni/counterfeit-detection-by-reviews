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
    # PROJECT
    # =========================================================