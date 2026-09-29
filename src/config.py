from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class ExperimentConfig:

    # =========================================================
    # PROJECT
    # =========================================================

    base_dir: Path = Path(__file__).resolve().parent.parent

    # Folder data mentah
    raw_data_dir: Path | None = Path(__file__).resolve().parent.parent / "data" / "raw"

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
    
    # ============================================================
    # LEXICAL TEXT PREPROCESSING
    # ============================================================
    
    lexical_dictionary_version: str = "lex-v1"
    lexical_mark_negation: bool = True
    lexical_min_token_frequency: int = 2
    
    # ============================================================
    # PMI / SEMANTIC ORIENTATION
    # ============================================================

    pmi_window_size: int = 10
    pmi_min_count: int = 2 
    
    # ============================================================
    # BOOTSTRAP / CONFIDENCE INTERVAL
    # ============================================================
    
    bootstrap_iterations: int = 50
    bootstrap_seed_sample_size: int = 7
    confidence_level: float = 0.95

    # =========================================================
    # TF-IDF
    # =========================================================

    tfidf_max_features: int = 30_000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_sublinear_tf: bool = True
    
    # ============================================================
    # SUSPICION SCORE
    # ============================================================

    suspicion_aggregation: str = "mean"
    
    # =========================================================
    # SENTIMENT
    # =========================================================
    # Model yang digunakan untuk sentiment inference
    sentiment_model_name: str = (
        "mdhugol/"
        "indonesia-bert-sentiment-classification"
    )
    
    sentiment_model_name: str | None = None
    sentiment_batch_size: int = 32


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
        
        if self.raw_data_dir is not None:
            return Path(self.raw_data_dir)

        return self.base_dir / "data"


    @property
    def reference_dir(self) -> Path:
        return self.base_dir / "references"


    @property
    def output_dir(self) -> Path:
        return self.base_dir / "outputs"
    # =========================================================
    # REFERENCE / LEXICAL RESOURCE PATHS
    # =========================================================
    @property
    def genuine_seed_path(self) -> Path:
        return (
            self.lexical_reference_dir
            / "genuine_seeds.txt"
        )


    @property
    def counterfeit_seed_path(self) -> Path:
        return (
            self.lexical_reference_dir
            / "counterfeit_seeds.txt"
        )


    @property
    def domain_dictionary_path(self) -> Path:
        return (
            self.lexical_reference_dir
            / "domain_dictionary.json"
        )


    @property
    def public_dictionary_path(self) -> Path:
        return (
            self.lexical_reference_dir
            / "public_dictionary.json"
        )


    @property
    def stopwords_path(self) -> Path:
        return (
            self.lexical_reference_dir
            / "stopwords.txt"
        )
    
    # =========================================================
    # OUTPUT PATHS
    # =========================================================

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
    # =========================================================
    # CHECKPOINT PATHS
    # =========================================================
    
    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"


    @property
    def cleaning_done_dir(self) -> Path:
        return self.checkpoint_dir / "cleaning"
    
    @property
    def sentiment_done_dir(self) -> Path:
        return self.checkpoint_dir / "sentiment"
    
    # =========================================================
    # MANIFEST / ARTIFACT PATHS
    # =========================================================
    
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
            
        # -----------------------------------------------------
        # PREPROCESSING
        # -----------------------------------------------------
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
            
        if self.lexical_min_token_frequency < 1: 
            raise ValueError(
                "lexical_min_token_frequency minimal 1."
            )
        # -----------------------------------------------------
        # PMI
        # -----------------------------------------------------
        
        if self.pmi_window_size < 1:
            raise ValueError(
                "pmi_window_size minimal 1"
            )
        
        if self.pmi_min_count < 1 : 
            raise ValueError(
                "pmi min count minimal 1"
            )
            
        # -----------------------------------------------------
        # BOOTSTRAP
        # ----------------------------------------------------
        if self.bootstrap_iterations < 1:
            raise ValueError(
                "bootstrap iter minimal 1"
            )
            
        if self.bootstrap_seed_sample_size < 1 : 
            raise ValueError(
                "bootstrap seed sample size minimal 1"
            )
        
        if not 0 < self.confidence_level < 1: 
            raise ValueError(
                "confidence level harus berada antara 0 dan 1"
            )
        
        # -----------------------------------------------------
        # SPLIT
        # -----------------------------------------------------
        
        if not 0 < self.test_size < 1:
            raise ValueError(
                "test_size harus berada antara 0 dan 1."
            )

        if self.min_reviews_per_product < 1:
            raise ValueError(
                "min_reviews_per_product minimal 1."
            )
            
        # -----------------------------------------------------
        # HMM
        # -----------------------------------------------------
        
        if self.hmm_components < 2:
            raise ValueError(
                "hmm_components minimal 2."
            )
        
        if self.hmm_n_iter <= 0: 
            raise ValueError(
                "hmm n iter harus lebih besar dari 0"
            )
        
        valid_covariance_types = {
            "spherical",
            "diag",
            "full",
            "tied",
        }
        
        if (
            self.hmm_covariance_type not in valid_covariance_types
        ): 
            raise ValueError(
                "hmm cov type tidak valid"
            )
        
        # -----------------------------------------------------
        # SENTIMENT
        # -----------------------------------------------------
        
        if self.sentiment_batch_size <= 0:
            raise ValueError(
                "sentiment_batch_size harus lebih besar dari 0."
            )

        if self.sentiment_max_length <= 0:
            raise ValueError(
                "sentiment_max_length harus lebih besar dari 0."
            )
        # -----------------------------------------------------
        # SUSPICION SCORE
        # -----------------------------------------------------
        
        valid_aggregations = {
            "mean",
            "max",
        }

        if (
            self.suspicion_aggregation
            not in valid_aggregations
        ):
            raise ValueError(
                "suspicion_aggregation harus "
                "'mean' atau 'max'."
            )

    def setup(self) -> None:

        self.validate()
        self.create_directories()


    