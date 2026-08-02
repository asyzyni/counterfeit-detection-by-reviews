import os

# Project Roots
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data Paths
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
INTERIM_DATA_DIR = os.path.join(BASE_DIR, "data", "interim")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Model Paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
INDOBERT_CHECKPOINT_DIR = os.path.join(MODELS_DIR, "indobert_finetuned")
HMM_PARAMS_PATH = os.path.join(MODELS_DIR, "hmm_params.json")

# Results Paths
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

# Ensure directories exist
for d in [INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Random Seed
RANDOM_SEED = 42

# Label Mappings (Store level: 0 = Asli, 1 = Counterfeit)
STORE_LABELS = {
    "Panda_Vivo Y71": 0,          # Genuine model and base
    "PinDuoDuo Y17 Ram": 1,       # Counterfeit (fake 8GB specs)
    "TAOB1688_Ip 17 Pro": 1,      # Counterfeit (HDC clone)
    "TAOB88.SHOP_ip17": 1,        # Counterfeit (clone)
    "Temu Toko_Ip 17 Pro": 1      # Counterfeit (clone)
}

# Sentiment Map (From mdhugol/indonesia-bert-sentiment-classification: 0=positive, 1=neutral, 2=negative)
SENTIMENT_LABELS = {
    0: "positif",
    1: "netral",
    2: "negatif"
}

# IndoBERT Settings
INDOBERT_MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3

# HMM Settings
HMM_ALPHA = 0.1  # Laplace smoothing factor
