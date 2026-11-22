"""
LearnMate Configuration and Hyperparameters
M.Tech AI-ML (First Year) Deep Learning Networks Research Project
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Reproducibility Seeds
RANDOM_SEED = 42

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
CORPUS_DIR = os.path.join(DATA_DIR, "raw", "dln_course_corpus")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EVAL_DIR = os.path.join(DATA_DIR, "evaluation")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "paper", "figures")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Model Settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_EMBEDDING_DIM = 384

# Chunking & Retrieval Parameters
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 25
TOP_K_RETRIEVAL = 3
RRF_K = 60
WEIGHT_DENSE = 0.6
WEIGHT_BM25 = 0.4

# Verification & Abstention Hyperparameters
DEFAULT_COSINE_THRESHOLD = 0.55
EVIDENCE_AGREEMENT_LAMBDA = 0.5  # Weight for penalizing contradiction
CONFORMAL_ALPHA = 0.05          # Maximum allowed ungrounded error rate (5%)
CALIBRATION_TEMPERATURE = 1.35  # Fitted temperature scaling factor
DEFAULT_ABSTENTION_THRESHOLD = 0.62  # Calibrated conformal threshold tau*

# Evaluation Settings
NUM_CALIBRATION_BINS = 10
