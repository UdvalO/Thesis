# ==============================================================================
# config_h1.py (H1)
# ==============================================================================
import torch
from pathlib import Path

# --- 1. PATHS ---
PROJECT_ROOT = Path(".../H1")

# Corresponds to R's 'SAVE_DIR'
DATA_DIR = PROJECT_ROOT / "data_processed"
SAVE_DIR = DATA_DIR

# Output Graph Directory (Where Python saves .pt files)
GRAPH_SAVE_DIR = PROJECT_ROOT / "graphs_processed"
GRAPH_SAVE_DIR.mkdir(exist_ok=True, parents=True)

# File Pointers (Mapped to R script outputs)
FINAL_DATA_FILE = DATA_DIR / "final_features_h1.rds"
TRAIN_TARGETS_FILE = DATA_DIR / "train_targets.rds"
TEST_TARGETS_FILE = DATA_DIR / "test_targets.rds"
PREPROC_PARAMS_FILE = DATA_DIR / "preproc_params.rds"

# --- 2. HARDWARE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 # adjust based on hardware

# --- 3. GRAPH HYPERPARAMETERS ---
# Explicit Topology caps (Calculated based on topology_analysis.py outputs)
MAX_DEGREE_STORAGE_GEO = 100     # Store up to 100 neighbors on disk
MAX_DEGREE_STORAGE_LENDER = 200  # Store up to 200 neighbors on disk

# --- 4. FEATURE DEFINITIONS ---
# Strict Paper Features
NUMERIC_FEATURES = [
    "fico", "mi_pct", "cnt_units", "dti", "ltv", "cnt_borr",
    "current_upb", "mths_remng", "current_int_rt"
]

BINARY_FEATURES = [
    "if_fthb", "if_prim_res", "if_corr", "if_sf", "if_purc", "if_sc", "if_delq_sts"
]

NODE_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES
NUM_INPUT_FEATURES = len(NODE_FEATURES)

# --- 5. MODEL ARCHITECTURE (Thesis Sec 2.2 / Table 2) ---
SEQUENCE_LENGTH = 6

# GAT Configurations
HIDDEN_DIM = 32       # Match H2 'GAT_HIDDEN'
NUM_HEADS = 2         # Multi-head attention count
DROPOUT = 0.5         # 50% dropout rate

# --- 6. TRAINING PARAMETERS ---
BATCH_SIZE = 4096
LEARNING_RATE = 0.001 # fixed according to reference model
EPOCHS = 200          # according to reference model
PATIENCE = 50         # according to reference model
NODE_ISOLATION_RATE = 0.5 # according to reference model

# [50] means: "For every node, pick at most 50 random neighbors"
# Central Limit Theorem, a sample size of 30-50 is sufficient to estimate the mean of a distribution
# with reasonable accuracy.
NUM_NEIGHBORS = [50]

LABEL_MASK_VALUE = -1