# ==============================================================================
# config_2.py (H2)
# ==============================================================================

import torch
from pathlib import Path

# --- 1. System Paths ---

DATA_DIR = Path("...H2\\data_processed")
SAVE_DIR = Path("...H2\\graphs_processed")
GRAPH_DIR = Path("...H2\\graphs_processed")
CHECKPOINT_DIR = Path("...H2\\checkpoints")
LOG_DIR = Path("...H2\\logs")

# --- 2. Data & Graph Parameters  ---
# R-Script file names
FINAL_DATA_FILE = DATA_DIR / "final_data_base_with_targets.rds"
TRAIN_TARGETS_FILE = DATA_DIR / "train_targets.rds"
TEST_TARGETS_FILE = DATA_DIR / "test_targets.rds"
PREPROC_PARAMS_FILE = DATA_DIR / "preproc_params.rds"

# --- IMPLICIT TOPOLOGY HYPERPARAMETERS (Thesis Sec 3.3.1) ---
# "To accommodate extreme variance... an adaptive k threshold is applied."
# Formula: k = BASE_K + 2 * ln(Cluster_Size)
BASE_K_GEO = 6      # Base k was calculated based on sqr.root of smallest group size divided by 2
BASE_K_LENDER = 7

# --- FEATURE DEFINITIONS (Thesis Sec 3.3.2) ---
# "Refined features... to normalize borrower behavior."
# NOTE: 'clean_loan_age' and 'upb_pct_remaining' are critical for
# measuring behavioral similarity in the k-NN step.
NUMERIC_FEATURES = [
    'fico', 'ltv', 'dti', 'mi_pct', 'cnt_units', 'cnt_borr',
    'upb_pct_remaining', 'current_int_rt', 'mths_remng', 'clean_loan_age'
]
BINARY_FEATURES = [
    'if_fthb', 'if_prim_res', 'if_corr', 'if_sf',
    'if_purc', 'if_sc', 'is_modified', 'if_delq_sts'
]
NODE_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES
NETWORK_KEYS = ['Geo_Key', 'Lender_Key']

BATCH_SIZE = 2048          # because of accumulation steps 2048 * 2 = 4096 effective batch size
NEIGHBOR_HOPS = 1          # borrower + their direct neighbors
ACCUMULATION_STEPS = 2     # update weights after 2 batches (Simulated Batch Size = 2048 * 2)
NODE_ISOLATION_RATE = 0.5  # Graph Dropout Rate

# --- 3. Model Architecture Parameters ---
NUM_FEATURES = len(NODE_FEATURES) # Auto-calculated
NUM_CLASSES = 1                # Binary Classification (Default vs Non-Default)
GAT_HIDDEN = 32                # Latent dimension for Graph Attention
GAT_HEADS = 2                  # Multi-head attention count
GAT_OUT = 20                   # Dimension fed into LSTM (matches LSTM_HIDDEN)
LSTM_HIDDEN = 20               # Inferred from Decoder input (Fig. 5)
LSTM_LAYERS = 2                # Stacked LSTM layers
MLP_HIDDEN = 10                # "Dense Layer (20, 10)" -> "Dense Layer (10, 1)"
DROPOUT = 0.5                  # Regularization rate


# --- 4. Training Protocol (Thesis Table 3) ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.001          # Starting learning rate for Scheduler
WEIGHT_DECAY = 1e-5            # L2 Regularization
NUM_EPOCHS = 200               # Max Epochs
SEQUENCE_LENGTH = 6            # "Strictly enforces a fixed window of tau=6"
LABEL_MASK_VALUE = -1          # Mask for padding/inactive nodes

# --- 5. Early Stopping Parameters ---
PATIENCE = 50                  # Stop if no improvement for 50 epochs
MIN_DELTA = 0.001              # Minimum improvement threshold

# --- 6. R code parameters
TRAIN_SPLIT_DATE = 201306     # Cutoff for Train/Test split