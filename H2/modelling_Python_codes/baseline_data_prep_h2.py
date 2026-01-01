# ==============================================================================
# baseline_data_prep_h2.py (H2: Implicit Topology / Feature-Based)
#
# PURPOSE:
#   Prepares aligned datasets for the H2 Baseline models (XGBoost, LR, Static GNNs)
#   using the "Implicit" feature set and "Behavioral" topology definitions.
#   Ensures strict methodological alignment with H1 regarding temporal splits
#   and validation protocols to enable fair hypothesis testing.
#
# LOGIC & THESIS ALIGNMENT:
#   1. Feature Set (Thesis Sec 3.3.2):
#      - Includes H2-specific engineered features (e.g., 'clean_loan_age',
#        'upb_pct_remaining') designed for the KNN similarity metric.
#
#   2. Temporal Averaging (Thesis Sec 4.3.1):
#      - "Node features within each window are aggregated via temporal averaging,
#        resulting in a single static feature matrix."
#      - This collapses the 6-month sequence into a static snapshot for baselines.
#
#   3. Validation Protocol (Thesis Sec 4.3.1):
#      - Enforces the same "Chronological Split" as H1:
#        Train (Windows 1-12) vs. Validation (Window 13).
#
# INPUTS:
#   - H2 Processed Features: train_graphs.pt, test_graphs.pt (Tensor objects)
#   - Raw Metadata: final_data_base_with_targets.rds
#   - Configuration: config_h2.py
#
# OUTPUTS:
# #   - train_graphs.pt: List of dynamic graph sequences for training.
# #   - test_graphs.pt: List of dynamic graph sequences for testing.
# #   - graph_metadata.pkl: Summary statistics.
# # ==============================================================================

import torch
import numpy as np
import pickle
import gc
import sys
from tqdm import tqdm
import config_h2 as cfg
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from python_data_loader_h2 import load_r_data


def process_windows(file_path, desc, max_valid_id):
    """
        Loads H2 tensor data and applies Temporal Averaging to create static snapshots.

        Methodology (Thesis Sec 4.3.1):
            - "To construct a fair static comparison, the 18-month behavioral history
              is compressed into a single snapshot... via temporal averaging."
            - This neutralizes the temporal variance, forcing the static baselines
              to rely solely on the implicit topological signal.

        Args:
            file_path (Path): Path to the .pt file (train or test).
            desc (str): Description for the progress bar.
            max_valid_id (int): Maximum node ID to ensure alignment with H1 boundaries.

        Returns:
            list: A list of dictionaries [{'X': np.array, 'y': np.array}, ...].
        """
    if not file_path.exists():
        print(f" Error: {file_path} not found.")
        return []

    print(f" Loading {file_path.name}...")
    try:
        # Safe load to handle PyG Data attributes
        with torch.serialization.safe_globals([DataTensorAttr, DataEdgeAttr]):
            data_list = torch.load(file_path, weights_only=False)
    except:
        data_list = torch.load(file_path)

    if not data_list:
        print(" Data list is empty.")
        return []

    windows = []

    # --- 1. AUTO-DETECT INPUT FORMAT ---
    # The loader might return different structures depending on the stage.
    first_item = data_list[0]

    # Mode 1: Pre-windowed sequences (List of Lists of Data)
    if isinstance(first_item, list):
        print(f"    Data is pre-windowed sequences (List[List[Data]]). Processing {len(data_list)} windows...")
        mode = "list_of_lists"
        loop_range = range(len(data_list))

    elif hasattr(first_item, 'x') and first_item.x.dim() == 3:
        # Mode 2: Pre-stacked tensors [Time, Nodes, Features]
        print(
            f"    Data is pre-stacked tensors (Shape: {first_item.x.dim()}). Processing {len(data_list)} windows...")
        mode = "pre_stacked"
        loop_range = range(len(data_list))

    elif hasattr(first_item, 'x') and first_item.x.dim() == 2:
        # Mode 3: Flat snapshots [Nodes, Features] -> Needs rolling window logic
        # Ref: Thesis Section 2.3 "Rolling Window Dynamics"
        WINDOW_SIZE = 6
        num_windows = len(data_list) - WINDOW_SIZE + 1
        if num_windows < 1:
            print(" Data too short for windowing.")
            return []
        print(f"    Data is flat snapshots. Creating {num_windows} rolling windows...")
        mode = "flat_snapshots"
        loop_range = range(num_windows)
    else:
        print(" Unknown data format.")
        return []

    for i in tqdm(loop_range, desc=desc):
        # --- 2. EXTRACTION LOGIC ---
        if mode == "list_of_lists":
            # Input: List of Data objects
            window_snaps = data_list[i]
            try:
                x_stack = torch.stack([s.x for s in window_snaps], dim=0)
            except RuntimeError:
                # Skip if sizes mismatch (rare edge case)
                continue

            last_snap = window_snaps[-1]
            y = last_snap.y
            n_id = last_snap.n_id if hasattr(last_snap, 'n_id') else None

        elif mode == "pre_stacked":
            # Input: Single Data object with 3D x
            window_data = data_list[i]
            x_stack = window_data.x
            y = window_data.y
            n_id = window_data.n_id if hasattr(window_data, 'n_id') else None

        elif mode == "flat_snapshots":
            # Input: Flat list, need to slice
            window_snaps = data_list[i: i + 6]
            try:
                x_stack = torch.stack([s.x for s in window_snaps], dim=0)
            except RuntimeError:
                continue

            last_snap = window_snaps[-1]
            y = last_snap.y
            n_id = last_snap.n_id if hasattr(last_snap, 'n_id') else None

        # --- 3. TEMPORAL AGGREGATION ---
        # "Node features within each window are aggregated via temporal averaging"
        # Input [6, N, F] -> Output [N, F]
        x_mean = x_stack.mean(dim=0)

        # Validation checks
        if n_id is None: continue

        # --- 4. FILTERING & ALIGNMENT ---
        # Filter valid labels (-1 is mask) and ensure within H1 Node Boundary
        global_ids = n_id.numpy()
        mask = (y != -1) & (n_id < max_valid_id)

        if mask.sum() == 0: continue

        # Strict Mask Refinement (Safety)
        # valid_ids = global_ids[mask]

        # Store aligned data
        windows.append({
            'X': x_mean[mask].numpy(),   # Static Averaged Features
            'y': y[mask].numpy()         # Targets
        })

    del data_list
    gc.collect()
    return windows


def main():
    print(" Starting H2 Baseline Data Prep (Implicit Topology)...")
    cfg.SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Establish Ground Truth Boundary
    # We must ensure H2 models predict on the exact same set of loans as H1.
    print("  Loading Raw Data Map...")
    try:
        final_data, _, _, _ = load_r_data()
        if 'Loan_Sequence_Number' not in final_data.columns:
            sys.exit(" Critical: Loan_Sequence_Number missing.")

        # Boundary check
        all_loans = final_data['Loan_Sequence_Number'].unique()
        max_valid_id = len(all_loans)
        print(f"   Boundary: {max_valid_id} Unique Loans")
        del final_data, all_loans
        gc.collect()
    except Exception as e:
        sys.exit(f" Error: {e}")

    # 2. Process Windows
    # Loads the H2 GNN tensors and converts them to static snapshots for baselines
    train_windows = process_windows(cfg.SAVE_DIR / "train_graphs.pt", "Train", max_valid_id)
    test_windows = process_windows(cfg.SAVE_DIR / "test_graphs.pt", "Test", max_valid_id)

    if not train_windows or not test_windows:
        sys.exit(" Critical: Processed data is empty.")

    # 3. Temporal Split Strategy (Thesis Sec 4.3.1)
    # Matches H1 exactly: Train=Windows 1-12, Val=Window 13.
    val_window = train_windows[-1]
    train_windows_final = train_windows[:-1]
    test_window = test_windows[0]  # Test is usually 1 window in this setup

    # 4. Concatenate for Tabular Models (XGBoost, LR)
    # Flattens the time-series windows into a single matrix.
    X_tr = np.concatenate([w['X'] for w in train_windows_final], axis=0)
    y_tr = np.concatenate([w['y'] for w in train_windows_final], axis=0)
    X_val, y_val = val_window['X'], val_window['y']
    X_test, y_test = test_window['X'], test_window['y']

    # 5. Save Data Dictionary
    data_dict = {
        # Flat Arrays -> Used by XGBoost / Logistic Regression
        'X_train': X_tr, 'y_train': y_tr,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,

        # Window Lists -> Used by Static GNNs for stochastic sampling
        'train_windows': train_windows_final,
        'val_window': val_window,
        'test_window': test_window
    }

    save_path = cfg.SAVE_DIR / "baseline_data_h2_aligned.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f" H2 Prep Complete. Saved to {save_path}")
    print(f"   Train: {len(X_tr):,} samples ({len(train_windows_final)} Windows)")
    print(f"   Val:   {len(X_val):,} samples (1 Window)")


if __name__ == "__main__":
    main()