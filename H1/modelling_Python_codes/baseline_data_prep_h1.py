# ==============================================================================
# baseline_data_prep_h1.py (H1: Constraint-Aware Replication)
#
# PURPOSE:
#   Prepares aligned datasets for the H1 Baseline models (Logistic Regression,
#   XGBoost, Static GNNs) as defined in Thesis Section 4.3. This script ensures
#   fair comparison by enforcing the exact same temporal splits and topology
#   definitions used in the dynamic H1 model.
#
# LOGIC & THESIS ALIGNMENT:
#   1. Topology Definition (Thesis Sec 4.3.1.1):
#      - Implements "Explicit Categorical Topology" logic.
#      - Geographic edges: Defined by first 2 digits of Zip Code.
#      - Lender edges: Defined by Lender ID.
#   2. Temporal Averaging (Thesis Sec 4.3.1):
#      - Compresses the 6-month behavioral history (Lookback T=6) into a single
#        static snapshot via mean pooling. This allows static models (LR/XGB)
#        to be benchmarked against the dynamic LSTM-GNN.
#   3. Validation Protocol (Thesis Sec 4.3.1):
#      - Enforces "Chronological Split" to prevent look-ahead bias.
#      - Train: Windows 1-12.
#      - Val: Window 13 (Future unseen data).
#
# INPUTS:
#   - Processed Features: train_features.pt, test_features.pt (Tensor objects)
#   - Raw Metadata: final_features_h1.rds
#   - Configuration: config_h1.py
#
# OUTPUTS:
#   - baseline_data_h1_aligned.pkl: Dictionary containing:
#       - 'X_train/val/test': Flattened arrays for Tabular Models (XGB/LR).
#       - 'train/val/test_windows': List structures for Static GNNs.
#       - 'geo/lend_lookup': Encoders for constructing adjacency matrices.
# ==============================================================================


import torch
import numpy as np
import pickle
import sys
from tqdm import tqdm
import config_h1 as cfg
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from sklearn.preprocessing import LabelEncoder
from python_data_loader_h1 import load_and_scale_data


def process_windows(file_path, desc, loan_id_to_keys):
    """
        Loads temporal graph data and aggregates node features to create static windows.

        Methodology:
            - Implements "Temporal Averaging" (Thesis Sec 4.3.1)
            - Converts each rolling/pre-windowed sequence [Time, Nodes, Features]
              into a single static snapshot [Nodes, Features].
            - Filters invalid or masked nodes and aligns node indices to the global topology.

        Args:
            file_path (Path): Path to serialized PyG dataset (.pt)
            desc (str): tqdm description for progress
            loan_id_to_keys (dict): Maps global node IDs to geo/lender keys

        Returns:
            list of dict: Each dict contains
                'X' (np.ndarray): Node features [Nodes, Features]
                'y' (np.ndarray): Node labels [Nodes]
                'geo' (np.ndarray): Node geo IDs
                'lend' (np.ndarray): Node lender IDs
        """
    if not file_path.exists():
        print(f" Error: {file_path} not found.")
        return None

    # Safe load to handle PyG Data attributes
    print(f" Loading {file_path.name}...")
    try:
        with torch.serialization.safe_globals([DataTensorAttr, DataEdgeAttr]):
            data_list = torch.load(file_path, weights_only=False)
    except:
        data_list = torch.load(file_path)

    windows = []
    max_geo_id = len(loan_id_to_keys['geo'])

    # --- Detect Input Format (Pre-windowed vs Snapshots) ---
    first_x = data_list[0].x
    is_pre_windowed = (first_x.dim() == 3) # Shape: [Time, Nodes, Features]

    if is_pre_windowed:
        print(f"   Data is pre-windowed (Shape: {first_x.shape}). Processing {len(data_list)} windows...")
        loop_range = range(len(data_list))
    else:
        # Standard rolling window logic for snapshots (Thesis Section 2.3 "Rolling Window Dynamics")
        WINDOW_SIZE = 6
        num_windows = len(data_list) - WINDOW_SIZE + 1
        print(f"   Data is snapshots. Creating {num_windows} rolling windows...")
        loop_range = range(num_windows)

    for i in tqdm(loop_range, desc=desc):
        # 1. Get Stacked Data [6, N, F] & Labels
        if is_pre_windowed:
            window_data = data_list[i]
            x_stack = window_data.x
            y = window_data.y
            n_id = window_data.n_id if hasattr(window_data, 'n_id') else None
        else:
            window_snaps = data_list[i: i + 6]
            try:
                x_stack = torch.stack([s.x for s in window_snaps], dim=0)
            except RuntimeError:
                continue

            last_snap = window_snaps[-1]
            y = last_snap.y
            n_id = last_snap.n_id if hasattr(last_snap, 'n_id') else None

        # 2. Temporal Aggregation [6, N, F] -> [N, F]: Implements "Temporal Averaging" (Sec 4.3.1).
        # We collapse the time dimension to create a static input for baselines.
        x_mean = x_stack.mean(dim=0)

        # 3. Validation & Filtering
        if n_id is None: continue
        global_ids = n_id.numpy()

        # Mask logic: Remove masked loans (-1) and ensure IDs are within topology bounds
        mask = (y != -1) & (n_id < max_geo_id)
        if mask.sum() == 0: continue

        # Filter valid IDs through the pre-computed topology lookups
        valid_ids = global_ids[mask]
        valid_mask = valid_ids < max_geo_id
        if not np.any(valid_mask): continue
        valid_ids = valid_ids[valid_mask]

        # Apply final mask to tensors
        final_mask = torch.isin(n_id, torch.tensor(valid_ids)) & mask

        # 4. Store Aligned Window
        windows.append({
            'X': x_mean[final_mask].numpy(),
            'y': y[final_mask].numpy(),
            'geo': loan_id_to_keys['geo'][valid_ids],
            'lend': loan_id_to_keys['lender'][valid_ids]
        })

    print(f"   Created {len(windows)} windows")
    return windows


def main():
    print(" Starting H1 Baseline Data Prep (Explicit Topology)...")
    cfg.SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Raw Data
    # Ensures baselines use the exact same scaling as the GNN (Sec 3.1.1)
    final_data, _, _ = load_and_scale_data()

    if 'Loan_Sequence_Number' not in final_data.columns:
        print(" Critical: Loan_Sequence_Number missing.")
        sys.exit(1)

    # 2. Define Explicit Topology (Thesis Section 4.3.1.1)
    print("  Mapping Topology Keys...")

    # Nodes are connected if they share the same first two digits of their ZIP code.
    final_data['Geo_Key'] = final_data['Geo_Key'].astype(str).str[:2]

    # Create Unique Lookups: Map every Loan ID to its Geo/Lender key
    df_uniq = final_data.drop_duplicates('Loan_Sequence_Number').sort_values('Loan_Sequence_Number')

    le_geo = LabelEncoder()
    le_lend = LabelEncoder()

    geo_encoded = le_geo.fit_transform(df_uniq['Geo_Key'])
    lend_encoded = le_lend.fit_transform(df_uniq['Lender_Key'].astype(str))

    key_lookup = {
        'geo': geo_encoded,
        'lender': lend_encoded
    }

    print(f"   Topology Boundary: {len(le_geo.classes_)} Areas, {len(le_lend.classes_)} Lenders")

    # 3. Process Windows
    # Loads the GNN tensors and converts them to static snapshots for baselines
    train_windows = process_windows(cfg.GRAPH_SAVE_DIR / "train_features.pt", "Train Prep", key_lookup)
    test_windows = process_windows(cfg.GRAPH_SAVE_DIR / "test_features.pt", "Test Prep", key_lookup)

    if not train_windows or not test_windows:
        print(" CRITICAL: Data empty.")
        sys.exit(1)

    # 4. Temporal Split Strategy (Thesis Section 4.3.1)
    # Train: Windows 1-12
    # Val:   Window 13 (Future Validation)
    val_window = train_windows[-1]
    train_windows_final = train_windows[:-1]
    test_window = test_windows[0]

    # 5. Concatenate for Tabular Models (LR, XGBoost)
    # Flattens the list of windows into a single large matrix.
    X_tr = np.concatenate([w['X'] for w in train_windows_final], axis=0)
    y_tr = np.concatenate([w['y'] for w in train_windows_final], axis=0)

    # Val/Test are single windows, so we just extract the arrays
    X_val, y_val = val_window['X'], val_window['y']

    print(f"\n Data Summary:")
    print(f"   Train: {len(X_tr):,} samples (12 Windows)")
    print(f"   Val:   {len(X_val):,} samples (1 Window)")
    print(f"   Test:  {len(test_window['X']):,} samples")

    # 6. Save Data Dictionary
    data_dict = {
        # Flat Arrays (For Non-GNN Models)
        'X_train': X_tr, 'y_train': y_tr,
        'X_val': X_val, 'y_val': y_val,
        'X_test': test_window['X'], 'y_test': test_window['y'],

        # Window Lists (For GNN Models)
        'train_windows': train_windows_final,
        'val_window': val_window,
        'test_window': test_window,

        # Topology Encoders -> Used to build Adjacency Matrices for GNNs
        'geo_lookup': key_lookup['geo'],
        'lend_lookup': key_lookup['lender']
    }

    save_path = cfg.DATA_DIR / "baseline_data_h1_aligned.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f" H1 Prep Complete. Saved to {save_path}")


if __name__ == "__main__":
    main()