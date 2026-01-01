# ==============================================================================
# python_data_loader_h2.py (H2: Implicit Topology / Dynamic Graph Construction)
#
# PURPOSE:
#   Implements the data engineering pipeline for Hypothesis 2.
#   Constructs DYNAMIC graphs where connectivity is defined by behavioral
#   similarity (k-NN) rather than static metadata.
#
# LOGIC & THESIS ALIGNMENT:
#   1. Implicit Topology (Thesis Sec 3.3.1):
#      - "Edges are defined dynamically based on the Euclidean distance between
#        borrowers in a behavioral feature space."
#      - Features: 'upb_pct_remaining', 'mths_remng', 'current_int_rt'.
#
#   2. Adaptive k-NN (Thesis Eq. 8):
#      - "To accommodate the extreme variance in cluster sizes... an adaptive k
#        threshold is applied: k = base_k + 2*ln(n)."
#      - Justification: Prevents sparse connectivity in large clusters while
#        maintaining local structure in small ones.
#
#   3. Dynamic Evolution (Thesis Sec 3.3):
#      - Unlike H1's static graph, H2 re-computes the k-NN structure at every
#        snapshot ($t$), allowing the topology to evolve as borrower risk profiles change.
#
# INPUTS:
#   - R Dataframes (.rds) specified in config_h2.py
#
# OUTPUTS:
#   - train_graphs.pt: List of dynamic graph sequences [T, Nodes, Features].
#   - test_graphs.pt: List of dynamic graph sequences for testing.
# ==============================================================================

import pyreadr
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict
import pickle
from tqdm import tqdm
import time
import config_h2 as cfg

# --- CONFIGURATION FOR H2 (KNN) ---
# Thesis Sec 3.3.2: "Refined features... to normalize borrower behavior."
BEHAVIORAL_FEATURES = ['upb_pct_remaining', 'mths_remng', 'current_int_rt', 'is_modified']


# --- 1. Data Loading ---
def load_r_data():
    """
        Loads raw R dataframes and enforces strict type safety for H2 features.

        Returns:
            tuple: (final_data, train_targets, test_targets, preproc_params)
        """
    print("📂 Loading R data files...")
    final_data = list(pyreadr.read_r(str(cfg.FINAL_DATA_FILE)).values())[0]
    train_targets = list(pyreadr.read_r(str(cfg.TRAIN_TARGETS_FILE)).values())[0]
    test_targets = list(pyreadr.read_r(str(cfg.TEST_TARGETS_FILE)).values())[0]
    preproc_params = list(pyreadr.read_r(str(cfg.PREPROC_PARAMS_FILE)).values())[0]

    # Target Types to numeric
    print("🛠️  Fixing Target Data Types...")
    if 'Target_Y' in train_targets.columns:
        train_targets['Target_Y'] = pd.to_numeric(train_targets['Target_Y'], errors='coerce').fillna(-1).astype(int)

    if 'Target_Y' in test_targets.columns:
        test_targets['Target_Y'] = pd.to_numeric(test_targets['Target_Y'], errors='coerce').fillna(-1).astype(int)

    if 'Loan_Sequence_Number' in final_data.columns:
        final_data['Loan_Sequence_Number'] = final_data['Loan_Sequence_Number'].astype(str)

        # Ensures it is strictly 0 or 1, even if R passed some NAs or floats
        if 'is_modified' in final_data.columns:
            final_data['is_modified'] = pd.to_numeric(final_data['is_modified'], errors='coerce').fillna(0).astype(int)
        else:
            print(" WARNING: 'is_modified' column not found! Defaulting to 0.")
            final_data['is_modified'] = 0

    # Map delinquency codes
    if 'if_delq_sts' in final_data.columns:
        final_data['if_delq_sts'] = final_data['if_delq_sts'].apply(lambda x: 1 if x >= 1 else 0)

    return final_data, train_targets, test_targets, preproc_params


# --- 2. Feature Scaling ---
class FeatureScaler:
    """
        Applies Min-Max scaling to ensure isotropic distance calculations in KNN.

        Methodology (Thesis Sec 3.3.1):
            - "Since Euclidean distance is sensitive to feature magnitudes, all
              behavioral features are strictly normalized to [0,1]."
        """
    def __init__(self, preproc_params: pd.DataFrame):
        self.params = preproc_params.set_index('Feature')

    def transform(self, df: pd.DataFrame, numeric_features: List[str]) -> pd.DataFrame:
        """
                Scales numeric features to [0, 1] range using pre-computed R statistics.

                Args:
                    df (pd.DataFrame): Raw dataframe.
                    numeric_features (List[str]): List of columns to scale.

                Returns:
                    pd.DataFrame: Scaled dataframe.
                """
        df_scaled = df.copy()
        for feat in numeric_features:
            if feat not in self.params.index:
                continue

            min_val = self.params.loc[feat, 'Min']
            max_val = self.params.loc[feat, 'Max']

            if max_val - min_val < 1e-8:
                df_scaled[feat] = 0.0
            else:
                df_scaled[feat] = (df[feat] - min_val) / (max_val - min_val)
                df_scaled[feat] = df_scaled[feat].clip(0.0, 1.0)

        return df_scaled


# --- 3. Graph Construction (Behavioral Adaptive KNN) ---
def build_adaptive_knn_edges(
        snapshot_df_active: pd.DataFrame,
        network_key: str,
        base_k: int,
        master_node_map: Dict[str, int],
        feature_cols: List[str]
) -> torch.LongTensor:
    """
        Constructs edges using an adaptive k-NN strategy based on behavioral similarity.
        Scales k logarithmically with cluster size to balance graph density.

        Args:
            snapshot_df_active (pd.DataFrame): Active loans in the current snapshot.
            network_key (str): Partition key (e.g., 'Geo_Key').
            base_k (int): Minimum neighbors to connect.
            master_node_map (Dict): Mapping from Loan ID to global node index.
            feature_cols (List[str]): Features used for distance calculation.

        Returns:
            torch.LongTensor: Edge index tensor of shape (2, Num_Edges).
        """
    edges = []

    valid_df = snapshot_df_active[
        snapshot_df_active['Loan_Sequence_Number'].isin(master_node_map.keys())
    ].copy()

    valid_df['master_idx'] = valid_df['Loan_Sequence_Number'].map(master_node_map)

    for group_name, group_df in valid_df.groupby(network_key):
        n = len(group_df)
        if n < 2: continue

        # Adaptive K Logic: k ~ log(n) for large groups
        if n < 10:
            k_adaptive = n - 1
        elif n < 100:
            k_adaptive = max(base_k, int(np.sqrt(n)))
        else:
            k_adaptive = max(base_k, int(base_k + np.log(n) * 2))

        k_actual = min(k_adaptive, n - 1)
        if k_actual <= 0: continue

        try:
            group_features = group_df[feature_cols].values
        except KeyError as e:
            print(f" CRITICAL ERROR: Missing column for KNN. {e}")
            raise e

        group_indices = group_df['master_idx'].values

        # --- SAFE KNN LOGIC ---
        target_k = k_actual + 1  # k neighbors + self

        # Case A: Small Group (Connect All)
        if target_k >= n:
            indices = np.arange(n)
            for i in range(n):
                for neighbor_idx in indices:
                    if i != neighbor_idx:
                        edges.append([group_indices[i], group_indices[neighbor_idx]])

        # Case B: Large Group (Nearest Neighbors via Euclidean Distance)
        else:
            for i in range(n):
                diff = group_features - group_features[i]
                dists = np.linalg.norm(diff, axis=1)

                # argpartition guarantees the smallest k are in the first k positions
                # We slice [:target_k] to get them.
                nearest_indices = np.argpartition(dists, target_k)[:target_k]

                for neighbor_idx in nearest_indices:
                    if i != neighbor_idx:
                        edges.append([group_indices[i], group_indices[neighbor_idx]])

    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def create_snapshot_graph(
        snapshot_df_active: pd.DataFrame,
        snapshot_date: int,
        scaler: FeatureScaler,
        master_node_map: Dict[str, int]
) -> Data:
    """
        Constructs a PyG Data object for a single time snapshot (t).
        Implements a Double-Layer graph (Master + Shadow nodes).

        Args:
            snapshot_df_active (pd.DataFrame): Active loans at time t.
            snapshot_date (int): The reporting period (YYYYMM).
            scaler (FeatureScaler): Scaler instance for feature normalization.
            master_node_map (Dict): Global ID mapping.

        Returns:
            Data: PyTorch Geometric Data object containing X, Y, and EdgeIndex.
        """
    num_master_nodes = len(master_node_map)
    num_features = len(cfg.NODE_FEATURES)

    x_single = torch.zeros((num_master_nodes, num_features), dtype=torch.float)
    y_single = torch.full((num_master_nodes,), cfg.LABEL_MASK_VALUE, dtype=torch.long)

    # Prepare features
    snapshot_scaled = scaler.transform(snapshot_df_active, cfg.NUMERIC_FEATURES)

    x_num = torch.tensor(snapshot_scaled[cfg.NUMERIC_FEATURES].values, dtype=torch.float)
    x_bin = torch.tensor(snapshot_df_active[cfg.BINARY_FEATURES].values, dtype=torch.float)
    x_active = torch.cat([x_num, x_bin], dim=1)

    # Prepare labels
    raw_targets = pd.to_numeric(snapshot_df_active['Target_Y'], errors='coerce').fillna(-1).astype(int).values
    y_active = torch.tensor(raw_targets, dtype=torch.long)

    # Map active nodes
    active_indices = []
    valid_mask = []
    for i, loan_id in enumerate(snapshot_df_active['Loan_Sequence_Number']):
        if loan_id in master_node_map:
            active_indices.append(master_node_map[loan_id])
            valid_mask.append(i)

    if not active_indices:
        return Data(
            x=torch.zeros((num_master_nodes * 2, num_features)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            y=torch.full((num_master_nodes * 2,), cfg.LABEL_MASK_VALUE),
            snapshot_date=snapshot_date,
            num_nodes=num_master_nodes * 2,
            n_id=torch.arange(num_master_nodes * 2, dtype=torch.long) # ✅ Fix 1
        )

    active_idx_tensor = torch.tensor(active_indices, dtype=torch.long)
    x_single[active_idx_tensor] = x_active[valid_mask]
    y_single[active_idx_tensor] = y_active[valid_mask]

    # Double-layer construction (Master + Shadow)
    x_multi = torch.cat([x_single, x_single], dim=0)
    y_multi = torch.cat([y_single, y_single], dim=0)

    # --- Edge Construction (H2 Implicit) ---
    knn_input_df = snapshot_df_active[['Loan_Sequence_Number', 'Geo_Key', 'Lender_Key']].copy()

    for feat in BEHAVIORAL_FEATURES:
        if feat in cfg.NUMERIC_FEATURES:
            knn_input_df[feat] = snapshot_scaled[feat].values
        elif feat in cfg.BINARY_FEATURES:
            knn_input_df[feat] = snapshot_df_active[feat].values

    # 1. Geographic Neighbors
    edge_index_geo = build_adaptive_knn_edges(
        knn_input_df, 'Geo_Key', cfg.BASE_K_GEO, master_node_map, BEHAVIORAL_FEATURES
    )

    # 2. Lender Neighbors
    edge_index_lender_base = build_adaptive_knn_edges(
        knn_input_df, 'Lender_Key', cfg.BASE_K_LENDER, master_node_map, BEHAVIORAL_FEATURES
    )
    edge_index_lender = edge_index_lender_base + num_master_nodes

    # 3. Inter-layer (Vertical) Edges
    inter_src = torch.arange(num_master_nodes, dtype=torch.long)
    inter_dst = inter_src + num_master_nodes
    inter_edges = torch.stack([
        torch.cat([inter_src, inter_dst]),
        torch.cat([inter_dst, inter_src])
    ], dim=0)

    edge_index = torch.cat([edge_index_geo, edge_index_lender, inter_edges], dim=1)

    return Data(
        x=x_multi,
        edge_index=edge_index,
        y=y_multi,
        snapshot_date=snapshot_date,
        num_nodes=x_multi.shape[0],
        # ✅ Fix 2: Explicit Global Node IDs
        n_id=torch.arange(x_multi.shape[0], dtype=torch.long)
    )

# --- 4. SEQUENCE GENERATION ---
def create_sliding_windows(final_data, targets, scaler, split_name, master_node_map):
    """
        Generates temporal graph sequences (sliding windows) for LSTM input.
        Filters out historically defaulted loans to ensure training on active loans only.

        Args:
            final_data (pd.DataFrame): R-processed dataset.
            targets (pd.DataFrame): Target labels.
            scaler (FeatureScaler): Scaler instance.
            split_name (str): 'TRAIN' or 'TEST'.
            master_node_map (Dict): Global ID mapping.

        Returns:
            List[List[Data]]: List of graph sequences.
        """
    print(f"\n🪟 Processing {split_name} Windows (H2 Implicit + Target Filter)...")

    # Ensure correct types
    final_data['Monthly_Reporting_Period'] = final_data['Monthly_Reporting_Period'].astype(int)

    # Pre-merge Target_Y for fast filtering
    if 'Target_Y' not in final_data.columns:
        temp_targets = targets[['Loan_Sequence_Number', 'Snapshot_Date', 'Target_Y']].copy()
        temp_targets.rename(columns={'Snapshot_Date': 'Monthly_Reporting_Period'}, inplace=True)
        final_data = final_data.merge(temp_targets, on=['Loan_Sequence_Number', 'Monthly_Reporting_Period'], how='left')
        final_data['Target_Y'] = final_data['Target_Y'].fillna(0).astype(int)

    target_dates = sorted(targets['Snapshot_Date'].unique().astype(int))
    data_by_month = {k: v for k, v in final_data.groupby('Monthly_Reporting_Period')}
    target_by_month = {k: v for k, v in targets.groupby('Snapshot_Date')}

    graphs = []

    for target_date in tqdm(target_dates, desc=f"{split_name}"):
        all_months = sorted(final_data['Monthly_Reporting_Period'].unique())
        try:
            t_idx = all_months.index(target_date)
        except ValueError:
            continue

        if t_idx < cfg.SEQUENCE_LENGTH - 1:
            continue

        window_months = all_months[t_idx - cfg.SEQUENCE_LENGTH + 1: t_idx + 1]
        window_start = window_months[0]

        if window_start not in data_by_month: continue

        start_df = data_by_month[window_start]

        # Filter: Exclude loans that already defaulted (-1)
        valid_loans = start_df[start_df['Target_Y'] != -1]['Loan_Sequence_Number']
        active_loan_ids = set(valid_loans)
        # ---------------------------------------------

        sub_sequence = []
        for m in window_months:
            if m not in data_by_month: continue
            df = data_by_month[m]

            active_df = df[
                df['Loan_Sequence_Number'].isin(active_loan_ids) &
                df['Loan_Sequence_Number'].isin(master_node_map)
                ].copy()

            if active_df.empty: continue

            graph = create_snapshot_graph(active_df, m, scaler, master_node_map)
            sub_sequence.append(graph)

        if len(sub_sequence) == cfg.SEQUENCE_LENGTH:
            # Assign Final Target from target file
            if target_date in target_by_month:
                t_df = target_by_month[target_date]
                valid_t = t_df[t_df['Loan_Sequence_Number'].isin(master_node_map)]

                y_final = torch.full((graph.num_nodes,), cfg.LABEL_MASK_VALUE, dtype=torch.long)

                if not valid_t.empty:
                    indices = valid_t['Loan_Sequence_Number'].map(master_node_map).values
                    vals = pd.to_numeric(valid_t['Target_Y'], errors='coerce').fillna(-1).astype(int).values

                    y_final[indices] = torch.tensor(vals, dtype=torch.long)
                    y_final[indices + len(master_node_map)] = torch.tensor(vals, dtype=torch.long)

                sub_sequence[-1].y = y_final

            graphs.append(sub_sequence)

    return graphs


def main(override_save_dir=None):
    total_start = time.time()

    if override_save_dir:
        save_dir = override_save_dir
        print(f"  DEBUG MODE: Saving outputs to {save_dir}")
    else:
        save_dir = cfg.SAVE_DIR

    save_dir.mkdir(exist_ok=True, parents=True)

    final_data, train_targets, test_targets, preproc_params = load_r_data()
    scaler = FeatureScaler(preproc_params)

    print("\n  Building Master Node Map...")
    all_loans = final_data['Loan_Sequence_Number'].unique()
    all_loans_sorted = sorted(all_loans.astype(str))
    master_map = {loan: i for i, loan in enumerate(all_loans_sorted)}
    print(f"   Total Unique Loans: {len(master_map)}")

    train_graphs = create_sliding_windows(final_data, train_targets, scaler, "TRAIN", master_map)
    test_graphs = create_sliding_windows(final_data, test_targets, scaler, "TEST", master_map)

    print("\n Saving graphs...")
    torch.save(train_graphs, save_dir / "train_graphs.pt")
    torch.save(test_graphs, save_dir / "test_graphs.pt")

    with open(save_dir / "graph_metadata.pkl", 'wb') as f:
        pickle.dump({
            'num_features': len(cfg.NODE_FEATURES),
            'train_snapshots': len(train_graphs),
            'test_snapshots': len(test_graphs)
        }, f)

    print(f"\n H2 Data Prep (Behavioral KNN) Complete. Runtime: {(time.time() - total_start) / 60:.2f} min")


if __name__ == "__main__":
    main()
