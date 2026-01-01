# ==============================================================================
# python_data_loader_h1.py (H1)
#
# PURPOSE:
#   Implements the core data engineering pipeline for Hypothesis 1.
#   It constructs the "Static Supra-Graph" and generates the temporal rolling
#   windows required for the DYMGNN architecture.
#
# LOGIC & THESIS ALIGNMENT:
#   1. Burn-in Recovery (Thesis Sec 3.1.1):
#      - "Utilizing this 'pre-history' (starting jun 2011)... the full 18 training windows
#        were recovered without data censoring."
#      - Ensures no training data is lost to LSTM initialization.
#
#   2. Static Supra-Graph (Thesis Sec 3.2.1):
#      - "A Static Supra-Graph (Disjoint Union) approach was utilized...
#        pre-loaded into a unified graph structure."
#      - Prevents the I/O overhead of reconstructing graphs at every epoch.
#
#   3. Hybrid Connectivity (Thesis Sec 3.2.1):
#      - "Small borrower groups retained full connectivity, while larger groups
#        were truncated using stochastic neighbor sampling."
#      - Implements the [100, 200] caps derived from topology analysis.
#
# INPUTS:
#   - final_features_h1.rds (Feature Matrix)
#   - train_targets.rds (Training Labels)
#   - test_targets.rds (Testing Labels)
#   - preproc_params.rds (Scaling Parameters)
#
# OUTPUTS:
#   - train_features.pt: List of PyG Data objects (Temporal Windows)
#   - static_edge_index.pt: The explicit adjacency matrix (Geo + Lender)
# ==============================================================================

import os
import torch
import numpy as np
import pandas as pd
import pyreadr
from torch_geometric.data import Data
from tqdm import tqdm
import config_h1 as cfg
import gc
from torch_geometric.utils import to_undirected


def load_and_scale_data():
    """
        Loads raw R dataframes and applies feature scaling/engineering.

        Methodology (Thesis Sec 3.1.1):
            - "Min-max normalization based on statistics derived from the training set"
              to prevent information leakage.
            - "Explicitly maps 'RA' (Foreclosure) to a delinquency severity," ensuring
              terminal losses are captured.

        Returns:
            tuple: (final_data, train_targets, test_targets)
        """

    print(" Loading R data files...")

    # 1. Load Feature Matrix
    try:
        final_data = pyreadr.read_r(str(cfg.FINAL_DATA_FILE))[None]
    except KeyError:
        df_dict = pyreadr.read_r(str(cfg.FINAL_DATA_FILE))
        final_data = df_dict[list(df_dict.keys())[0]]

    # 2. Load Targets
    train_targets = pyreadr.read_r(str(cfg.TRAIN_TARGETS_FILE))[None]
    test_targets = pyreadr.read_r(str(cfg.TEST_TARGETS_FILE))[None]

    # 3. Load Scaling Params
    preproc = pyreadr.read_r(str(cfg.PREPROC_PARAMS_FILE))[None].set_index('Feature')

    final_data['Loan_Sequence_Number'] = final_data['Loan_Sequence_Number'].astype(str)

    # --- FEATURE ENGINEERING: BINARY FLAGS ---
    # in data prep R script, we created 'delq_numeric' column for target derivation
    print("  Enforcing Binary 'if_delq_sts'...")

    if 'delq_numeric' not in final_data.columns:
        raise ValueError("CRITICAL: 'delq_numeric' column missing! Check R script 002.")

    # Map > 0 (1=30 days, 2=60 days, 3=Repo/Foreclosure) to 1 (Delinquent)
    # Thesis Sec 3.1.1: "This study explicitly maps 'RA' to a delinquency severity...
    # ensuring these events are correctly classified as defaults."
    final_data['if_delq_sts'] = (final_data['delq_numeric'] > 0).astype(int)

    # --- SCALING (Numeric Only) ---
    print("  Scaling numeric features...")
    for feat in cfg.NUMERIC_FEATURES:
        if feat not in final_data.columns:
            continue

        if feat in preproc.index:
            min_v = preproc.loc[feat, 'Min']
            max_v = preproc.loc[feat, 'Max']

            if max_v - min_v > 1e-9:
                final_data[feat] = (final_data[feat] - min_v) / (max_v - min_v)
                final_data[feat] = final_data[feat].clip(0.0, 1.0)
            else:
                final_data[feat] = 0.0

    # Clean binary features
    for feat in cfg.BINARY_FEATURES:
        if feat in final_data.columns:
            final_data[feat] = final_data[feat].fillna(0).astype(int)

    return final_data, train_targets, test_targets


def build_static_supra_graph(df_unique, master_map):
    """
        Constructs the Explicit Topology using a Hybrid Connectivity Strategy.

        Methodology (Thesis Sec 3.2.1):
            - "Instead of constructing full cliques... larger groups were truncated
              using stochastic neighbor sampling."
            - [cite_start]Caps: 100 (Geo) and 200 (Lender) to resolve memory bottlenecks [cite: 320-321].
            - Structure: "A dual-layer structure combining both edge sets" (Double Graph).

        Args:
            df_unique (pd.DataFrame): Unique loan metadata.
            master_map (dict): Mapping from LoanID to global node index.

        Returns:
            Tensor: Edge Index [2, E] representing the static connectivity.
        """
    print("  Building Static Supra-Graph...")
    num_nodes_layer = len(master_map)

    # Map Loan IDs to 0..N-1 indices
    df_unique['node_idx'] = df_unique['Loan_Sequence_Number'].map(master_map)

    edges_src, edges_dst = [], []

    def process_layer(group_col, limit, offset):
        groups = df_unique.groupby(group_col)['node_idx'].apply(list)
        rng = np.random.default_rng(42)

        for group_nodes in tqdm(groups, desc=f"Layer {group_col}", leave=False):
            n = len(group_nodes)
            if n < 2: continue

            nodes = np.array(group_nodes) + offset

            if n <= limit:
                # Full Clique (Small Groups)
                # "Small borrower groups retained full connectivity"
                grid = np.meshgrid(nodes, nodes)
                s, d = grid[0].flatten(), grid[1].flatten()
                mask = s != d
                edges_src.extend(s[mask])
                edges_dst.extend(d[mask])
            else:
                # Stochastic Sampling (Large Groups)
                # "Larger groups were truncated using stochastic neighbor sampling"
                for node in nodes:
                    neighbors = rng.choice(nodes, size=limit, replace=False)
                    neighbors = neighbors[neighbors != node]
                    edges_src.extend([node] * len(neighbors))
                    edges_dst.extend(neighbors)

    # 1. Build Layer 1 (Geographical Location - Zip Code)
    # Cap = 100 (Thesis Sec 3.2.1)
    process_layer('Geo_Key', cfg.MAX_DEGREE_STORAGE_GEO, offset=0)

    # 2. Build Layer 2 (Lender Company)
    # Cap = 200 (Thesis Sec 3.2.1)
    process_layer('Lender_Key', cfg.MAX_DEGREE_STORAGE_LENDER, offset=num_nodes_layer)

    # 3. Build Inter-Layer Identity Links
    # Connects the Geo-Node to its corresponding Lender-Node
    idx = torch.arange(num_nodes_layer, dtype=torch.long)
    edges_src.extend(idx.tolist())
    edges_dst.extend((idx + num_nodes_layer).tolist())
    edges_src.extend((idx + num_nodes_layer).tolist())
    edges_dst.extend(idx.tolist())

    # 4. Construct Tensor
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_index = to_undirected(edge_index)

    print(f"    Total Edges: {edge_index.shape[1]}")
    return edge_index


def create_sliding_windows(final_data, target_df, master_map, split_name):
    """
        Generates temporal sliding windows for LSTM training.

        Methodology (Thesis Sec 2.3 & 3.1.1):
            - Rolling Window: "As the rolling window shifts forward... the set of active
              [cite_start]nodes is updated" .
            - Burn-in Recovery: "Utilizing this 'pre-history' (2009-2011) to populate
              [cite_start]the 6-month Lookback Sequence" to avoid data censoring [cite: 301-302].

        Args:
            final_data (pd.DataFrame): The full behavioral dataset.
            target_df (pd.DataFrame): Target labels for specific snapshot dates.
            master_map (dict): Global node index mapping.
            split_name (str): 'TRAIN' or 'TEST' for logging.

        Returns:
            list: List of PyG Data objects, each representing one temporal window.
        """
    print(f" Processing {split_name} Windows...")

    target_dates = sorted(target_df['Snapshot_Date'].unique())
    data_by_month = {k: v for k, v in final_data.groupby('Monthly_Reporting_Period')}
    target_by_month = {k: v for k, v in target_df.groupby('Snapshot_Date')}

    num_nodes = len(master_map)
    graphs = []

    for target_date in tqdm(target_dates, desc=f"{split_name} Batches"):
        all_months = sorted(final_data['Monthly_Reporting_Period'].unique())
        try:
            t_idx = all_months.index(target_date)
        except ValueError:
            continue

        # Ensure enough history exists (T=6)
        if t_idx < cfg.SEQUENCE_LENGTH - 1:
            continue

        # Define Lookback Period (T=6)
        window_months = all_months[t_idx - cfg.SEQUENCE_LENGTH + 1: t_idx + 1]
        window_start = window_months[0]
        window_end = target_date

        if window_end not in data_by_month:
            continue

        anchor_df = data_by_month[window_end]

        # Active Loan Logic (Thesis Sec 2.3):
        # "A loan that defaults remains in the graph... until the rolling window
        # moves past the default event."
        active_loan_ids = set(
            anchor_df[anchor_df['Default_Month'] >= window_start]['Loan_Sequence_Number']
        )

        x_seq_list = []
        for m in window_months:
            x_snap = torch.zeros((num_nodes, cfg.NUM_INPUT_FEATURES), dtype=torch.float)

            if m in data_by_month:
                df = data_by_month[m]
                valid = df[
                    df['Loan_Sequence_Number'].isin(active_loan_ids) &
                    df['Loan_Sequence_Number'].isin(master_map)
                    ]

                if not valid.empty:
                    indices = valid['Loan_Sequence_Number'].map(master_map).values
                    feats = valid[cfg.NODE_FEATURES].values
                    x_snap[indices] = torch.tensor(feats, dtype=torch.float)

            # Duplicate features for Double Layer (Geo + Lender)
            x_seq_list.append(torch.cat([x_snap, x_snap], dim=0))

        x_sequence = torch.stack(x_seq_list, dim=0)

        # Targets (Masked)
        # "Loans that have already defaulted... are masked (assigned -1)"
        y_supra = torch.full((2 * num_nodes,), -1, dtype=torch.long)

        if target_date in target_by_month:
            t_df = target_by_month[target_date]
            valid_t = t_df[t_df['Loan_Sequence_Number'].isin(master_map)]

            if not valid_t.empty:
                indices = valid_t['Loan_Sequence_Number'].map(master_map).values
                targets_raw = pd.to_numeric(valid_t['Target_Y'], errors='coerce')
                targets_clean = targets_raw.fillna(-1).astype(int).values

                t_tensor = torch.tensor(targets_clean, dtype=torch.long)
                y_supra[indices] = t_tensor
                y_supra[indices + num_nodes] = t_tensor

        # Global Node IDs (Critical for mapping back to Static Topology)
        n_id = torch.arange(2 * num_nodes, dtype=torch.long)

        graphs.append(Data(x=x_sequence, y=y_supra, n_id=n_id))

    return graphs


if __name__ == "__main__":
    cfg.GRAPH_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    final_data, train_targets, test_targets = load_and_scale_data()

    all_loans = final_data['Loan_Sequence_Number'].unique()
    master_map = {loan: i for i, loan in enumerate(sorted(all_loans))}

    # 1. Static Topology Construction (Thesis Sec 3.2.1)
    edge_save_path = cfg.GRAPH_SAVE_DIR / "static_edge_index.pt"
    if edge_save_path.exists():
        print(f" Found existing {edge_save_path}. Skipping build.")
    else:
        df_uniq = final_data.drop_duplicates('Loan_Sequence_Number')[['Loan_Sequence_Number', 'Geo_Key', 'Lender_Key']]
        edge_index = build_static_supra_graph(df_uniq, master_map)
        torch.save(edge_index, edge_save_path)
        print(f" Saved {edge_save_path}")
        del edge_index, df_uniq
        gc.collect()

    # 2. Train Windows Generation (Thesis Sec 3.1.1)
    train_save_path = cfg.GRAPH_SAVE_DIR / "train_features.pt"
    if train_save_path.exists():
        print(f" Found existing {train_save_path}. Skipping build.")
        print(" NOTE: If you are fixing the 'n_id' error, DELETE this file and rerun!")
    else:
        train_graphs = create_sliding_windows(final_data, train_targets, master_map, "TRAIN")
        torch.save(train_graphs, train_save_path)
        print(f" Saved {train_save_path}")
        del train_graphs
        gc.collect()

    # 3. Test Windows Generation
    test_save_path = cfg.GRAPH_SAVE_DIR / "test_features.pt"
    if test_save_path.exists():
        print(f" Found existing {test_save_path}. Skipping build.")
    else:
        test_graphs = create_sliding_windows(final_data, test_targets, master_map, "TEST")
        torch.save(test_graphs, test_save_path)
        print(f" Saved {test_save_path}")

    # Metadata storage
    torch.save({'num_nodes': len(master_map) * 2, 'num_features': cfg.NUM_INPUT_FEATURES},
               cfg.GRAPH_SAVE_DIR / "metadata.pt")

    print("\n H1 Data Prep Complete.")