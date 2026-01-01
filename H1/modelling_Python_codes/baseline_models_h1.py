# ==============================================================================
# baseline_models_h1.py (H1)
#
# PURPOSE:
#   Trains and evaluates the complete set of baseline models defined in Thesis
#   Section 4.3 to establish a performance floor for the H1 experiment.
#   This includes both "Non-GNN Benchmarks" (LR, XGB, DNN) and "Static GNN
#   Benchmarks" (GCN, GAT) operating on the Explicit Topology.
#
# LOGIC & THESIS ALIGNMENT:
#   1. Non-GNN Baselines (Thesis Sec 4.3.2):
#      - Logistic Regression: Uses class-balanced weights.
#      - XGBoost: Optimized via GridSearch, uses scale_pos_weight.
#      - DNN: Strict replication of the Reference Model's Appendix B architecture
#        (Input -> 30 -> 50 -> 20 -> 1) with 50% dropout.
#   2. Static GNN Baselines (Thesis Sec 4.3.1):
#      - "Stochastic Static" Training: Samples one random temporal snapshot per epoch
#        to approximate the distribution without full-batch memory costs.
#      - Node Isolation: Applies 50% random node masking during Training, Validation,
#        and Test to match the Reference Model's sparsity protocol.
#      - Explicit Topology: Constructs graphs based on Zip/Lender keys.
#   3. Metrics (Thesis Sec 4.1):
#      - Reports AUC and F1 (Fixed Threshold 0.5) for consistency.
#
# INPUTS:
#   - baseline_data_h1_aligned.pkl (From baseline_data_prep_h1.py)
#
# OUTPUTS:
#   - baseline_results_h1.json: Dictionary of metrics (AUC, F1) for all models.
# ==============================================================================

import sys
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from tqdm import tqdm
import config_h1 as cfg
import copy
import time
import datetime
import random


# --- Reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# --- Utilities ---
def load_data():
    """
    Loads the preprocessed and aligned baseline dataset from disk.

    The function reads the serialized dataset produced by the H1 baseline
    preparation script, validates its existence, and ensures all feature
    matrices are cast to float32 for neural network compatibility.

    Returns:
        dict: Dictionary containing train/validation/test splits and labels
              (e.g., X_train, y_train, X_val, y_val, X_test, y_test)
    """
    load_path = cfg.DATA_DIR / "baseline_data_h1_aligned.pkl"
    if not load_path.exists():
        print(f" Run baseline_data_prep_h1.py first!")
        sys.exit(1)

    with open(load_path, 'rb') as f:
        data = pickle.load(f)

    # Ensure float32 for Neural Network compatibility
    for k in ['X_train', 'X_val', 'X_test']:
        if k in data:
            data[k] = np.nan_to_num(data[k].astype('float32'))
    return data


def get_metrics(y_true, y_proba, name="Test"):
    """
    Computes AUC and F1-score for binary classification.

    Predicted labels are obtained using a fixed probability threshold of 0.5.
    Safely handles degenerate cases where only one class is present.

    Args:
        y_true (array-like): Ground-truth binary labels.
        y_proba (array-like): Predicted positive class probabilities.
        name (str): Label used when printing metric results.

    Returns:
        dict: Dictionary containing AUC and F1-score.
    """
    try:
        if len(np.unique(y_true)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(y_true, y_proba)
        y_pred = (y_proba > 0.5).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    except Exception:
        auc, f1 = 0.5, 0.0
    print(f"   {name} | AUC: {auc:.4f} | F1: {f1:.4f}")
    return {"auc": auc, "f1": f1}


# --- Strict Graph Construction (With Isolation) ---
def get_adj_indices_strict(geo, lend, mode, num_nodes, isolation_prob=0.5):
    """
    Constructs an explicit categorical graph topology with stochastic node isolation.

    Edges are formed by shared geographical and lender identifiers, while a
    fraction of nodes are randomly isolated by removing all incident edges.
    This regularization simulates graph sparsity and mitigates overfitting.

    Args:
        geo (Tensor): Geographical group identifiers per node.
        lend (Tensor): Lender group identifiers per node.
        mode (str): Topology mode controlling edge construction logic.
        num_nodes (int): Total number of nodes in the graph.
        isolation_prob (float): Probability of isolating a node (default: 0.5).

    Returns:
        Tensor: Edge index defining the constructed graph topology.
    """
    # 1. Isolation Mask (The "Sparsity" Simulation)
    if isolation_prob > 0:
        active_mask = np.random.rand(num_nodes) > isolation_prob
        active_indices = np.where(active_mask)[0]
    else:
        active_indices = np.arange(num_nodes)
        active_mask = np.ones(num_nodes, dtype=bool)

    # Filter keys to only active nodes (Isolated nodes lose all edges)
    geo_active = geo.cpu().numpy()[active_mask]
    lend_active = lend.cpu().numpy()[active_mask]

    # The 'idx' in dataframe must map back to original indices
    node_indices = active_indices

    def build_cliques(keys):
        """
            Builds intra-group graph edges by forming cliques over shared categorical keys.

            Nodes with the same key (zip code or lender) are fully connected.
            For large groups, connectivity is capped by randomly sampling a fixed number
            of neighbors per node to control memory usage.

            Args:
                keys (array-like): Categorical group identifiers for active nodes.

            Returns:
                Tuple[np.ndarray, np.ndarray]: Source and destination node indices
                representing directed edges within each group.
            """
        df = pd.DataFrame({'key': keys, 'idx': node_indices})
        groups = df.groupby('key')['idx'].apply(list)

        src_list, dst_list = [], []
        for grp in groups:
            grp_np = np.array(grp)
            k = len(grp_np)
            if k <= 1: continue

            # RAM Safety Cap (Constraint-Awareness Sec 3.2.1)
            # Matches the "Hybrid Connectivity Strategy" where large groups are truncated.
            MAX_NEIGHBORS = 50
            if k > MAX_NEIGHBORS + 1:
                for node_idx in grp_np:
                    targets = np.random.choice(grp_np, size=MAX_NEIGHBORS, replace=False)
                    targets = targets[targets != node_idx]
                    src_list.append(np.full(len(targets), node_idx))
                    dst_list.append(targets)
            else:
                # Full Clique for small blocks
                r = np.repeat(grp_np, k)
                t = np.tile(grp_np, k)
                mask = r != t
                src_list.append(r[mask])
                dst_list.append(t[mask])

        if len(src_list) > 0:
            return np.concatenate(src_list), np.concatenate(dst_list)
        return np.array([]), np.array([])

    src_total, dst_total = [], []

    # Area Graph (Zip Code)
    if mode in ['area', 'double']:
        s, d = build_cliques(geo_active)
        if len(s) > 0: src_total.append(s); dst_total.append(d)

    # Company Graph (Lender)
    if mode in ['company', 'double']:
        s, d = build_cliques(lend_active)
        if mode == 'double':
            s += num_nodes
            d += num_nodes
        if len(s) > 0: src_total.append(s); dst_total.append(d)

    # Double Graph Inter-layer links
    if mode == 'double':
        # Link active nodes to their twins in the other layer
        inter_src = active_indices
        inter_dst = active_indices + num_nodes
        src_total.append(np.concatenate([inter_src, inter_dst]))
        dst_total.append(np.concatenate([inter_dst, inter_src]))

    if not src_total:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor([np.concatenate(src_total), np.concatenate(dst_total)], dtype=torch.long)


# --- Non-GNN Baselines (Thesis Sec 4.3.2) ---
def train_logreg(X_train, y_train, X_val, y_val, X_test, y_test):
    """
        Trains and evaluates a logistic regression baseline model.

        The model is fitted using class-balanced weights and tuned over L1/L2
        regularization via grid search on a fixed validation split. Performance
        is reported on the held-out test set using AUC and F1-score.

        Returns:
            dict: Test-set evaluation metrics (AUC, F1).
        """
    print("\n--- Logistic Regression (Paper Baseline) ---")
    X_comb = np.vstack((X_train, X_val))
    y_comb = np.hstack((y_train, y_val))

    # PredefinedSplit ensures GridSearch uses our fixed Validation set
    split_index = [-1] * len(X_train) + [0] * len(X_val)
    pds = PredefinedSplit(test_fold=split_index)

    lr = LogisticRegression(class_weight='balanced', solver='saga', max_iter=1000, random_state=42)
    param_grid = {'penalty': ['l1', 'l2']}

    grid = GridSearchCV(lr, param_grid, cv=pds, scoring='roc_auc', n_jobs=1, verbose=1)
    grid.fit(X_comb, y_comb)
    print(f"   Best Params: {grid.best_params_}")
    return get_metrics(y_test, grid.predict_proba(X_test)[:, 1], "LR")


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """
        Trains and evaluates an XGBoost baseline classifier.

        The model is tuned via grid search on a fixed validation split and
        incorporates class imbalance handling through scale_pos_weight.
        Performance is evaluated on the held-out test set using AUC and F1-score.

        Returns:
            dict: Test-set evaluation metrics (AUC, F1).
        """
    print("\n--- XGBoost (Paper Baseline) ---")
    X_comb = np.vstack((X_train, X_val))
    y_comb = np.hstack((y_train, y_val))
    split_index = [-1] * len(X_train) + [0] * len(X_val)
    pds = PredefinedSplit(test_fold=split_index)

    # Calculate scale_pos_weight for imbalance
    pos_count = (y_train == 1).sum()
    scale_weight = (y_train == 0).sum() / max(pos_count, 1)
    xgb = XGBClassifier(scale_pos_weight=scale_weight, eval_metric='logloss', tree_method='hist', random_state=42,
                        n_jobs=4)
    param_grid = {'learning_rate': [0.001, 0.01, 0.1], 'max_depth': [2, 3, 4], 'n_estimators': [50, 100, 250, 500],
                  'reg_alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    grid = GridSearchCV(xgb, param_grid, cv=pds, scoring='roc_auc', n_jobs=1, verbose=1)
    grid.fit(X_comb, y_comb)
    print(f"   Best Params: {grid.best_params_}")
    return get_metrics(y_test, grid.predict_proba(X_test)[:, 1], "XGB")


class PaperDNN(nn.Module):
    """
        Feed-forward neural network baseline for binary classification.

        Architecture:
            - Linear(input_dim → 30) + ReLU + Dropout(0.5)
            - Linear(30 → 50) + ReLU + Dropout(0.5)
            - Linear(50 → 20) + ReLU + Dropout(0.5)
            - Linear(20 → 1) → Output logit
        """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 30), nn.ReLU(), nn.Dropout(0.5), nn.Linear(30, 50), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(50, 20), nn.ReLU(), nn.Dropout(0.5), nn.Linear(20, 1))

    def forward(self, x): return self.net(x)


def train_dnn(X_train, y_train, X_val, y_val, X_test, y_test):
    """
        Trains the feed-forward PaperDNN on the baseline dataset.

        - Uses BCEWithLogitsLoss with pos_weight to address class imbalance.
        - Early stopping based on validation loss (patience=50).
        - Batch training with Adam optimizer (lr=0.001).
        - Maximum epochs: 200, batch size: 4096.

        Returns:
            dict: Standard evaluation metrics on test set (AUC, F1).
        """
    print("\n--- DNN (Appendix B) ---")
    device = cfg.DEVICE
    model = PaperDNN(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Cost-Sensitive Loss
    w_val = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pos_weight = torch.tensor(min(w_val, 10.0), device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    xt_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    yt_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    xv_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    best_loss = float('inf');
    best_model = None;
    patience_curr = 0;
    EPOCHS = 200;
    PATIENCE = 50
    batch_size = 4096;
    n_batches = int(np.ceil(len(X_train) / batch_size))

    for epoch in range(EPOCHS):
        model.train()
        indices = torch.randperm(len(xt_t))
        for i in range(n_batches):
            idx = indices[i * batch_size: (i + 1) * batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xt_t[idx]), yt_t[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(xv_t), yv_t).item()
        if val_loss < best_loss:
            best_loss = val_loss; best_model = copy.deepcopy(model.state_dict()); patience_curr = 0
        else:
            patience_curr += 1;
        if patience_curr >= PATIENCE: break

    model.load_state_dict(best_model)
    with torch.no_grad():
        preds = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32).to(device))).cpu().numpy()
    return get_metrics(y_test, preds, "DNN")


# --- Static GNN Baselines (Thesis Sec 4.3.1) ---
class PaperDecoder(nn.Module):
    """
        Standard decoder used in the reference model (Fig. 4).

        Architecture:
            - Linear(input_dim → 10) + ReLU + Dropout(0.5)
            - Linear(10 → 1)

        Forward pass:
            - Applies the sequential layers to input tensor x.
        """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 10), nn.ReLU(), nn.Dropout(0.5), nn.Linear(10, 1))

    def forward(self, x): return self.net(x)


class StaticGNN(nn.Module):
    """
        Baseline static GNN for node-level prediction.

        Architecture:
            - Two GCN/GAT layers (hidden=20 by default) with ELU activation
            - Dropout (p=0.5) after first layer
            - Linear decoder (PaperDecoder) mapping hidden features to output

        Args:
            in_channels (int): Number of input node features
            hidden (int, optional): Hidden dimension for GNN layers (default: 20)
            model_type (str, optional): 'GCN' or 'GAT' (default: 'GCN')

        Forward pass:
            - x: node feature tensor
            - edge_index: graph connectivity
            - returns node-level predictions
        """
    def __init__(self, in_channels, hidden=20, model_type='GCN'):
        super().__init__()
        conv = GATConv if model_type == 'GAT' else GCNConv
        self.conv1 = conv(in_channels, hidden)
        self.conv2 = conv(hidden, hidden)
        self.decoder = PaperDecoder(hidden)
        self.dropout = 0.5

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.decoder(x)


def train_gnn_static_baseline(data, model_type, mode):
    """
        Trains a static GNN baseline on H1 data using stochastic single-window selection.

        Thesis Alignment:
            - Sec 4.3.1: Explicit categorical topology with node isolation
            - Uses 50% node isolation during training, validation, and testing
            - Weighted BCE loss to handle class imbalance
            - Early stopping based on validation loss (patience=50)

        Args:
            data (dict): Preprocessed H1 dataset with 'train_windows', 'val_window', 'test_window'
            model_type (str): 'GCN' or 'GAT'
            mode (str): 'area', 'company', or 'double' (controls graph layers)

        Returns:
            dict: Test metrics {'auc': float, 'f1': float}
        """
    print(f"\n Training H1 {model_type} ({mode}) [Stochastic Static]...")
    device = cfg.DEVICE

    # --- Initialize model and optimizer ---
    first_window = data['train_windows'][0]
    model = StaticGNN(first_window['X'].shape[1], 20, model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Weighted BCE loss ---
    all_y = np.concatenate([w['y'] for w in data['train_windows']])
    pos = (all_y == 1).sum();
    neg = (all_y == 0).sum()
    pos_weight = torch.tensor(min(neg / max(pos, 1), 10.0), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Early stopping config ---
    EPOCHS = 200
    PATIENCE = 50
    best_loss = float('inf')
    patience_curr = 0
    best_model = None

    # --- Prepare validation data ---
    val_w = data['val_window']
    X_val = torch.tensor(val_w['X'], dtype=torch.float32).to(device)
    y_val = torch.tensor(val_w['y'], dtype=torch.float32).view(-1, 1).to(device)
    val_edges = get_adj_indices_strict(
        torch.tensor(val_w['geo']), torch.tensor(val_w['lend']), mode, len(X_val),
        isolation_prob=0.5
    ).to(device)

    if mode == 'double':
        X_val_in = torch.cat([X_val, X_val], dim=0)
        y_val_in = torch.cat([y_val, y_val], dim=0)
    else:
        X_val_in, y_val_in = X_val, y_val

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()

        # Stochastic static: pick one random window per epoch
        idx = np.random.randint(0, len(data['train_windows']))
        window = data['train_windows'][idx]

        X_w = torch.tensor(window['X'], dtype=torch.float32).to(device)
        y_w = torch.tensor(window['y'], dtype=torch.float32).view(-1, 1).to(device)

        # Construct Graph with 50% Isolation
        edge_index = get_adj_indices_strict(
            torch.tensor(window['geo']),
            torch.tensor(window['lend']),
            mode,
            len(X_w),
            isolation_prob=0.5
        ).to(device)

        if mode == 'double':
            X_in = torch.cat([X_w, X_w], dim=0)
            y_in = torch.cat([y_w, y_w], dim=0)
        else:
            X_in, y_in = X_w, y_w

        optimizer.zero_grad()
        out = model(X_in, edge_index)
        loss = criterion(out, y_in)
        loss.backward()
        optimizer.step()

        # --- VALIDATION ---
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_in, val_edges)
            val_loss = criterion(val_out, y_val_in).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_curr = 0
        else:
            patience_curr += 1
            if patience_curr >= PATIENCE:
                print(f"   Early stopping at epoch {epoch}")
                break

    # Test
    model.load_state_dict(best_model)
    model.eval()
    test_w = data['test_window']
    X_test = torch.tensor(test_w['X'], dtype=torch.float32).to(device)

    # Test Isolation = 0.5
    test_edges = get_adj_indices_strict(
        torch.tensor(test_w['geo']), torch.tensor(test_w['lend']), mode, len(X_test),
        isolation_prob=0.5
    ).to(device)

    if mode == 'double':
        X_test_in = torch.cat([X_test, X_test], dim=0)
    else:
        X_test_in = X_test

    with torch.no_grad():
        out_test = model(X_test_in, test_edges)
        if mode == 'double':
            preds = torch.sigmoid(out_test).cpu().numpy()[:len(X_test)]
        else:
            preds = torch.sigmoid(out_test).cpu().numpy().flatten()

    return get_metrics(test_w['y'], preds, f"{model_type}-{mode}")


def main():
    data = load_data()
    results_path = cfg.GRAPH_SAVE_DIR / "baseline_results_h1.json"

    # Checkpoint Resume Logic
    if results_path.exists():
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            print(f" Resuming from existing results ({len(results)} found).")
        except:
            results = {}
    else:
        results = {}

    def save_snapshot():
        with open(results_path, "w") as f:
            class NpEncoder(json.JSONEncoder):
                def default(self, o): return int(o) if isinstance(o, np.integer) else float(o) if isinstance(o,
                                                                                                             np.floating) else super().default(
                    o)

            json.dump(results, f, indent=4, cls=NpEncoder)
        print(f"   Checkpoint saved.")

    # --- Run Models (If not already in results) ---
    if 'LR' not in results:
        results['LR'] = train_logreg(data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'],
                                     data['y_test'])
        save_snapshot()
    else:
        print("   LR already done.")

    if 'XGB' not in results:
        results['XGB'] = train_xgboost(data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'],
                                       data['y_test'])
        save_snapshot()
    else:
        print("   XGB already done.")

    if 'DNN' not in results:
        results['DNN'] = train_dnn(data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'],
                                   data['y_test'])
        save_snapshot()
    else:
        print("   DNN already done.")

    # GNN Loop: Area, Company, Double
    for m in ['area', 'company', 'double']:
        gcn_key = f'GCN_{m}'
        if gcn_key not in results:
            results[gcn_key] = train_gnn_static_baseline(data, 'GCN', m)
            save_snapshot()
        else:
            print(f"   {gcn_key} already done.")

        gat_key = f'GAT_{m}'
        if gat_key not in results:
            results[gat_key] = train_gnn_static_baseline(data, 'GAT', m)
            save_snapshot()
        else:
            print(f"   {gat_key} already done.")

    print(" H1 Baselines Complete.")


if __name__ == "__main__":
    start_time = time.time()
    print(f" Script started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    main()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n Script finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Total Runtime: {str(datetime.timedelta(seconds=int(elapsed)))}")