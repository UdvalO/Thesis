# ==============================================================================
# baseline_models_h2.py (H2: Implicit Topology Baselines)
#
# PURPOSE:
#   Trains and evaluates the static baseline models for Hypothesis 2.
#   Unlike H1 (Explicit Topology), these models operate on "Implicit" graphs
#   constructed dynamically via k-Nearest Neighbors (KNN) on behavioral features.
#
# LOGIC & THESIS ALIGNMENT:
#   1. Implicit Topology (Thesis Sec 3.3.1):
#      - "Edges are defined dynamically... based on behavioral similarity."
#      - The GCN/GAT baselines build graphs at runtime using KNN, strictly
#        testing the hypothesis that behavior > metadata.
#
#   2. Node Isolation (Thesis Sec 4.3.1):
#      - "To prevent overfitting... 50% of nodes are randomly isolated (masked)
#        during training."
#      - This is strictly replicated here by masking nodes before KNN construction.
#
#   3. Model Architecture (Thesis Appendix B):
#      - DNN: Replicates the specific 4-layer perceptron (30-50-20-1) defined
#        in the Reference Model's supplementary material.
#
# INPUTS:
#   - baseline_data_h2_aligned.pkl: The temporally averaged static datasets.
#
# OUTPUTS:
#   - baseline_results_h2.json: Metrics for LR, XGB, DNN, GCN-KNN, GAT-KNN.
# ==============================================================================

import sys
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, knn_graph
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
import config_h2 as cfg
import copy
import random
import time
import datetime
from sklearn.neighbors import NearestNeighbors


# --- Reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

def load_data():
    """
        Loads the temporally aligned H2 data from the pickle file.
        Ensures float32 precision for PyTorch compatibility.

        Returns:
            dict: A dictionary containing 'X_train', 'y_train', 'X_val', 'y_val',
                  'X_test', 'y_test' (numpy arrays) and 'train_windows' (list).
        """
    load_path = cfg.SAVE_DIR / "baseline_data_h2_aligned.pkl"
    if not load_path.exists():
        sys.exit(" Run baseline_data_prep_h2.py first!")
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    for k in ['X_train', 'X_val', 'X_test']:
        if k in data:
            data[k] = np.nan_to_num(data[k].astype('float32'))
    return data


def get_metrics(y_true, y_proba, name="Test"):
    """
        Computes standard evaluation metrics (AUC, F1).

        Args:
            y_true (np.ndarray): True binary labels (0 or 1).
            y_proba (np.ndarray): Predicted probabilities (0.0 to 1.0).
            name (str): Name of the model or split for logging.

        Returns:
            dict: Dictionary containing {"auc": float, "f1": float}.
        """
    try:
        if len(np.unique(y_true)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(y_true, y_proba)
        y_pred = (y_proba > 0.5).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    except Exception as e:
        print(f"   Metric Error ({name}): {e}")
        auc, f1 = 0.5, 0.0
    print(f"   {name} | AUC: {auc:.4f} | F1: {f1:.4f}")
    return {"auc": auc, "f1": f1}


# --- Non-GNN Baselines (Exact Copy of H1 for Fairness) ---
def train_logreg(X_train, y_train, X_val, y_val, X_test, y_test):
    """
        Trains Logistic Regression with Grid Search.

        Methodology (Thesis Sec 4.3.1):
            - Strictly aligns with H1 methodology to isolate the impact of
              features vs. topology.
            - Uses 'PredefinedSplit' to enforce validation on Window 13.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.

        Returns:
            dict: Metrics {"auc": float, "f1": float} on the Test set.
        """
    print("\n--- Logistic Regression (H2: Grid Search) ---")
    X_comb = np.vstack((X_train, X_val))
    y_comb = np.hstack((y_train, y_val))
    # PredefinedSplit ensures we validate on the explicit Validation Set (Window 13)
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
        Trains XGBoost with Grid Search.

        Methodology:
            - Benchmarking standard for tabular efficiency on the H2 feature set.
            - Optimizes tree depth and learning rate via predefined validation split.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.

        Returns:
            dict: Metrics {"auc": float, "f1": float} on the Test set.
        """
    print("\n--- XGBoost (H2: Grid Search) ---")
    X_comb = np.vstack((X_train, X_val))
    y_comb = np.hstack((y_train, y_val))
    split_index = [-1] * len(X_train) + [0] * len(X_val)
    pds = PredefinedSplit(test_fold=split_index)

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
        Reference DNN Architecture (Thesis Appendix B).
        Structure: Input -> 30 -> 50 -> 20 -> 1.
        """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 30), nn.ReLU(), nn.Dropout(0.5), nn.Linear(30, 50), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(50, 20), nn.ReLU(), nn.Dropout(0.5), nn.Linear(20, 1))

    def forward(self, x): return self.net(x)


def train_dnn(X_train, y_train, X_val, y_val, X_test, y_test):
    """
        Trains the Appendix B DNN using the H2 feature set.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.

        Returns:
            dict: Metrics {"auc": float, "f1": float} on the Test set.
        """
    print("\n--- DNN (H2 Appendix B) ---")
    device = cfg.DEVICE
    model = PaperDNN(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Weighted Loss for Imbalance
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


# --- GNN (H2: Implicit KNN) ---
class PaperDecoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 10), nn.ReLU(), nn.Dropout(0.5), nn.Linear(10, 1))

    def forward(self, x): return self.net(x)


class StaticGNN(nn.Module):
    """
        Standard GNN baseline (GCN/GAT) applied to implicit graphs.
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


def get_knn_graph(x, k):
    """
    Robust KNN Graph Construction (H2 Core Logic).

    Methodology (Thesis Sec 3.3.1):
        - Constructs edges dynamically based on feature similarity.
        - Uses Sklearn (CPU) as a robust fallback to bypass potential
          torch-cluster dependencies on consumer hardware.

    Args:
        x (Tensor): Node features [Nodes, Features].
        k (int): Number of neighbors to find.

    Returns:
        Tensor: Edge Index [2, E] (Source, Target).
    """
    # 1. Move to CPU for Sklearn
    x_np = x.detach().cpu().numpy()

    # 2. Compute KNN (k+1 because sklearn includes the node itself as neighbor 0)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(x_np)
    _, indices = nbrs.kneighbors(x_np)

    # 3. Exclude self-loops (column 0 is self)
    neighbors = indices[:, 1:]

    # 4. Construct Edge Index (Source -> Target)
    n_nodes = x.shape[0]

    # Source: The neighbors found
    src = neighbors.flatten()

    # Target: The query node (repeated k times)
    dst = np.repeat(np.arange(n_nodes), k)

    # 5. Convert back to Tensor on original Device
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=x.device)

    return edge_index

def train_gnn_implicit_baseline(data, model_type, k_neighbors):
    """
        Trains a Static GNN on Implicit Topology graphs.

        Methodology (Thesis Sec 4.3.1):
            - Stochastic Static Training: Randomly samples temporal windows.
            - Implicit Construction: Builds KNN graph dynamically at runtime.
            - Node Isolation: Applies "50% random node isolation"
              to prevent overfitting to static feature correlations.

        Args:
            data (dict): Dictionary containing 'train_windows', 'val_window', 'test_window'.
            model_type (str): 'GCN' or 'GAT'.
            k_neighbors (int): Number of neighbors for KNN construction.

        Returns:
            dict: Metrics {"auc": float, "f1": float} on the Test set.
        """
    print(f"\n🚀 Training H2 {model_type} [Implicit KNN k={k_neighbors}]...")
    device = cfg.DEVICE

    # Initialize from first window size
    first_window = data['train_windows'][0]
    model = StaticGNN(first_window['X'].shape[1], 20, model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Weighted Loss (Global)
    all_y = np.concatenate([w['y'] for w in data['train_windows']])
    pos = (all_y == 1).sum();
    neg = (all_y == 0).sum()
    pos_weight = torch.tensor(min(neg / max(pos, 1), 10.0), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss = float('inf')
    best_model = None
    patience_curr = 0
    EPOCHS = 200
    PATIENCE = 50

    # Load Validation Data & Pre-calc KNN
    val_w = data['val_window']
    X_val = torch.tensor(val_w['X'], dtype=torch.float32).to(device)
    y_val = torch.tensor(val_w['y'], dtype=torch.float32).view(-1, 1).to(device)

    # --- VALIDATION ISOLATION (Strict Alignment) ---
    # Thesis Sec 4.3.1: "50% of nodes are randomly isolated... in each snapshot."
    mask_val = torch.rand(X_val.shape[0], device=device) > 0.5
    if mask_val.sum() < k_neighbors + 1: mask_val = torch.ones_like(mask_val).bool()

    # Build Validation Graph (Implicit KNN)
    val_active_indices = torch.where(mask_val)[0]
    val_active_edges_local = get_knn_graph(X_val[mask_val], k=k_neighbors)

    # Remap edges to global indices
    src_local, dst_local = val_active_edges_local
    src_global = val_active_indices[src_local]
    dst_global = val_active_indices[dst_local]
    val_edge_index = torch.stack([src_global, dst_global], dim=0)

    for epoch in range(EPOCHS):
        model.train()

        # --- STOCHASTIC STATIC TRAINING ---
        # 1. Sample Window
        idx = np.random.randint(0, len(data['train_windows']))
        window = data['train_windows'][idx]

        X_w = torch.tensor(window['X'], dtype=torch.float32).to(device)
        y_w = torch.tensor(window['y'], dtype=torch.float32).view(-1, 1).to(device)

        # 2. Apply 50% Node Isolation (Thesis Sec 4.3.1)
        mask_tr = torch.rand(X_w.shape[0], device=device) > 0.5
        if mask_tr.sum() < k_neighbors + 1: mask_tr = torch.ones_like(mask_tr).bool()

        active_indices = torch.where(mask_tr)[0]

        # 3. Build Implicit KNN on Active Nodes Only
        edges_local = get_knn_graph(X_w[mask_tr], k=k_neighbors)

        # Remap to global size
        src_loc, dst_loc = edges_local
        src_glob = active_indices[src_loc]
        dst_glob = active_indices[dst_loc]
        edge_index_tr = torch.stack([src_glob, dst_glob], dim=0)

        optimizer.zero_grad()
        out = model(X_w, edge_index_tr)
        loss = criterion(out, y_w)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val, val_edge_index)
            val_loss = criterion(val_out, y_val).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_curr = 0
        else:
            patience_curr += 1
            if patience_curr >= PATIENCE:
                print(f"   Early stopping at epoch {epoch}")
                break

    # Inference (Test)
    model.load_state_dict(best_model)
    model.eval()
    test_w = data['test_window']
    X_test = torch.tensor(test_w['X'], dtype=torch.float32).to(device)

    # Test Isolation 50%
    mask_te = torch.rand(X_test.shape[0], device=device) > 0.5
    if mask_te.sum() < k_neighbors + 1: mask_te = torch.ones_like(mask_te).bool()

    active_te = torch.where(mask_te)[0]
    edges_local_te = get_knn_graph(X_test[mask_te], k=k_neighbors)
    src_te = active_te[edges_local_te[0]]
    dst_te = active_te[edges_local_te[1]]
    te_edge_index = torch.stack([src_te, dst_te], dim=0)

    with torch.no_grad():
        out = model(X_test, te_edge_index)
        preds = torch.sigmoid(out).cpu().numpy().flatten()

    return get_metrics(test_w['y'], preds, f"{model_type}-KNN{k_neighbors}")


def main():
    data = load_data()
    results_path = cfg.SAVE_DIR / "baseline_results_h2.json"

    if results_path.exists():
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            print(f" Resuming ({len(results)} found).")
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

    # Baselines
    if 'LR' not in results:
        results['LR'] = train_logreg(data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'],
                                     data['y_test'])
        save_snapshot()

    if 'XGB' not in results:
        results['XGB'] = train_xgboost(data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'],
                                       data['y_test'])
        save_snapshot()

    if 'DNN' not in results:
        results['DNN'] = train_dnn(data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'],
                                   data['y_test'])
        save_snapshot()

    # H2: Implicit KNN
    for k in [6, 7]:
        if f'GCN_KNN_{k}' not in results:
            results[f'GCN_KNN_{k}'] = train_gnn_implicit_baseline(data, 'GCN', k)
            save_snapshot()

        if f'GAT_KNN_{k}' not in results:
            results[f'GAT_KNN_{k}'] = train_gnn_implicit_baseline(data, 'GAT', k)
            save_snapshot()

    print("H2 Baselines Complete.")


if __name__ == "__main__":
    start_time = time.time()
    print(f" Script started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    main()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n Script finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Total Runtime: {str(datetime.timedelta(seconds=int(elapsed)))}")