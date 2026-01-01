# ==============================================================================
# interpretability_h2.py (H2)
#
# PURPOSE:
#   Provide post-hoc interpretability for the trained H2 GNN model using SHAP (KernelExplainer).
#
# STEPS:
#   1. Load preprocessed test graph sequences and filter valid labeled nodes.
#   2. Flatten temporal node features for SHAP compatibility.
#   3. Construct a reduced background distribution using K-Means summarization.
#   4. Wrap the DT-GNN model to match the SHAP KernelExplainer interface,
#      including synthetic double-layer graph construction.
#   5. Compute SHAP values for a subset of test nodes.
#   6. Aggregate SHAP values across time to obtain feature-level interpretation.
#   7. Generate and save global summary and dependence plots.
#
# INPUTS:
#   - Test graph sequences: test_graphs.pt (on remote computer)
#   - Trained DT-GNN model checkpoint: best_model_final.pt (on remote computer)
#   - Configuration: config.py
#
# OUTPUTS:
#   - Global SHAP summary plot: h2_shap_summary.png
#   - Feature dependence plots: h2_shap_dep_<feature>.png
# ==============================================================================

# --- 0. DEPENDENCIES ---
import torch
import torch.nn as nn
import numpy as np
import shap
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import config_h2 as cfg
from model_architecture_h2 import DynamicTemporalGNN
import sys

# --- 1. CONFIGURATION ---
N_BACKGROUND = 50   # Background samples for SHAP baseline (K-Means)
N_EXPLAIN = 100     # Number of node sequences to explain
FIGURE_DIR = cfg.SAVE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True, parents=True)

FEATURE_NAMES = cfg.NODE_FEATURES

# --- 2. SHAP MODEL WRAPPER ---
class H2KernelWrapper:
    """
    Adapter class that wraps the H2 DT-GNN model for SHAP KernelExplainer.

    The wrapper:
        - Reshapes flattened temporal inputs into (Batch, Time, Features)
        - Builds a synthetic double-layer graph with self-loops and vertical connections
        - Performs a forward pass and returns probabilities for master nodes
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, x_flat):
        """
                Forward function used by SHAP KernelExplainer.

                Args:
                    x_flat (np.ndarray): Flattened node features (Batch, Time*Features)

                Returns:
                    np.ndarray: Prediction probabilities for master nodes (Batch,)
                """
        batch_size = x_flat.shape[0]
        num_feats = len(FEATURE_NAMES)
        seq_len = x_flat.shape[1] // num_feats

        # --- Reshape input to temporal tensor ---
        x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=self.device)
        x_reshaped = x_tensor.view(batch_size, seq_len, num_feats)

        # --- Build synthetic double-layer graph ---
        idx_l1 = torch.arange(batch_size, device=self.device)
        idx_l2 = idx_l1 + batch_size

        edge_index = torch.cat([
            torch.stack([idx_l1, idx_l1], dim=0),  # L1 self-loops
            torch.stack([idx_l2, idx_l2], dim=0),  # L2 self-loops
            torch.stack([idx_l1, idx_l2], dim=0),  # L1 → L2
            torch.stack([idx_l2, idx_l1], dim=0),  # L2 → L1
        ], dim=1)

        # --- Construct sequence for model ---
        sequence = []
        for t in range(seq_len):
            x_t = x_reshaped[:, t, :]
            x_dup = torch.cat([x_t, x_t], dim=0) # Duplicate for double-layer
            data = Data(x=x_dup, edge_index=edge_index)
            sequence.append(data)

        # --- Forward pass ---
        self.model.eval()
        with torch.no_grad():
            logits = self.model(sequence)  # (2*Batch, 1)
            logits_master = logits[:batch_size]  # Select master nodes
            probs = torch.sigmoid(logits_master).cpu().numpy()

        return probs

# --- 3. MAIN PIPELINE ---
def main():
    print(" Starting SHAP Analysis (Kernel Method)...")

    # 3.1 Load Test Graph Sequences
    load_path = cfg.SAVE_DIR / "test_graphs.pt"
    if not load_path.exists():
        print(f" File not found: {load_path}")
        sys.exit(1)

    print(f" Loading {load_path}...")
    test_graphs = torch.load(load_path, map_location='cpu', weights_only=False)

    # 3.2 Extract valid labeled node sequences
    tensor_pool = []
    pool_target = 1000

    print("   Extracting valid node sequences...")
    for sequence in test_graphs:
        if len(tensor_pool) * sequence[0].num_nodes > pool_target * 6: break

        target_snap = sequence[-1]
        num_nodes = target_snap.num_nodes // 2
        valid_mask = target_snap.y[:num_nodes] != cfg.LABEL_MASK_VALUE
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) == 0: continue

        seq_feats = []
        for data in sequence:
            x_master = data.x[:num_nodes]
            seq_feats.append(x_master[valid_indices])

        # (Time, Valid_Batch, Feat) -> (Valid_Batch, Time, Feat)
        batch_tensor = torch.stack(seq_feats, dim=0).permute(1, 0, 2)
        tensor_pool.append(batch_tensor)

    if not tensor_pool:
        print(" No valid samples found.")
        return

    full_tensor = torch.cat(tensor_pool, dim=0)
    print(f"   Tensor Pool Shape: {full_tensor.shape}")  # (N, 6, 18)

    # 3.3 Flatten data for KernelExplainer
    X_flat = full_tensor.reshape(full_tensor.shape[0], -1).cpu().numpy()

    # 3.4 Background summarization using K-Means
    print("   Summarizing background data...")
    bg_data = shap.kmeans(X_flat, N_BACKGROUND)

    # 3.5 Subset for explanation
    X_explain = X_flat[:N_EXPLAIN]

    # 3.6 Load Trained Model
    print("   Loading Model...")
    model = DynamicTemporalGNN(
        num_features=cfg.NUM_FEATURES, gat_hidden=cfg.GAT_HIDDEN, gat_heads=cfg.GAT_HEADS,
        gat_out=cfg.GAT_OUT, lstm_hidden=cfg.LSTM_HIDDEN, lstm_layers=cfg.LSTM_LAYERS,
        mlp_hidden=cfg.MLP_HIDDEN, num_classes=cfg.NUM_CLASSES, dropout=cfg.DROPOUT
    ).to(cfg.DEVICE)

    model_path = cfg.CHECKPOINT_DIR / "best_model_final.pt"
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE, weights_only=True))
    model.eval()

    # 3.7 Initialize SHAP wrapper
    wrapper = H2KernelWrapper(model, cfg.DEVICE)

    # 3.8 Run KernelExplainer
    print(f" Running KernelExplainer on {N_EXPLAIN} samples...")
    explainer = shap.KernelExplainer(wrapper, bg_data)
    shap_values = explainer.shap_values(X_explain)

    # KernelExplainer returns a list for outputs (even if just 1 output)
    if isinstance(shap_values, list):
        shap_vals_flat = shap_values[0]  # (N, 108)
    else:
        shap_vals_flat = shap_values

    print(f"   SHAP Output Shape: {shap_vals_flat.shape}")

    # 3.9 Reshape & aggregate SHAP values across time
    num_feats = len(FEATURE_NAMES)
    seq_len = 6

    # Reshape back to (N, Time, Feat)
    shap_vals_3d = shap_vals_flat.reshape(N_EXPLAIN, seq_len, num_feats)

    # Sum across time (Axis 1) -> (N, Feat)
    shap_vals_aggregated = np.sum(shap_vals_3d, axis=1)

    # Get Feature Values (Mean over time) for plotting
    X_explain_3d = X_explain.reshape(N_EXPLAIN, seq_len, num_feats)
    X_explain_aggregated = np.mean(X_explain_3d, axis=1)

    # 3.10 Generate and save figures
    print("📊 Generating Figures...")

    # Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_aggregated, X_explain_aggregated, feature_names=FEATURE_NAMES, show=False)
    plt.title("H2 Feature Importance (Fig 7)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "h2_shap_summary.png", dpi=300)
    plt.close()

    # Dependence Plots
    mean_imp = np.mean(np.abs(shap_vals_aggregated), axis=0)
    top_indices = np.argsort(mean_imp)[::-1][:4]

    for rank, idx in enumerate(top_indices):
        feat_name = FEATURE_NAMES[idx]
        print(f"   Plotting Dependence for #{rank + 1}: {feat_name}")

        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            idx, shap_vals_aggregated, X_explain_aggregated,
            feature_names=FEATURE_NAMES, show=False,
            interaction_index=None
        )
        plt.title(f"H2 Dependence: {feat_name}")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"h2_shap_dep_{feat_name}.png", dpi=300)
        plt.close()

    print(f"H2 Interpretation Complete. Figures saved to {FIGURE_DIR}")

# --- 4. SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    main()