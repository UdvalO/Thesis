# ==============================================================================
# interpretability_h1.py (H1)
#
# PURPOSE:
#   Provide post-hoc interpretability for the trained H1 DYMGNN model using
#   SHAP (KernelExplainer).
#
# STEPS:
#   1. Load trained DYMGNN model and test graph data.
#   2. Align feature and label dimensions and filter unlabeled nodes.
#   3. Flatten temporal node features for SHAP compatibility.
#   4. Construct a background distribution using K-Means summarization.
#   5. Wrap the DYMGNN model to match the SHAP KernelExplainer interface.
#   6. Compute SHAP values for a subset of test nodes.
#   7. Aggregate SHAP values across time for feature-level interpretation.
#   8. Generate and save summary and dependence plots.
#
# INPUTS:
#   - Test graph features: test_features.pt (on remote computer)
#   - Trained DYMGNN model: best_model_h1.pt (on remote computer)
#   - Metadata (optional): metadata.pt (on flash drive)
#   - Configuration: config_h1.py (on flash drive)
#
# OUTPUTS:
#   - Global SHAP summary plot: h1_shap_summary.png (on flash drive)
#   - Feature dependence plots: h1_shap_dep_<feature>.png (on flash drive)
# ==============================================================================

# --- 0. DEPENDENCIES ---
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import config_h1 as cfg
from model_architecture_h1 import DYMGNN
import sys

# --- 1. CONFIGURATION ---
N_BACKGROUND = 50   # Background samples for SHAP baseline
N_EXPLAIN = 100     # Number of samples to explain
FIGURE_DIR = cfg.GRAPH_SAVE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True, parents=True)

FEATURE_NAMES = cfg.NODE_FEATURES

# --- 2. SHAP MODEL WRAPPER ---
class H1KernelWrapper:
    """
    Adapter class that wraps the H1 DYMGNN model so it can be used with
    SHAP's KernelExplainer.

    The wrapper:
        - Reconstructs temporal tensors from flattened inputs
        - Permutes dimensions to match the DYMGNN input format
        - Builds a minimal self-loop graph
        - Returns node-level prediction probabilities
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.num_feats = len(FEATURE_NAMES)

    def __call__(self, x_flat):
        """
                Forward function used by KernelExplainer.

                Args:
                    x_flat (np.ndarray): Flattened node features
                                        Shape: (Batch, Time * Features)

                Returns:
                    np.ndarray: Prediction probabilities
                                Shape: (Batch, Time * Features)
                """

        # --- Restore temporal structure ---
        batch_size = x_flat.shape[0]
        seq_len = x_flat.shape[1] // self.num_feats

        x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=self.device)
        x_reshaped = x_tensor.view(batch_size, seq_len, self.num_feats)

        # --- Permute for DYMGNN ---
        # Expected shape: (Time, Batch, Features)
        x_model_input = x_reshaped.permute(1, 0, 2)

        # --- Construct simple self-loop graph ---
        node_indices = torch.arange(batch_size, device=self.device)
        edge_index = torch.stack([node_indices, node_indices], dim=0)

        # --- Forward pass ---
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_model_input, edge_index)
            probs = torch.sigmoid(logits).view(-1)

        return probs.cpu().numpy()

# --- 3. MAIN INTERPRETABILITY PIPELINE ---
def main():
    print("Starting H1 SHAP Analysis (Kernel Method)...")

    # 3.1 Load Test Data
    load_path = cfg.GRAPH_SAVE_DIR / "test_features.pt"
    if not load_path.exists():
        print(f" File not found: {load_path}")
        sys.exit(1)

    print(f" Loading {load_path}...")
    data_list = torch.load(load_path, map_location='cpu', weights_only=False)

    # Use the first test window
    data = data_list[0]
    x = data.x
    y = data.y

    print(f"    Raw x Shape: {x.shape}")
    print(f"    Raw y Shape: {y.shape}")

    # 3.2 Align Feature and Label Dimensions
    if x.shape[0] == y.shape[0]:
        pass
    elif x.shape[1] == y.shape[0]:
        print("    🔄 Detected (Time, Node, Feat). Permuting to (Node, Time, Feat)...")
        x = x.permute(1, 0, 2)
    else:
        print(f" CRITICAL ERROR: Cannot align x {x.shape} with labels {y.shape}")
        sys.exit(1)

    # 3.3 Select Labeled Nodes Only
    valid_mask = y != cfg.LABEL_MASK_VALUE
    X_valid = x[valid_mask]

    if len(X_valid) == 0:
        print(" No valid labeled samples found.")
        sys.exit(1)

    print(f"    Valid Samples Pool: {X_valid.shape}")

    # 3.4 Flatten Temporal Features
    # (Nodes, Time, Features) -> (Nodes, Time * Features)
    X_flat = X_valid.reshape(X_valid.shape[0], -1).numpy()

    # Background distribution for SHAP
    print("    Summarizing background data (K-Means)...")
    bg_data = shap.kmeans(X_flat, N_BACKGROUND)

    # Samples to explain
    X_explain = X_flat[:N_EXPLAIN]
    print(f"    Explaining {X_explain.shape[0]} samples.")

    # 3.5 Load Trained Model
    print("    Loading Model...")
    meta_path = cfg.GRAPH_SAVE_DIR / "metadata.pt"
    if meta_path.exists():
        meta = torch.load(meta_path, weights_only=False)
        num_features = meta['num_features']
    else:
        num_features = len(FEATURE_NAMES)

    model = DYMGNN(
        num_features=num_features,
        hidden_dim=cfg.HIDDEN_DIM,
        num_heads=cfg.NUM_HEADS,
        dropout=cfg.DROPOUT
    ).to(cfg.DEVICE)

    model_path = cfg.GRAPH_SAVE_DIR / "best_model_h1.pt"
    if not model_path.exists():
        print(f" Model not found at {model_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE, weights_only=True))
    model.eval()

    # 3.6 Initialize SHAP Explainer
    wrapper = H1KernelWrapper(model, cfg.DEVICE)

    print(f" Running KernelExplainer...")
    explainer = shap.KernelExplainer(wrapper, bg_data)

    shap_values = explainer.shap_values(X_explain, nsamples='auto', silent=False)

    if isinstance(shap_values, list):
        shap_vals_flat = shap_values[0]
    else:
        shap_vals_flat = shap_values

    print(f"    SHAP Output Shape: {shap_vals_flat.shape}")

    # 3.7 Aggregate SHAP Values Across Time
    num_feats = len(FEATURE_NAMES)
    seq_len = shap_vals_flat.shape[1] // num_feats

    # Reshape (N, Time, Feat) -> Sum Time -> (N, Feat)
    shap_vals_3d = shap_vals_flat.reshape(N_EXPLAIN, seq_len, num_feats)
    shap_vals_aggregated = np.sum(shap_vals_3d, axis=1)

    # Get Feature Values (Mean over time)
    X_explain_3d = X_explain.reshape(N_EXPLAIN, seq_len, num_feats)
    X_explain_aggregated = np.mean(X_explain_3d, axis=1)

    # 3.8 Generate Plots
    print("Generating Figures...")

    # Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_aggregated, X_explain_aggregated, feature_names=FEATURE_NAMES, show=False)
    plt.title("H1 Feature Importance (Kernel Method)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "h1_shap_summary.png", dpi=300)
    plt.close()

    # Dependence Plots for Top Features
    mean_imp = np.mean(np.abs(shap_vals_aggregated), axis=0)
    top_indices = np.argsort(mean_imp)[::-1][:4]

    for rank, idx in enumerate(top_indices):
        feat_name = FEATURE_NAMES[idx]
        print(f"    Plotting Dependence for #{rank + 1}: {feat_name}")

        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            idx, shap_vals_aggregated, X_explain_aggregated,
            feature_names=FEATURE_NAMES, show=False,
            interaction_index=None
        )
        plt.title(f"H1 Dependence: {feat_name}")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"h1_shap_dep_{feat_name}.png", dpi=300)
        plt.close()

    print(f" H1 Interpretation Complete. Figures saved to {FIGURE_DIR}")

# --- 4. SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    main()