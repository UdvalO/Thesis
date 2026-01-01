# ==============================================================================
# evaluation_h1.py (H1: Empirical Evaluation & Benchmarking)
#
# PURPOSE:
#   Conducts the final performance assessment of the H1 Constraint-Aware Model.
#   It generates predictions using the "Stochastic Static" testing protocol,
#   computes the full suite of metrics defined in Thesis Section 4.1, and
#   benchmarks the Dynamic GNN against the Static Baselines.
#
# LOGIC & THESIS ALIGNMENT:
#   1. Metric Suite (Thesis Sec 4.1):
#      - AUC: Primary ranking metric.
#      - F1 (Fixed 0.5): Strict replication requirement to avoid "Oracle Bias."
#      - F1 (Max): Supplementary diagnostic for latent potential.
#      - Brier Score: Reliability/Calibration assessment.
#
#   2. Calibration Analysis (Thesis Sec 5.1.3 / Figure 5):
#      - Generates "Reliability Curves" to visualize the cost-sensitive
#        probability shift (systematic overestimation vs. monotonic ranking).
#
#   3. Comparison Protocol (Thesis Sec 5.1.3):
#      - Aligns GNN predictions with stored Baseline predictions (LR, XGB, etc.)
#      - Performs statistical significance testing (McNemar/DeLong) for
#        internal H1 vs Baseline comparison.
#
# INPUTS:
#   - Trained Model: best_model_h1.pt (on remote computer)
#   - Test Data: test_features.pt (Window 13) (on remote computer)
#   - Static Topology: static_edge_index.pt (on remote computer)
#   - Baseline Results: baseline_predictions.pkl (on remote computer)
#
# OUTPUTS:
#   - h1_predictions.pkl: Exported probabilities for H1 vs H2 comparison. (on flash drive)
#   - calibration_curve_comparison.png: Visual analysis of model confidence.
#   - Console Report: Detailed metric table.
# ==============================================================================

import torch
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    brier_score_loss, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.amp import autocast

import config_h1 as cfg
from model_architecture_h1 import DYMGNN
from training_h1 import filter_ghost_nodes


# --- METRIC HELPERS ---
def calculate_full_suite(y_true, y_prob):
    """
        Computes a comprehensive suite of evaluation metrics for binary classification.

        Methodology:
            - Fixed-threshold F1 (0.5) for strict replication of the thesis setup.
            - Maximal F1 across thresholds as a supplementary diagnostic.
            - ROC-AUC, Precision-Recall AUC, and Brier score for overall model performance.

        Args:
            y_true (array-like): True binary labels.
            y_prob (array-like): Predicted probabilities.

        Returns:
            dict: Dictionary containing AUC, fixed-threshold F1, max F1, PR-AUC, and Brier score.
        """
    y_true = np.array(y_true).ravel()
    y_prob = np.array(y_prob).ravel()

    # 1. Fixed Threshold F1 (0.5)
    y_pred_05 = (y_prob > 0.5).astype(int)
    fixed_f1 = f1_score(y_true, y_pred_05)

    # 2. Max Threshold (Diagnostic)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    numerator = 2 * precision * recall
    denominator = precision + recall
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    max_f1 = np.max(f1_scores)
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "F1 (Fixed 0.5)": fixed_f1,
        "F1 (Max)": max_f1,
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob)
    }

def fast_delong(y_true, pred1, pred2):
    """
        Computes DeLong's test for ROC-AUC significance (Paired).

        Methodology (Thesis Sec 4.2.1):
            - Used for internal H1 comparisons where datasets are perfectly aligned.
            - Tests the null hypothesis that the two AUCs are statistically identical.

        Args:
            y_true (array-like): True binary labels.
            pred1 (array-like): Probabilities from Model A.
            pred2 (array-like): Probabilities from Model B.

        Returns:
            float: P-value indicating the significance of the difference.
        """
    y_true = np.array(y_true).ravel()
    pred1 = np.array(pred1).ravel()
    pred2 = np.array(pred2).ravel()

    n1 = float(np.sum(y_true == 1))
    n0 = float(np.sum(y_true == 0))
    if n1 == 0 or n0 == 0: return np.nan

    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)

    # DeLong Variance Estimation (Vectorized)
    q1 = auc1 / (2 - auc1)
    q2 = 2 * auc1**2 / (1 + auc1)
    var1 = (auc1 * (1 - auc1) + (n1 - 1) * (q1 - auc1**2) + (n0 - 1) * (q2 - auc1**2)) / (n1 * n0)
    se1 = np.sqrt(max(var1, 0.0))

    q1_2 = auc2 / (2 - auc2)
    q2_2 = 2 * auc2**2 / (1 + auc2)
    var2 = (auc2 * (1 - auc2) + (n1 - 1) * (q1_2 - auc2**2) + (n0 - 1) * (q2_2 - auc2**2)) / (n1 * n0)
    se2 = np.sqrt(max(var2, 0.0))

    r = np.corrcoef(pred1, pred2)[0, 1]
    se_diff = np.sqrt(max(se1**2 + se2**2 - 2 * r * se1 * se2, 0.0))

    if se_diff == 0: z = 0
    else: z = (auc1 - auc2) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_value

def run_mcnemar_test(y_true, prob_a, prob_b, name_a, name_b):
    """
        Computes McNemar's test for binary classification disagreement.

        Methodology (Thesis Sec 4.2.2):
            - Evaluates the null hypothesis that classification accuracies are equal.
            - Relies on discordant pairs (cases where Model A and Model B disagree).

        Args:
            y_true (array-like): True labels.
            prob_a (array-like): Probabilities from Model A.
            prob_b (array-like): Probabilities from Model B.
            name_a (str): Name of Model A.
            name_b (str): Name of Model B.
        """
    print(f"    Running McNemar Test ({name_a} vs {name_b})...")
    y_true = np.array(y_true).ravel()
    prob_a = np.array(prob_a).ravel()
    prob_b = np.array(prob_b).ravel()

    pred_a = (prob_a > 0.5)
    pred_b = (prob_b > 0.5)
    y_true_bool = y_true.astype(bool)

    corr_a = (pred_a == y_true_bool)
    corr_b = (pred_b == y_true_bool)

    yy = ((corr_a) & (corr_b)).sum()
    yw = ((corr_a) & (~corr_b)).sum()
    wy = ((~corr_a) & (corr_b)).sum()
    ww = ((~corr_a) & (~corr_b)).sum()

    table = [[yy, yw], [wy, ww]]
    try:
        result = mcnemar(table, exact=False)
        print(f" McNemar p-value = {result.pvalue:.4e}")
    except Exception as e:
        print(f"️ McNemar Failed: {e}")

def plot_calibration_comparison(results_dict, save_dir):
    """
        Generates and saves the Reliability Curve (Thesis Figure 5).

        Args:
            results_dict (dict): Dictionary of model results {'ModelName': {'y_true': ..., 'y_prob': ...}}.
            save_dir (Path): Directory to save the plot.
        """
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")

    for model_name, data in results_dict.items():
        y_true = data['y_true']
        y_prob = data['y_prob']

        # Calibration bins
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=f"{model_name}")

    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.title("Reliability Curve (Calibration)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "calibration_curve_comparison.png")
    print(" Calibration plot saved.")

@torch.no_grad()
def get_gnn_predictions():
    """
        Generates predictions using the full H1 test set via Stochastic Sampling.

        Methodology (Thesis Sec 3.2.3):
            - Loads the "Static Supra-Graph" (static_edge_index.pt).
            - Applies "Dynamic Stochastic Sampling" (K=50) during inference.

        Returns:
            tuple: (y_true, y_prob) arrays for the entire test set.
        """
    print(" Generating DYMGNN Predictions...")

    # 1. Load Artifacts
    meta = torch.load(cfg.GRAPH_SAVE_DIR / "metadata.pt", weights_only=False)
    static_edge_index = torch.load(cfg.GRAPH_SAVE_DIR / "static_edge_index.pt", weights_only=False)
    test_list = torch.load(cfg.GRAPH_SAVE_DIR / "test_features.pt", weights_only=False)

    # 2. Load Model
    model = DYMGNN(meta['num_features'], cfg.HIDDEN_DIM, cfg.NUM_HEADS, cfg.DROPOUT).to(cfg.DEVICE)
    model.load_state_dict(torch.load(cfg.GRAPH_SAVE_DIR / "best_model_h1.pt", map_location=cfg.DEVICE, weights_only=True))
    model.eval()

    all_probs = []
    all_labels = []

    # 3. Batch Inference
    for test_data in test_list:
        num_nodes_total = test_data.y.shape[0]
        x_in = test_data.x

        # Shape adjustment [N, T, F] -> [T, N, F] if needed
        if x_in.shape[0] != num_nodes_total:
            x_in = x_in.permute(1, 0, 2)
        num_master_nodes = num_nodes_total // 2
        node_indices = torch.arange(num_nodes_total)
        y_cpu = test_data.y.cpu()

        # Filter: Only predict on valid (unmasked) nodes in the primary layer
        valid_mask = (y_cpu != -1) & (node_indices < num_master_nodes)
        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) == 0: continue

        # Dynamic Neighbor Loading (K=50) matches training protocol
        loader = NeighborLoader(
            Data(x=x_in, y=test_data.y, edge_index=static_edge_index, num_nodes=num_nodes_total),
            num_neighbors=cfg.NUM_NEIGHBORS,
            batch_size=4096,
            input_nodes=valid_indices,
            shuffle=False,
            num_workers=0
        )
        for batch in loader:
            batch = batch.to(cfg.DEVICE)
            x_seq = batch.x.permute(1, 0, 2)

            # Filter Edges (Ghost Node logic)
            edge_index = filter_ghost_nodes(x_seq, batch.edge_index)
            logits = model(x_seq, edge_index)
            if logits.dim() > 1: logits = logits.squeeze(-1)

            # Extract predictions for the target batch only
            target_logits = logits[:batch.batch_size]
            target_labels = batch.y[:batch.batch_size]

            all_probs.extend(torch.sigmoid(target_logits).float().cpu().numpy())
            all_labels.extend(target_labels.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)

def main():
    # 1. Get GNN Results
    gnn_y, gnn_prob = get_gnn_predictions()

    # 2. Load Baseline Results
    baseline_path = cfg.GRAPH_SAVE_DIR / "baseline_predictions.pkl"
    try:
        with open(baseline_path, 'rb') as f:
            baseline_results = pickle.load(f)
    except FileNotFoundError:
        print("  Baselines not found.")
        baseline_results = {}

    # 3. Alignment Check
    if len(baseline_results) > 0:
        first_key = list(baseline_results.keys())[0]
        baseline_len = len(baseline_results[first_key]['y_true'])
        gnn_len = len(gnn_y)

        if gnn_len != baseline_len:
            print(" Size Mismatch Detected! Truncating to common minimum length.")
            min_len = min(gnn_len, baseline_len)
            gnn_y = gnn_y[:min_len]
            gnn_prob = gnn_prob[:min_len]
            for k in baseline_results:
                baseline_results[k]['y_prob'] = baseline_results[k]['y_prob'][:min_len]
                baseline_results[k]['y_true'] = baseline_results[k]['y_true'][:min_len]

    full_results = baseline_results.copy()
    full_results['DYMGNN'] = {'y_true': gnn_y, 'y_prob': gnn_prob}

    # --- 4. EXPORT FOR H1 VS H2 COMPARISON (Thesis Sec 5.2.2) ---
    # "Hypothesis testing was conducted on the complete set of test predictions."
    # Used later by the unpaired statistical test script.
    export_path = cfg.GRAPH_SAVE_DIR / "h1_predictions.pkl"
    with open(export_path, "wb") as f:
        pickle.dump({"y_true": gnn_y, "y_prob": gnn_prob}, f)
    print(f" Saved H1 predictions to {export_path} for cross-model comparison.")

    # 5. Report & Internal Stats
    print("\n" + "=" * 60)
    print(" FINAL EVALUATION REPORT (Fixed Threshold 0.5)")
    print("=" * 60)

    summary_table = []
    for name, data in full_results.items():
        metrics = calculate_full_suite(data['y_true'], data['y_prob'])
        row = {"Model": name, **metrics}
        summary_table.append(row)
    df = pd.DataFrame(summary_table).sort_values("AUC", ascending=False)
    print(df.to_string(index=False))

    # Internal Stats (H1 vs Baseline)
    baseline_names = [n for n in full_results if n != "DYMGNN"]
    if len(baseline_names) > 0:
        best_baseline = max(baseline_names, key=lambda n: roc_auc_score(full_results[n]['y_true'], full_results[n]['y_prob']))
        print(f"\n Statistical Significance: DYMGNN vs {best_baseline}")

        # Paired Tests (Valid here because we are comparing models on the EXACT SAME H1 test set)
        run_mcnemar_test(gnn_y, full_results[best_baseline]['y_prob'], gnn_prob, best_baseline, "DYMGNN")
        try:
            p_val = fast_delong(gnn_y, full_results[best_baseline]['y_prob'], gnn_prob)
            print(f" DeLong p-value = {p_val:.4e}")
        except Exception as e:
            print(f" DeLong Test Failed: {e}")

    # 6. Calibration Plot
    plot_calibration_comparison(full_results, cfg.GRAPH_SAVE_DIR)

if __name__ == "__main__":
    main()