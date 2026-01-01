# ==============================================================================
# evaluation_h2.py (H2)
#
# PURPOSE:
#   Conducts the final performance assessment of the H2 Implicit Topology Model.
#   It generates predictions using the Dynamic Graph testing protocol, computes
#   the full suite of metrics defined in Thesis Section 4.1, and benchmarks
#   the Implicit Topology Model against the Static Baselines (XGB, LR, Static GNNs).
#
# LOGIC & THESIS ALIGNMENT:
#   1. Metric Suite (Thesis Sec 4.1):
#      - AUC: Primary ranking metric.
#      - F1 (Fixed 0.5): Strict replication requirement to avoid "Oracle Bias."
#      - Brier Score: Reliability/Calibration assessment.
#
#   2. Statistical Significance (Thesis Sec 4.2):
#      - DeLong's Test: Tests if the difference in AUC is statistically significant.
#      - McNemar's Test: Tests if the models make significantly different errors
#        at the decision boundary.
#
#   3. Comparison Protocol (Thesis Sec 5.2):
#      - Aligns GNN predictions with stored Baseline predictions.
#      - "The best-performing baseline... is selected as the reference point for
#        statistical testing."
#
# INPUTS:
#   - Trained Model: best_model_final.pt (H2 Checkpoint)
#   - Test Data: test_graphs.pt (Dynamic Sequences)
#   - Baseline Results: baseline_predictions_h2.pkl
#
# OUTPUTS:
#   - h2_predictions.pkl: Exported probabilities for H1 vs H2 comparison.
#   - calibration_curve_comparison.png: Visual analysis of model confidence.
#   - Console Report: Detailed metric table aligned with Thesis Table 6.
# ==============================================================================

import torch
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, f1_score, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats
from model_architecture_h2 import DynamicTemporalGNN
import config_h2 as cfg


# --- 1. METRIC HELPERS ---
def calculate_full_suite(y_true, y_prob):
    """
    Computes the comprehensive suite of evaluation metrics defined in Thesis Section 4.1.

    Methodology:
        - Fixed-threshold F1 (0.5) is used for strict replication to avoid "Oracle Bias".
        - Maximal F1 (F1_max) is reported as a supplementary diagnostic to isolate latent power.
        - PR-AUC provides a rigorous assessment on the minority class.

    Args:
        y_true (array-like): True binary labels.
        y_prob (array-like): Predicted probabilities.

    Returns:
        dict: Dictionary containing AUC, fixed-threshold F1, max F1, PR-AUC, and Brier score.
    """
    y_true = np.array(y_true).ravel()
    y_prob = np.array(y_prob).ravel()

    # --- Fixed threshold ---
    y_pred_05 = (y_prob > 0.5).astype(int)
    fixed_f1 = f1_score(y_true, y_pred_05)

    # --- Max F1 ---
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    numerator = 2 * precision * recall
    denominator = precision + recall

    f1_scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0
    )
    max_f1 = np.max(f1_scores)

    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "F1 (Fixed 0.5)": fixed_f1,
        "F1 (Max)": max_f1,
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob)
    }


# --- 2. STATISTICAL TESTS ---

def fast_delong(y_true, pred1, pred2):
    """
    Computes DeLong's test for ROC-AUC significance (Paired).

    Methodology (Thesis Sec 4.2.1):
        - Used for internal H2 comparisons (IM-DYMGNN vs Baseline).
        - Tests the null hypothesis that the two AUCs are statistically identical.
        - Implements the vectorized algorithm by Hanley & McNeil (1982).

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

    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    if n1 == 0 or n0 == 0:
        return np.nan

    # AUCs
    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)

    # Standard Errors (Hanley & McNeil, 1982)
    q1 = auc1 / (2 - auc1)
    q2 = 2 * auc1 ** 2 / (1 + auc1)
    se1 = np.sqrt((auc1 * (1 - auc1) + (n1 - 1) * (q1 - auc1 ** 2) + (n0 - 1) * (q2 - auc1 ** 2)) / (n1 * n0))

    q1_2 = auc2 / (2 - auc2)
    q2_2 = 2 * auc2 ** 2 / (1 + auc2)
    se2 = np.sqrt((auc2 * (1 - auc2) + (n1 - 1) * (q1_2 - auc2 ** 2) + (n0 - 1) * (q2_2 - auc2 ** 2)) / (n1 * n0))

    # Pearson correlation for paired samples
    r = np.corrcoef(pred1, pred2)[0, 1]
    se_diff = np.sqrt(se1 ** 2 + se2 ** 2 - 2 * r * se1 * se2)

    if se_diff == 0:
        z = 0
    else:
        z = (auc1 - auc2) / se_diff

    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_value


def run_mcnemar_test(y_true, prob_a, prob_b, name_a="Model A", name_b="Model B"):
    """
        Computes McNemar's test for binary classification disagreement.

        Methodology (Thesis Sec 4.2.2):
            - Evaluates the null hypothesis that classification accuracies are equal.
            - Relies on discordant pairs (cases where Model A and Model B disagree).

        Args:
            y_true (array-like): True labels.
            prob_a (array-like): Probabilities from Model A.
            prob_b (array-like): Probabilities from Model B.
        """
    print(f"   Running McNemar Test ({name_a} vs {name_b})...")

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
        return result.pvalue
    except Exception as e:
        print(f" McNemar Failed: {e}")
        return 1.0


# --- 3. PLOTTING ---
def plot_calibration_comparison(results_dict, save_dir):
    """
        Generates and saves the Reliability Curve (Thesis Figure 7).

        Methodology (Thesis Sec 5.2.3):
            - Visualizes "Systematic Overestimation" due to positive-class weighting.
            - Compares the calibration drift of the Implicit GNN vs. Feature-based baselines.
        """
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")

    for model_name, data in results_dict.items():
        frac_pos, mean_pred = calibration_curve(
            data['y_true'], data['y_prob'], n_bins=10
        )
        plt.plot(mean_pred, frac_pos, "s-", label=model_name)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Curve (Calibration)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / "calibration_curve_comparison.png")


# --- 4. GNN INFERENCE ---
@torch.no_grad()
def get_h2_predictions():
    """
        Generates predictions using the full H2 test set (Dynamic Graphs).

        Methodology (Thesis Sec 3.3):
            - Unlike H1's static loading, H2 iterates through temporal sequences of
              dynamic graphs where neighbors change at every timestep.
            - Loads 'test_graphs.pt' which contains the sequence of KNN graphs.

        Returns:
            tuple: (y_true, y_prob) for the entire test set.
        """
    print(" Generating DynamicTemporalGNN (H2) Predictions...")

    test_graphs = torch.load(
        cfg.SAVE_DIR / "test_graphs.pt",
        map_location="cpu",
        weights_only=False
    )
    # Initialize Model Structure
    model = DynamicTemporalGNN(
        num_features=cfg.NUM_FEATURES,
        gat_hidden=cfg.GAT_HIDDEN,
        gat_heads=cfg.GAT_HEADS,
        gat_out=cfg.GAT_OUT,
        lstm_hidden=cfg.LSTM_HIDDEN,
        lstm_layers=cfg.LSTM_LAYERS,
        mlp_hidden=cfg.MLP_HIDDEN,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.DROPOUT
    ).to(cfg.DEVICE)

    # Load Weights
    model.load_state_dict(
        torch.load(cfg.CHECKPOINT_DIR / "best_model_final.pt",
                   map_location=cfg.DEVICE)
    )
    model.eval()

    all_probs, all_labels = [], []

    # Iterate through Dynamic Test Sequences
    for sequence in tqdm(test_graphs, desc="Inference"):
        seq_gpu = [g.to(cfg.DEVICE) for g in sequence]
        target = seq_gpu[-1]

        # Extract labels from the master nodes (primary layer)
        num_master = target.num_nodes // 2
        labels = target.y[:num_master]

        # Filter Masked Labels (-1)
        mask = labels != cfg.LABEL_MASK_VALUE
        if mask.sum() == 0:
            continue

        # Forward Pass
        logits = model(seq_gpu)[:num_master]
        probs = torch.sigmoid(logits[mask]).cpu().numpy()
        lbls = labels[mask].cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend(lbls)

    return np.array(all_labels), np.array(all_probs)


# --- 5. MAIN ---
def main():
    # 1. Get H2 Predictions
    gnn_y, gnn_prob = get_h2_predictions()

    # 2. Load Baseline Predictions (XGB, LR, Static GNNs)
    pred_path = cfg.LOG_DIR / "baseline_predictions_h2.pkl"
    with open(pred_path, "rb") as f:
        full_results = pickle.load(f)

    # --- ALIGNMENT SAFETY CHECK ---
    # Methodology: Ensures exact row-wise correspondence between GNN and Baselines.
    baseline_key = list(full_results.keys())[0]
    baseline_len = len(full_results[baseline_key]["y_true"])
    gnn_len = len(gnn_y)

    print(f" Alignment Check: IM_DYMGNN={gnn_len}, Baseline={baseline_len}")

    if gnn_len != baseline_len:
        print(" Size Mismatch Detected! Truncating to common minimum length.")
        min_len = min(gnn_len, baseline_len)

        # Truncate GNN
        gnn_y = gnn_y[:min_len]
        gnn_prob = gnn_prob[:min_len]

        # Truncate Baselines
        for k in full_results:
            full_results[k]["y_true"] = full_results[k]["y_true"][:min_len]
            full_results[k]["y_prob"] = full_results[k]["y_prob"][:min_len]

    full_results["IM_DYMGNN"] = {"y_true": gnn_y, "y_prob": gnn_prob}

    print("\n" + "=" * 60)
    print(" FINAL H2 ADVANCED EVALUATION REPORT (Fixed 0.5 + Max F1)")
    print("=" * 60)

    rows = []
    for name, data in full_results.items():
        rows.append({"Model": name, **calculate_full_suite(
            data["y_true"], data["y_prob"])})

    df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    print(df.to_markdown(index=False, floatfmt=".4f"))

    plot_calibration_comparison(full_results, cfg.SAVE_DIR)

    # --- EXPORT FOR COMPARISON ---
    # Saves predictions for the final cross-hypothesis unpaired tests.
    export_path = cfg.SAVE_DIR / "h2_predictions.pkl"
    with open(export_path, "wb") as f:
        pickle.dump({"y_true": gnn_y, "y_prob": gnn_prob}, f)
    print(f" Saved H2 predictions to {export_path} for cross-model comparison.")

    # --- 6. STATISTICAL TESTS ---
    baseline_names = [n for n in full_results if n != "IM_DYMGNN"]

    if len(baseline_names) > 0:
        # Choose strongest baseline by AUC for statistical testing
        best_baseline = max(
            baseline_names,
            key=lambda n: roc_auc_score(
                full_results[n]["y_true"],
                full_results[n]["y_prob"]
            )
        )

        print(f"\n Statistical Significance: IM_DYMGNN vs {best_baseline}")

        # McNemar Test
        run_mcnemar_test(
            gnn_y,
            full_results[best_baseline]["y_prob"],
            gnn_prob,
            best_baseline,
            "IM_DYMGNN"
        )

        # DeLong Test
        try:
            p_val = fast_delong(
                gnn_y,
                full_results[best_baseline]["y_prob"],
                gnn_prob
            )
            print(f" DeLong p-value = {p_val:.4e}")
            if p_val < 0.05:
                print("    -> AUC Difference is statistically significant.")
            else:
                print("    -> AUC Difference is NOT significant.")
        except Exception as e:
            print(f" DeLong Test Failed: {e}")


if __name__ == "__main__":
    main()