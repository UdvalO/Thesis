# ==============================================================================
# compare_unpaired_with_ci.py
#
# PURPOSE:
#   Perform independent (unpaired) statistical comparison between two models (H1 vs H2)
#   by computing:
#       1. Individual performance metrics (AUC & Accuracy) with 95% Confidence Intervals
#       2. Unpaired statistical tests for significance of differences
#
# STEPS:
#   1. Load predictions from both models (pickle files)
#   2. Compute individual performance metrics:
#       a) ROC-AUC + 95% CI (Hanley & McNeil)
#       b) Accuracy + 95% CI (Binomial Proportion)
#   3. Compare models using independent statistical tests:
#       a) Unpaired DeLong test for AUC difference
#       b) Two-proportion Z-test for accuracy difference
#   4. Print results with p-values and significance markers
#
# INPUTS:
#   - H1 predictions: h1_predictions.pkl
#   - H2 predictions: h2_predictions.pkl
#
# OUTPUTS:
#   - Console summary of performance metrics and statistical significance
# ==============================================================================

# --- 0. DEPENDENCIES ---
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import stats
from pathlib import Path

# --- 1. CONFIGURATION / FILE PATHS ---
H1_PRED_FILE = Path(r"...\H1\graphs_processed\h1_predictions.pkl")
H2_PRED_FILE = Path(r"...\H2\graphs_processed\h2_predictions.pkl")

# --- 2. HELPER FUNCTIONS ---

# 2A. AUC Confidence Interval (Single Model)
def compute_auc_ci(y_true, y_prob, alpha=0.95):
    """
    Computes AUC and 95% confidence interval using Hanley & McNeil method for a SINGLE model.

    Args:
        y_true (array-like): True binary labels
        y_prob (array-like): Predicted probabilities
        alpha (float): Confidence level (default 0.95)

    Returns:
        auc (float): ROC-AUC score
        lower (float): Lower bound of CI
        upper (float): Upper bound of CI
        var (float): Estimated variance of AUC
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    auc = roc_auc_score(y_true, y_prob)
    n1 = float(np.sum(y_true == 1))
    n0 = float(np.sum(y_true == 0))

    # Hanley & McNeil Variance quantities
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)

    # Variance
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)
    se = np.sqrt(max(var, 0.0))

    # CI
    z_score = stats.norm.ppf((1 + alpha) / 2)  # 1.96 for 95%
    lower = auc - z_score * se
    upper = auc + z_score * se

    return auc, max(0.0, lower), min(1.0, upper), var


# 2B. Accuracy Confidence Interval (Single Model)
def compute_acc_ci(y_true, y_prob, alpha=0.95):
    """
    Computes Accuracy and 95% CI for a SINGLE model using binomial proportion.

    Args:
        y_true (array-like): True binary labels
        y_prob (array-like): Predicted probabilities
        alpha (float): Confidence level (default 0.95)

    Returns:
        acc (float): Accuracy score
        lower (float): Lower bound of CI
        upper (float): Upper bound of CI
    """
    y_true = np.asarray(y_true).ravel()
    pred = (np.asarray(y_prob).ravel() > 0.5)

    acc = accuracy_score(y_true, pred)
    n = len(y_true)

    # Standard error for binomial proportion
    se = np.sqrt((acc * (1 - acc)) / n)

    z_score = stats.norm.ppf((1 + alpha) / 2)
    lower = acc - z_score * se
    upper = acc + z_score * se

    return acc, max(0.0, lower), min(1.0, upper)


# 2C. Independent DeLong Test (Comparison)
def independent_delong_test(auc1, var1, auc2, var2):
    """
        Performs unpaired (independent) test for difference in AUC between two models.

        Args:
            auc1, auc2 (float): Model AUCs
            var1, var2 (float): Estimated AUC variances

        Returns:
            z (float): Test statistic
            p_value (float): Two-tailed p-value
        """
    se_diff = np.sqrt(var1 + var2)
    z = (auc1 - auc2) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value


# 2D. HELPER: 2-Proportion Z-Test (Comparison)
def two_prop_z_test(acc1, n1, acc2, n2):
    """
        Performs two-proportion Z-test for difference in accuracy between two models.

        Args:
            acc1, acc2 (float): Model accuracies
            n1, n2 (int): Sample sizes

        Returns:
            z (float): Test statistic
            p_value (float): Two-tailed p-value
        """
    x1 = acc1 * n1
    x2 = acc2 * n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (acc1 - acc2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value

# --- 3. MAIN EXECUTION ---
def main():
    print("Starting Unpaired Comparison with Confidence Intervals")

    # --- 3A. Load Predictions ---
    print(f"Loading H1...")
    with open(H1_PRED_FILE, "rb") as f:
        h1 = pickle.load(f)
    print(f"Loading H2...")
    with open(H2_PRED_FILE, "rb") as f:
        h2 = pickle.load(f)

    y1, p1 = h1['y_true'], h1['y_prob']
    y2, p2 = h2['y_true'], h2['y_prob']
    n1, n2 = len(y1), len(y2)

    print(f"\n Sample Sizes: H1={n1:,}, H2={n2:,}")

    # --- 3B. Individual Performance Metrics (95% CI) ---
    print("\n" + "=" * 60)
    print("INDIVIDUAL PERFORMANCE (95% CI)")
    print("=" * 60)

    # AUC
    auc1, l1, u1, var1 = compute_auc_ci(y1, p1)
    auc2, l2, u2, var2 = compute_auc_ci(y2, p2)

    print(f"AUC H1: {auc1:.4f} [{l1:.4f} - {u1:.4f}]")
    print(f"AUC H2: {auc2:.4f} [{l2:.4f} - {u2:.4f}]")

    # Accuracy
    acc1, al1, au1 = compute_acc_ci(y1, p1)
    acc2, al2, au2 = compute_acc_ci(y2, p2)

    print(f"ACC H1: {acc1:.4f} [{al1:.4f} - {au1:.4f}]")
    print(f"ACC H2: {acc2:.4f} [{al2:.4f} - {au2:.4f}]")

    # --- 3C. Statistical Comparison (Unpaired / Conservative) ---
    print("\n" + "=" * 60)
    print("🔬 STATISTICAL SIGNIFICANCE (Unpaired / Conservative)")
    print("=" * 60)

    # AUC Difference Test
    z_auc, p_auc = independent_delong_test(auc1, var1, auc2, var2)
    print(f"1. AUC Difference (H2 - H1 = {auc2 - auc1:.4f})")
    print(f"   p-value: {p_auc:.4e}")
    if p_auc < 0.05:
        print("Significant")
    else:
        print("Not Significant")

    # Accuracy Difference Test
    z_acc, p_acc = two_prop_z_test(acc1, n1, acc2, n2)
    print(f"\n2. Accuracy Difference (H2 - H1 = {acc2 - acc1:.4f})")
    print(f"   p-value: {p_acc:.4e}")
    if p_acc < 0.05:
        print("Significant")
    else:
        print("Not Significant")

# --- 4. RUN SCRIPT ---
if __name__ == "__main__":
    main()