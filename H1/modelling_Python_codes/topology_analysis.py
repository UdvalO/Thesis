# ==============================================================================
# topology_analysis.py (H1)
#
# PURPOSE:
#   Analyze and validate the stochastic block graph construction used in H1.
#   The script evaluates graph size, node degree distributions, and memory
#   implications in order to justify topology design choices and recommend
#   suitable NeighborLoader sampling parameters.
#
#   Specifically, this script:
#     - Analyzes group sizes in the geographical and lender layers
#     - Estimates the total number of edges induced by stochastic blocking
#     - Computes per-node degree statistics
#     - Recommends an appropriate NUM_NEIGHBORS value for training
#     - Estimates approximate VRAM requirements for model training
#
# STEPS:
#   1. Load final H1 dataset and deduplicate to unique loans (nodes).
#   2. Analyze group sizes for each relational layer.
#   3. Estimate total edge counts per layer under stochastic caps.
#   4. Compute per-node degree statistics in the supra-graph.
#   5. Recommend NeighborLoader sampling size based on degree percentiles.
#   6. Estimate total edge count and approximate GPU memory usage.
#
# INPUTS:
#   - final_features_h1.rds
#   - config_h1.py
#
# OUTPUTS:
#   - Console Output (Diagnostics, Degree Stats, VRAM Estimates)
# ==============================================================================

# --- 0. DEPENDENCIES ---
import pandas as pd
import numpy as np
import pyreadr
import config_h1 as cfg
import gc

# --- 1. CONFIGURATION ---
TEST_GEO_CAP = 1_000_000      # just to make sure all geo nodes are included, using high cap
TEST_LENDER_CAP = 1_000_000   # just to make sure all lender nodes are included,using high cap

# --- 2. GROUP SIZE ANALYSIS ---
def analyze_layer_groups(df_unique, group_col, cap, layer_name):
    """
    Analyze group size statistics for a given relational layer and estimate
    the number of edges introduced by stochastic blocking.

    Args:
        df_unique (pd.DataFrame): Unique loan-level dataset
        group_col (str): Column defining the group (e.g., Geo_Key, Lender_Key)
        cap (int): Maximum neighbors per node within this layer
        layer_name (str): Human-readable layer name

    Returns:
        int: Estimated number of directed edges in this layer
    """
    print(f"\n{'=' * 20} ANALYZING {layer_name} (Cap: {cap}) {'=' * 20}")

    # 1. Group Size Distribution
    group_counts = df_unique.groupby(group_col).size().reset_index(name='count')
    sizes = group_counts['count'].values
    num_groups = len(sizes)

    # 2. Statistics
    avg_size = np.mean(sizes)
    max_size = np.max(sizes)

    print(f"  - Total Groups: {num_groups:,}")
    print(f"  - Group Sizes:  Avg={avg_size:.1f} | Max={max_size:,}")

    # 3. Edge Count Estimation (Global)
    # Each node connects to at most `cap` neighbors within its group
    estimated_edges = np.sum(sizes * np.minimum(sizes, cap))

    print(f"  - Estimated Layer Edges: {estimated_edges:,}")
    return estimated_edges

# --- 3. NODE DEGREE CALCULATION ---
def calculate_node_degrees(df_unique, geo_cap, lender_cap):
    """
    Compute per-node degree statistics induced by stochastic blocking.

    For each node:
        Degree = Geo-layer neighbors + Lender-layer neighbors

    Caps are applied independently per layer.

    Args:
        df_unique (pd.DataFrame): Unique loan-level dataset
        geo_cap (int): Geo-layer degree cap
        lender_cap (int): Lender-layer degree cap

    Returns:
        np.ndarray: Total degree per node
    """
    print(f"\n{'=' * 20}  CALCULATING NODE DEGREES {'=' * 20}")
    print("  ... Mapping group sizes to individual nodes (Vectorized)...")

    # Compute group sizes aligned per node
    geo_sizes = df_unique.groupby('Geo_Key')['Loan_Sequence_Number'].transform('count')
    lender_sizes = df_unique.groupby('Lender_Key')['Loan_Sequence_Number'].transform('count')

    # Apply caps (subtract 1 to exclude self-connections)
    geo_degrees = np.minimum(geo_sizes - 1, geo_cap).clip(lower=0)
    lender_degrees = np.minimum(lender_sizes - 1, lender_cap).clip(lower=0)

    # Total supra-graph degree
    total_degrees = geo_degrees + lender_degrees

    return total_degrees

# --- 4. MAIN ANALYSIS PIPELINE ---
def main():
    # Load Data
    print("Loading H1 Features...")
    try:
        df = pyreadr.read_r(str(cfg.FINAL_DATA_FILE))[None]
    except KeyError:
        # Handle case where pyreadr returns dict
        raw = pyreadr.read_r(str(cfg.FINAL_DATA_FILE))
        df = raw[list(raw.keys())[0]]

    print(f"   Raw Rows (Loan-Months): {len(df):,}")

    # Deduplicate to unique loans (graph nodes)
    print("   Deduplicating to Unique Loans...")
    df_unique = df.drop_duplicates(subset='Loan_Sequence_Number')[['Loan_Sequence_Number', 'Geo_Key', 'Lender_Key']]
    print(f"   Unique Loans (Nodes):   {len(df_unique):,}")

    del df
    gc.collect()

    # 4.1 Global Layer Analysis
    geo_edges = analyze_layer_groups(df_unique, 'Geo_Key', TEST_GEO_CAP, "GEOGRAPHICAL LAYER")
    lender_edges = analyze_layer_groups(df_unique, 'Lender_Key', TEST_LENDER_CAP, "LENDER LAYER")

    # 4.2 Node Degree Statistics
    total_degrees = calculate_node_degrees(df_unique, TEST_GEO_CAP, TEST_LENDER_CAP)

    # Calculate Percentiles
    p50 = int(np.percentile(total_degrees, 50))
    p90 = int(np.percentile(total_degrees, 90))
    p95 = int(np.percentile(total_degrees, 95))
    p99 = int(np.percentile(total_degrees, 99))
    max_d = int(np.max(total_degrees))

    avg_degree = np.mean(total_degrees)

    print("\n BORROWER NEIGHBOR STATISTICS:")
    print(f"   Average Density:  {avg_degree:.2f} neighbors")
    print(f"   Median (P50):     {p50} neighbors")
    print(f"   90th Percentile:  {p90} neighbors")
    print(f"   95th Percentile:  {p95} neighbors")
    print(f"   99th Percentile:  {p99} neighbors  <-- USE THIS FOR CONFIG")
    print(f"   Max Degree:       {max_d} neighbors")

    print(f"\n OPTIMIZATION RECOMMENDATION:")
    print(f"   Set NUM_NEIGHBORS = [{p99}] in config_h1.py")
    print("   (This ensures 99% of borrowers see their FULL graph structure)")

    # 4.3 System Summary and Memory Estimation
    total_edges = geo_edges + lender_edges + len(df_unique)

    print(f"\n{'=' * 20} SYSTEM SUMMARY {'=' * 20}")
    print(f"Total Nodes:      {len(df_unique):,}")
    print(f"Total Edges (Est): {total_edges:,}")

    # Approximate VRAM estimation (rule-of-thumb for GAT-style backprop)
    est_vram_gb = (total_edges * 400) / (1024 ** 3)

    print(f"Est. Training VRAM:  ~{est_vram_gb:.2f} GB")

    if total_edges > 60_000_000:
        print(" CRITICAL: Edges exceed 60M. Reduce MAX_DEGREE_LENDER in config_h1.py.")
    else:
        print(" SYSTEM CHECK: Graph size fits RTX 4090 (24GB).")

# --- 5. SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    main()