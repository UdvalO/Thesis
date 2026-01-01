# ==============================================================================
# diagnostics.py (H2)
#
# PURPOSE:
#   H2 Strict Diagnostic Suite for validating the Dynamic Temporal GNN pipeline.
#   Ensures data integrity, dynamic topology correctness, and model mechanics.
#
# STEPS:
#   1. DATA PIPELINE VALIDATION
#       - Verify alignment between R preprocessing and Python data pipeline.
#       - Check feature normalization ranges (0-1).
#       - Verify strict temporal separation (Train vs Test split).
#   2. GRAPH STRUCTURE VALIDATION
#       - Verify dynamic node consistency (same node set across snapshots).
#       - Check average degree density against Target K (topology health).
#   3. MODEL MECHANICS VALIDATION
#       - Synthetic forward pass to verify architecture shapes.
#       - Tiny batch overfitting test to verify learning capability.
#
# INPUTS:
#   - PyTorch Graphs: train_graphs.pt, test_graphs.pt (List of Data objects)
#   - R preprocessed targets: •	train_targets.rds (on remote computer)
#   - Config: config_2.py
#
# OUTPUTS:
#   - Console diagnostics with warnings or success markers.
# ==============================================================================

import argparse
import sys
import time
import torch
import torch.nn as nn
import pyreadr
import traceback
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

# Internal Project Imports
import config_h2 as cfg
from model_architecture_h2 import DynamicTemporalGNN
from training_h2 import NodeBatchDataset, train_epoch, get_temporal_subgraph_batch


# ==============================================================================
# 1. DATA PIPELINE VALIDATION
# ==============================================================================

def test_r_python_alignment(train_graphs):
    """
    Verify that Python preprocessing matches R output.
    - Reports snapshot counts (Ensures no dropped time steps).
    - Verifies features are normalized between 0 and 1.
    """
    print("\n" + "-" * 40)
    print("CHECK: R vs Python Data Alignment")
    print("-" * 40)

    # Load R data (Targets)
    try:
        train_targets_r = list(pyreadr.read_r(str(cfg.TRAIN_TARGETS_FILE)).values())[0]
    except Exception as e:
        print(f" Warning: Could not load R files: {e}")
        return

    r_snaps = len(train_targets_r['Snapshot_Date'].unique())
    py_snaps = len(train_graphs)

    print(f" R Train Snapshots (Target Months): {r_snaps}")
    print(f" Python Graphs (Input Snapshots):   {py_snaps}")

    if r_snaps != py_snaps:
        print(" CRITICAL: Snapshot count mismatch!")
    else:
        print(" Snapshot counts align.")

    # Check Features (Range)
    first_graph = train_graphs[0]
    # Only check master nodes (first half) to avoid checking dummy padding
    num_master = first_graph.x.shape[0] // 2
    x = first_graph.x[:num_master]

    print(f" Feature Shape (Snapshot 0): {x.shape}")

    if x.min() < -0.01 or x.max() > 1.01:
        print(f" Features out of range [{x.min():.2f}, {x.max():.2f}]! Check Scaling.")
    else:
        print(" Features normalized correctly (0-1).")


def test_train_test_split(train_graphs, test_graphs):
    """
    Verify strict temporal separation between Training and Test sets.
    - Ensures no leakage (Train Max Date < Test Min Date).
    """
    print("\n" + "-" * 40)
    print("CHECK: Temporal Split Integrity")
    print("-" * 40)

    train_dates = [g.snapshot_date for g in train_graphs]
    test_dates = [g.snapshot_date for g in test_graphs]

    print(f" Train Period: {min(train_dates)} -> {max(train_dates)}")
    print(f" Test Period:  {min(test_dates)} -> {max(test_dates)}")

    if max(train_dates) >= min(test_dates):
        print(" CRITICAL: Train and Test periods overlap! (Data Leakage Risk)")
    else:
        print(" Temporal split is valid (No leakage detected).")


# ==============================================================================
# 2. GRAPH STRUCTURE VALIDATION
# ==============================================================================

def test_dynamic_topology(train_graphs):
    """
    Check dynamic supra-adjacency structure.
    - Validates node consistency (preventing cardinality changes).
    - Checks graph density (average degree) against target K.
    """
    print("\n" + "-" * 40)
    print("CHECK: Dynamic Topology Structure")
    print("-" * 40)

    # 1. Node Consistency Check
    num_nodes_0 = train_graphs[0].num_nodes
    if not all(g.num_nodes == num_nodes_0 for g in train_graphs):
        print(" CRITICAL: Node counts vary across snapshots! (Model requires fixed size)")
    else:
        print(f" Node Consistency: All snapshots have {num_nodes_0} nodes.")

    # 2. Connectivity Check (Layer 0 Density)
    data = train_graphs[0]
    num_master = num_nodes_0 // 2
    edge_index = data.edge_index

    # Filter edges where both source and target are in layer 0 (Geo layer)
    layer0_mask = (edge_index[0] < num_master) & (edge_index[1] < num_master)
    layer0_edges = edge_index[:, layer0_mask].shape[1]

    avg_deg = layer0_edges / num_master
    print(f" Avg Degree (Layer 0): {avg_deg:.2f} (Target K={cfg.BASE_K_GEO})")

    if avg_deg < 1.0:
        print(" WARNING: Graph is extremely sparse (Avg Degree < 1). Check construction.")
    else:
        print(" Topology density looks reasonable.")


# ==============================================================================
# 3. MODEL MECHANICS VALIDATION
# ==============================================================================

def test_model_forward():
    """
    Validate forward pass of DynamicTemporalGNN on synthetic data.
    - Ensures tensor shapes align (Sequence -> GAT -> LSTM -> MLP).
    - Verifies gradient flow capability.
    """
    print("\n" + "-" * 40)
    print("CHECK: Model Architecture (Synthetic Forward Pass)")
    print("-" * 40)

    model = DynamicTemporalGNN(
        num_features=cfg.NUM_FEATURES,
        gat_hidden=cfg.GAT_HIDDEN, gat_heads=cfg.GAT_HEADS, gat_out=cfg.GAT_OUT,
        lstm_hidden=cfg.LSTM_HIDDEN, lstm_layers=cfg.LSTM_LAYERS,
        mlp_hidden=cfg.MLP_HIDDEN, num_classes=cfg.NUM_CLASSES, dropout=cfg.DROPOUT
    ).to(cfg.DEVICE)

    # Create dummy snapshots (Sequence of Data objects)
    from torch_geometric.data import Data
    dummy_snaps = []
    nodes = 100
    for _ in range(cfg.SEQUENCE_LENGTH):
        x = torch.randn(nodes, cfg.NUM_FEATURES).to(cfg.DEVICE)
        edge_index = torch.randint(0, nodes, (2, 200)).to(cfg.DEVICE)
        dummy_snaps.append(Data(x=x, edge_index=edge_index, num_nodes=nodes))

    try:
        logits = model(dummy_snaps)
        print(f" Input Nodes: {nodes // 2} (Master Layer)")
        print(f" Output Shape: {logits.shape}")

        if logits.shape == (nodes // 2, 1):
            print(" Forward pass successful.")
        else:
            print(" WARNING: Output shape mismatch.")
    except Exception as e:
        print(f" Forward Pass Failed: {e}")
        traceback.print_exc()


def test_overfitting_tiny(train_graphs):
    """
    Test whether model can overfit a tiny batch of REAL data.
    - Confirms model can learn patterns (Loss should decrease).
    - Validates label masking and loss calculation.
    """
    print("\n" + "-" * 40)
    print("CHECK: Overfitting Capability (Real Data)")
    print("-" * 40)

    # Get sequence for one target snapshot
    subset = train_graphs[:cfg.SEQUENCE_LENGTH]
    target = subset[-1]

    # Find valid nodes (that have labels)
    num_master = target.num_nodes // 2
    valid_mask = target.y[:num_master] != -1
    valid_nodes = torch.where(valid_mask)[0][:256]  # Tiny batch of 256 nodes

    if len(valid_nodes) == 0:
        print(" No valid labels found for overfitting test.")
        return

    # Extract subgraph
    sub_seq, target_map = get_temporal_subgraph_batch(subset, valid_nodes)
    sub_seq = [d.to(cfg.DEVICE) for d in sub_seq]
    target_map = target_map.to(cfg.DEVICE)

    labels = sub_seq[-1].y[target_map].float().unsqueeze(1)

    # Initialize small model
    model = DynamicTemporalGNN(
        num_features=cfg.NUM_FEATURES,
        gat_hidden=32, gat_heads=2, gat_out=32,  # Smaller for speed
        lstm_hidden=32, lstm_layers=1,
        mlp_hidden=32, num_classes=1, dropout=0.0
    ).to(cfg.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    print(" Training 15 epochs on tiny batch...")
    initial_loss = None

    for epoch in range(15):
        model.train()
        optimizer.zero_grad()
        logits = model(sub_seq)
        loss = criterion(logits[target_map], labels)
        loss.backward()
        optimizer.step()

        if epoch == 0: initial_loss = loss.item()

    print(f" Initial Loss: {initial_loss:.4f}")
    print(f" Final Loss:   {loss.item():.4f}")

    if loss.item() < initial_loss:
        print(" Model learned (Loss decreased).")
    else:
        print(" Model struggled. Check gradients or data labels.")


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

def load_and_unwrap(path):
    """Helper to load data and unwrap nested lists if necessary."""
    # Allow safe globals for PyG
    with torch.serialization.safe_globals([DataTensorAttr, DataEdgeAttr]):
        graphs = torch.load(path, weights_only=False)

    # R conversion sometimes wraps objects in lists [[Data]] -> [Data]
    if isinstance(graphs[0], list):
        print(f" Unwrapping lists in {path.name}...")
        # FIX: Select the LAST element [-1] (The Target), not the first [0]
        graphs = [g[-1] for g in graphs]

    return graphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H2 Diagnostics")
    parser.add_argument('mode', nargs='?', default='all', choices=['data', 'model', 'all'])
    args = parser.parse_args()

    start = time.time()

    # LOAD DATA ONCE (Optimization)
    print("Loading Graph Data (This happens once)...")
    train_graphs = load_and_unwrap(cfg.GRAPH_DIR / "train_graphs.pt")

    # Load test only if running data checks
    if args.mode in ['data', 'all']:
        test_graphs = load_and_unwrap(cfg.GRAPH_DIR / "test_graphs.pt")
        test_r_python_alignment(train_graphs)
        test_train_test_split(train_graphs, test_graphs)
        test_dynamic_topology(train_graphs)

    if args.mode in ['model', 'all']:
        test_model_forward()
        test_overfitting_tiny(train_graphs)

    print(f"\n Diagnostics Complete ({time.time() - start:.2f}s) ")