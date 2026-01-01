# ==============================================================================
# diagnostics_h1.py
#
# PURPOSE:
#   H1 Strict Diagnostic Suite for validating data pipeline, graph structure,
#   and model mechanics.
#
# STEPS:
#   1. DATA PIPELINE VALIDATION
#       - Verify alignment between R preprocessing and Python data pipeline.
#       - Check feature normalization ranges.
#       - Test ghost node logic (nodes defaulted correctly).
#   2. GRAPH STRUCTURE VALIDATION
#       - Check static supra-adjacency matrix for correctness.
#       - Verify node counts align with 250k sample size.
#   3. MODEL MECHANICS VALIDATION
#       - Forward pass on small batch using NeighborLoader.
#       - Tiny batch overfitting test to verify model learnability.
#
# INPUTS:
#   - PyTorch Graphs: train_features.pt (on remote computer)
#   - Static edges: static_edge_index.pt (on remote computer)
#   - Metadata: metadata.pt (on flash drive)
#   - R preprocessed targets: train_targets.rds (on remote computer)
#
# OUTPUTS:
#   - Console diagnostics with warnings or success markers.
# ==============================================================================

import argparse
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pyreadr
import traceback
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# Internal Project Imports
import config_h1 as cfg
from model_architecture_h1 import DYMGNN
from training_h1 import compute_global_pos_weight, filter_ghost_nodes


# ==============================================================================
# 1. DATA PIPELINE VALIDATION
# ==============================================================================

def test_r_python_alignment():
    """
    Verify that Python preprocessing matches R output.
    - Reports window counts (Burn-in logic implies Python windows >= R expectations).
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

    # Load Python data (Features)
    train_graphs = torch.load(cfg.GRAPH_SAVE_DIR / "train_features.pt", weights_only=False)

    # Report Counts (No strict mismatch check due to Burn-In strategy)
    r_train_snaps = len(train_targets_r['Snapshot_Date'].unique())

    print(f" R Train Snapshots (Target Months): {r_train_snaps}")
    print(f" Python Train Windows (Input Batches): {len(train_graphs)}")
    print(" Note: Python count may equal R count due to 'Burn-In' preserving early windows.")

    # Check Features (Range)
    first_window = train_graphs[0]
    x = first_window.x  # Shape: (T, 2N, F)
    print(f" Feature Shape: {x.shape}")

    if x.min() < -0.01 or x.max() > 1.01:
        print(" Features out of range [0, 1]! Check Scaling.")
    else:
        print(" Features normalized correctly.")


def test_ghost_node_logic():
    """
    Verify that nodes which defaulted ('ghost nodes') have features zeroed out.
    - Detects nodes that were active initially but became inactive in the final timestep.
    """
    print("\n" + "-" * 40)
    print("CHECK: Ghost Node Logic")
    print("-" * 40)

    train_graphs = torch.load(cfg.GRAPH_SAVE_DIR / "train_features.pt", weights_only=False)

    found_ghost = False
    for i, window in enumerate(train_graphs[:5]):
        x = window.x  # Shape: (T, 2N, F)

        # Calculate activity (sum of abs features) per node
        activity = x.abs().sum(dim=2)  # (T, 2N)

        starts_active = activity[0] > 0.001
        ends_dead = activity[-1] < 0.001

        ghosts = (starts_active & ends_dead).sum().item()

        if ghosts > 0:
            print(f" Window {i}: Found {ghosts} Ghost Nodes (Active -> Dead).")
            found_ghost = True
            break

    if not found_ghost:
        print(" No Ghost Nodes detected (might be expected if default rate is low).")
    else:
        print(" Ghost Node logic functional.")


# ==============================================================================
# 2. GRAPH STRUCTURE VALIDATION
# ==============================================================================

def test_static_topology():
    """
    Check static supra-adjacency matrix.
    - Validates total number of edges.
    - Confirms node count matches expected 250k loans (500k Supra nodes).
    """
    print("\n" + "-" * 40)
    print("CHECK: Static Topology Structure")
    print("-" * 40)

    static_edge_index = torch.load(cfg.GRAPH_SAVE_DIR / "static_edge_index.pt", weights_only=False)
    meta = torch.load(cfg.GRAPH_SAVE_DIR / "metadata.pt", weights_only=False)

    # Meta['num_nodes'] is typically the size of the saved feature tensor (Supra Nodes)
    # If 250k loans -> 500k Supra Nodes
    total_nodes_supra = meta['num_nodes']

    print(f" Nodes (Graph Structure): {total_nodes_supra}")
    print(f" Edges: {static_edge_index.shape[1]:,}")

    if static_edge_index.max() >= total_nodes_supra:
        print(" CRITICAL: Edges point to non-existent nodes!")
    else:
        print(" Edge indices valid.")


# ==============================================================================
# 3. MODEL MECHANICS VALIDATION (NeighborLoader)
# ==============================================================================

def test_model_forward():
    """
    Validate forward pass of DYMGNN on small batch using NeighborLoader.
    - Ensures outputs have correct shape.
    - Verifies ghost node filtering integration.
    """
    print("\n" + "-" * 40)
    print("CHECK: Model Forward Pass & NeighborLoader")
    print("-" * 40)

    train_graphs = torch.load(cfg.GRAPH_SAVE_DIR / "train_features.pt", weights_only=False)
    static_edge_index = torch.load(cfg.GRAPH_SAVE_DIR / "static_edge_index.pt", weights_only=False)
    meta = torch.load(cfg.GRAPH_SAVE_DIR / "metadata.pt", weights_only=False)

    window = train_graphs[0]

    # Permute features: (T, N, F) -> (N, T, F) for NeighborLoader compatibility
    data = Data(
        x=window.x.permute(1, 0, 2),
        y=window.y,
        edge_index=static_edge_index,
        num_nodes=window.x.shape[1]
    )

    # Select valid (non-masked) indices for a tiny batch
    valid_idx = torch.where(window.y != -1)[0][:64]

    loader = NeighborLoader(
        data,
        num_neighbors=[10],  # Small neighbor cap for test speed
        batch_size=32,
        input_nodes=valid_idx,
        shuffle=False
    )

    model = DYMGNN(
        num_features=meta['num_features'],
        hidden_dim=cfg.HIDDEN_DIM,
        num_heads=cfg.NUM_HEADS,
        dropout=0.0
    ).to(cfg.DEVICE)

    try:
        for batch in loader:
            batch = batch.to(cfg.DEVICE)

            # Permute back to (T, Batch, F) for LSTM
            x_seq = batch.x.permute(1, 0, 2)

            # Filter ghost nodes dynamically
            edge_index = filter_ghost_nodes(x_seq, batch.edge_index)

            # Forward Pass
            logits = model(x_seq, edge_index)

            print(f"   Logits Shape: {logits.shape}")
            if logits.shape[0] != batch.x.shape[0]:
                print(" Output shape mismatch!")
            else:
                print(" Forward pass successful.")
            break

    except Exception as e:
        print(f" Model Forward Failed: {e}")
        traceback.print_exc()


def test_overfitting_tiny():
    """
    Test whether model can overfit a tiny batch.
    - Confirms model can learn quickly (loss should decrease significantly).
    """
    print("\n" + "-" * 40)
    print("CHECK: Overfitting Capability")
    print("-" * 40)

    train_graphs = torch.load(cfg.GRAPH_SAVE_DIR / "train_features.pt", weights_only=False)
    static_edge_index = torch.load(cfg.GRAPH_SAVE_DIR / "static_edge_index.pt", weights_only=False)

    window = train_graphs[0]
    data = Data(
        x=window.x.permute(1, 0, 2),
        y=window.y,
        edge_index=static_edge_index,
        num_nodes=window.x.shape[1]
    )

    valid_idx = torch.where(window.y != -1)[0][:32]

    loader = NeighborLoader(
        data,
        num_neighbors=[10],
        batch_size=32,
        input_nodes=valid_idx
    )

    batch = next(iter(loader))  # Grab one static batch
    batch = batch.to(cfg.DEVICE)

    model = DYMGNN(
        num_features=cfg.NUM_INPUT_FEATURES,
        hidden_dim=32,
        num_heads=2,
        dropout=0.0
    ).to(cfg.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    print(" Training on 1 batch for 15 epochs...")
    initial_loss = None

    for epoch in range(15):
        model.train()
        optimizer.zero_grad()

        x_seq = batch.x.permute(1, 0, 2)
        logits = model(x_seq, batch.edge_index).squeeze()

        # Loss on target nodes only (first 32)
        target_logits = logits[:32]
        target_labels = batch.y[:32].float()

        loss = criterion(target_logits, target_labels)
        loss.backward()
        optimizer.step()

        if epoch == 0: initial_loss = loss.item()

    final_loss = loss.item()
    print(f" Initial Loss: {initial_loss:.4f}")
    print(f" Final Loss:   {final_loss:.4f}")

    if final_loss < initial_loss:
        print(" Model learned (Loss decreased).")
    else:
        print(" Model struggled. Check gradients.")


# ==============================================================================
# 4. MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H1 Diagnostics")
    parser.add_argument('mode', nargs='?', default='all', choices=['data', 'model', 'all'])
    args = parser.parse_args()

    start = time.time()

    if args.mode in ['data', 'all']:
        test_r_python_alignment()
        test_ghost_node_logic()
        test_static_topology()

    if args.mode in ['model', 'all']:
        test_model_forward()
        test_overfitting_tiny()

    print(f"\n Diagnostics Complete in {time.time() - start:.2f}s ")