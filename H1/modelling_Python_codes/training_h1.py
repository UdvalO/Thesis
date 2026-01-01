# ==============================================================================
# training_h1.py (H1)
#
# PURPOSE:
#   Orchestrates the training of the Dynamic GNN (H1) using a "Structure-First"
#   loading strategy to optimize VRAM usage on consumer hardware.
#   Implements the specific training protocols defined in Thesis Section 3.2.3,
#   including Cost-Sensitive Loss, Mixed Precision, and Ghost Node filtering.
#
# LOGIC & THESIS ALIGNMENT:
#   1. Structure-First Loading (Thesis Sec 3.2.1):
#      - Decouples the massive node features from the graph topology.
#      - The NeighborLoader samples only the *structure* (indices), and features
#        are sliced dynamically on the GPU. This reduces memory pressure.
#
#   2. Ghost Node Filtering (Thesis Sec 3.2.1):
#      - "A dynamic 'Temporal Mask' is applied to strictly zero out nodes that
#        are inactive... preserving the temporal integrity."
#      - Prevents future defaults or past closed loans from leaking into the
#        current prediction window.
#
#   3. Cost-Sensitive Loss (Thesis Sec 3.2.3):
#      - "A scaling factor (pos_weight) amplifies the penalty on positive-class
#        errors... prioritizing recall for high-risk loans."
#
# INPUTS:
#   - Pre-computed Tensor Files (train_features.pt, static_edge_index.pt)
#   - Metadata (metadata.pt) for model initialization.
#
# OUTPUTS:
#   - best_model_h1.pt: The optimal model state (based on Validation AUC).
#   - training_results.json: Final metrics for the run.
#   - TensorBoard Logs: Real-time visualization of Loss/AUC curves.
# ==============================================================================

import os
import time
import logging
import sys
import gc
import json  # <--- Added for your result saving

import torch
import torch.nn as nn
import torch.optim as optim
# Native Mixed Precision
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter  # <--- Added for writer
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index
from sklearn.metrics import roc_auc_score, precision_recall_curve
import numpy as np
from tqdm import tqdm

# Local Imports
import config_h1 as cfg
from model_architecture_h1 import DYMGNN

# --- 1. LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(cfg.GRAPH_SAVE_DIR / "training_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# --- 2. HELPERS ---

def compute_global_pos_weight(train_windows):
    """
    Calculates the global positive class weight for the Cost-Sensitive Loss.

    Methodology (Thesis Sec 3.2.3):
        - "Due to a strong class imbalance (~19 non-defaults per default)...
          a scaling factor amplifies the penalty on positive-class errors."
        - Weight = Negatives / Positives.

    Args:
        train_windows (list): List of training Data objects.

    Returns:
        torch.Tensor: Scalar weight for the positive class.
    """
    logger.info("  Calculating global class weights...")
    total_pos = 0
    total_neg = 0

    try:
        # Aggregate counts across all temporal windows
        for data in train_windows:
            y = data.y
            mask = y != -1
            total_pos += (y[mask] == 1).sum().item()
            total_neg += (y[mask] == 0).sum().item()

        if total_pos == 0:
            logger.warning(" No positive samples found! Defaulting weight to 1.0")
            return torch.tensor(1.0)

        weight = total_neg / total_pos
        logger.info(f"   Pos Weight: {weight:.4f}")
        return torch.tensor(weight, dtype=torch.float)
    except Exception as e:
        logger.error(f"Failed to compute class weights: {e}")
        return torch.tensor(1.0)


def filter_ghost_nodes(x_seq, edge_index):
    """
        Applies the "Temporal Mask" to remove inactive (Ghost) nodes from the graph.

        Methodology (Thesis Sec 3.2.1):
            - "A dynamic 'Temporal Mask' is applied to strictly zero out nodes...
              Masking is critical to prevent information leakage from future states
              or stale historical states."

        Args:
            x_seq (Tensor): Feature sequence [Time, Nodes, Feats].
            edge_index (Tensor): Graph connectivity [2, Edges].

        Returns:
            Tensor: Filtered edge_index containing only active connections.
        """
    # Detect inactive nodes (all-zero features across time)
    activity_score = x_seq.abs().sum(dim=(0, 2))
    active_mask = activity_score > 1e-6

    # Keep edge only if BOTH source and target are active
    src, dst = edge_index
    keep_mask = active_mask[src] & active_mask[dst]
    return edge_index[:, keep_mask]


@torch.no_grad()
def evaluate(model, full_x, full_y, static_edge_index, device, batch_size=4096):
    """
    Evaluation loop utilizing "Structure-First" loading for memory safety.

    Methodology (Thesis Sec 4.3.1):
        - Validates on the "Future" window (Chronological Split).
        - Applies the same Stochastic Neighbor Sampling (K=50) as training
          to ensure consistent distribution approximation.

    Args:
        model (nn.Module): The trained DYMGNN.
        full_x (Tensor): Features for the validation window.
        full_y (Tensor): Targets for the validation window.
        static_edge_index (Tensor): The explicit topology.
        device (torch.device): GPU/CPU.

    Returns:
        tuple: (AUC, Max_F1)
    """
    model.eval()
    all_preds = []
    all_labels = []

    # Identify valid nodes (Target exists and is within Layer 1 bounds)
    num_total_nodes = full_x.shape[1]
    num_unique_loans = num_total_nodes // 2
    node_indices = torch.arange(num_total_nodes, device=full_y.device)
    valid_mask = (full_y != -1) & (node_indices < num_unique_loans)
    valid_indices = torch.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return 0.5, 0.0

    # Lightweight Structural Data (Indices only)
    eval_struct_data = Data(edge_index=static_edge_index, num_nodes=num_total_nodes)

    loader = NeighborLoader(
        eval_struct_data,
        num_neighbors=cfg.NUM_NEIGHBORS,
        batch_size=batch_size,
        input_nodes=valid_indices,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )

    for batch in loader:
        batch = batch.to(device)

        # Dynamic Feature Slicing: Only load features for sampled nodes
        x_seq = full_x[:, batch.n_id, :]
        batch_labels = full_y[batch.n_id[:batch.batch_size]]

        with autocast('cuda'):
            # Apply Temporal Mask
            edge_index_filtered = filter_ghost_nodes(x_seq, batch.edge_index)
            logits = model(x_seq, edge_index_filtered).squeeze()

            # Slice logits to match target batch size (exclude neighbors)
            logits_batch = logits[:batch.batch_size]

        all_preds.extend(torch.sigmoid(logits_batch).float().cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

    if len(all_labels) == 0: return 0.5, 0.0

    try:
        auc = roc_auc_score(all_labels, all_preds)
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        denom = precision + recall
        f1_scores = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom != 0)
        best_f1 = np.max(f1_scores)
    except Exception as e:
        logger.warning(f"Metric calculation failed: {e}")
        auc, best_f1 = 0.5, 0.0

    return auc, best_f1


# --- 3. MAIN TRAINING LOOP ---
def main():
    torch.set_float32_matmul_precision('medium')
    logger.info(" Starting Production-Optimized H1 Training...")

    # 3.1 Initialize TensorBoard Writer
    log_dir = cfg.GRAPH_SAVE_DIR / "runs"
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f" TensorBoard logging to: {log_dir}")

    try:
        cfg.GRAPH_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        device = cfg.DEVICE

        logger.info(" Loading Tensors...")
        train_windows = torch.load(cfg.GRAPH_SAVE_DIR / "train_features.pt", weights_only=False)
        test_list = torch.load(cfg.GRAPH_SAVE_DIR / "test_features.pt", weights_only=False)
        static_edge_index = torch.load(cfg.GRAPH_SAVE_DIR / "static_edge_index.pt", weights_only=False)
        static_edge_index = sort_edge_index(static_edge_index)
        meta = torch.load(cfg.GRAPH_SAVE_DIR / "metadata.pt", weights_only=False)
        num_nodes = meta['num_nodes']

        logger.info(f"   Nodes: {num_nodes} | Train Windows: {len(train_windows)}")

    except Exception as e:
        logger.critical(f" Critical error loading data: {e}")
        sys.exit(1)

    # 3.2 Structure-First Loader Initialization
    # Thesis Sec 3.2.3: "NeighborLoader was configured to dynamically sample a random
    # subset of neighbors (set to K=50)... minimizing VRAM footprint."
    structural_data = Data(edge_index=static_edge_index, num_nodes=num_nodes)
    master_loader = NeighborLoader(
        structural_data,
        num_neighbors=cfg.NUM_NEIGHBORS,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )

    # 3.3 Model & Optimizer Setup
    try:
        model = DYMGNN(
            num_features=meta['num_features'],
            hidden_dim=cfg.HIDDEN_DIM,
            num_heads=cfg.NUM_HEADS,
            dropout=cfg.DROPOUT
        ).to(device)

        # Optimizer: Fixed LR 0.001 (Thesis Table 3)
        optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

        # Mixed Precision Scaler
        scaler = GradScaler('cuda')

        # Cost-Sensitive Loss (Thesis Sec 3.2.3)
        # implicit sigmoid by pytorch
        criterion = nn.BCEWithLogitsLoss(pos_weight=compute_global_pos_weight(train_windows).to(device))
    except Exception as e:
        logger.critical(f" Model initialization failed: {e}")
        sys.exit(1)

    # Test Data Prep (Single Window Validation)
    try:
        test_window = test_list[0]
        test_x = test_window.x.to(device)
        test_y = test_window.y.to(device)
    except torch.cuda.OutOfMemoryError:
        logger.warning(" Test set too large for VRAM. Evaluation will use CPU-paging.")
        test_x = test_window.x
        test_y = test_window.y

    best_val_auc = 0
    patience = 0
    total_start = time.time()

    epoch = 0
    val_f1 = 0.0

    # 3.4 Training Loop
    logger.info(" Starting Epochs...")

    try:
        for epoch in range(cfg.EPOCHS):
            epoch_start = time.time()
            model.train()
            total_loss = 0
            total_batches = 0

            # Randomize Window Order (Temporal Regularization)
            window_indices = np.random.permutation(len(train_windows))
            pbar = tqdm(window_indices, desc=f"Ep {epoch + 1}", leave=False)

            for win_idx in pbar:
                window_data = train_windows[win_idx]

                for batch in master_loader:
                    try:
                        # Dynamic Slicing: Map global IDs to current window features
                        n_id = batch.n_id
                        x_seq = window_data.x[:, n_id, :]
                        center_n_id = n_id[:batch.batch_size]
                        batch_y = window_data.y[center_n_id]

                        # Skip batch if no valid targets
                        valid_mask = batch_y != -1
                        if valid_mask.sum() == 0: continue

                        x_seq = x_seq.to(device, non_blocking=True)
                        batch_y = batch_y.to(device, non_blocking=True)
                        edge_index = batch.edge_index.to(device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)

                        with autocast('cuda'):
                            # Ghost Filter: Remove inactive edges in this window
                            edge_index = filter_ghost_nodes(x_seq, edge_index)

                            # Forward Pass
                            logits = model(x_seq, edge_index).squeeze()

                            # Slice to get only the 'center' nodes (exclude sampled neighbors)
                            logits_batch = logits[:batch.batch_size]

                            # Compute Loss on Valid Nodes Only
                            loss = criterion(logits_batch[valid_mask], batch_y[valid_mask].float())

                        # Backward Pass with Scaler
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        total_loss += loss.item()
                        total_batches += 1

                        # Memory Cleanup
                        del logits, loss, x_seq, edge_index, batch_y

                    except torch.cuda.OutOfMemoryError:
                        logger.error(" CUDA OOM! Skipping batch.")
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue

            # End of Epoch
            avg_loss = total_loss / max(total_batches, 1)
            val_auc, val_f1 = evaluate(model, test_x, test_y, static_edge_index, device)

            duration = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch + 1:03d} | {duration:.1f}s | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | F1: {val_f1:.4f}")

            # --- TensorBoard Logging ---
            writer.add_scalar("Train/Loss", avg_loss, epoch)
            writer.add_scalar("Val/AUC", val_auc, epoch)
            writer.add_scalar("Val/F1", val_f1, epoch)

            # Checkpoint & Early Stopping (Thesis Sec 4.3.1: Patience=50)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience = 0
                torch.save(model.state_dict(), cfg.GRAPH_SAVE_DIR / "best_model_h1.pt")
            else:
                patience += 1
                if patience >= cfg.PATIENCE:
                    logger.info("  Early Stopping Triggered")
                    break

            gc.collect()

    except KeyboardInterrupt:
        logger.warning(" Training interrupted by user.")
    except Exception as e:
        logger.critical(f" Unexpected crash: {e}", exc_info=True)


    finally:
        # --- 4. SAVE RESULTS ---
        total_time_sec = time.time() - total_start
        trained_epochs = epoch + 1 if 'epoch' in locals() else 0
        final_f1 = val_f1 if 'val_f1' in locals() else 0.0

        results = {
            "total_training_time_sec": total_time_sec,
            "best_val_auc": best_val_auc,
            "epochs_trained": trained_epochs,
            "final_val_f1": final_f1
        }

        # Save to JSON
        json_path = cfg.GRAPH_SAVE_DIR / "training_results.json"
        try:
            with open(json_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f" Results saved to {json_path}")
        except Exception as e:
            logger.error(f" Failed to save results JSON: {e}")

        # Close TensorBoard
        if 'writer' in locals():
            writer.close()

        logger.info(f" Training Session Ended. Total Time: {total_time_sec / 60:.1f} min")

if __name__ == "__main__":
    main()