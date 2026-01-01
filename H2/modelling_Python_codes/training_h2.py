# ==============================================================================
# training_h2.py (H2: Implicit Topology Training Loop)
#
# PURPOSE:
#   Trains the Dynamic Temporal GNN (DT-GNN) for Hypothesis 2 using an implicit,
#   behavior-driven graph topology.
#
# STEPS:
#   1. Load pre-packaged sliding window graph sequences (Train/Test).
#   2. Initialize the DT-GNN model (GAT + LSTM + Node-Independent Attention).
#   3. Compute class weights dynamically from a random training sample.
#   4. Train using Focal Loss to address extreme class imbalance (~50:1).
#   5. Validate using Area Under the Curve (AUC) and F1 Score (fixed 0.5 threshold).
#   6. Save the best model checkpoint based on validation AUC.
#
# INPUTS:
#   - Graph Sequences: train_graphs.pt (on remote storage)
#   - Configuration: config_h2.py
#
# OUTPUTS:
#   - Best Model Checkpoint: best_model_final.pt
#   - Training Logs: TensorBoard logs in /logs
#   - Results Summary: training_results.json
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
from tqdm import tqdm
import time
import gc
import os
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph

from model_architecture_h2 import DynamicTemporalGNN
import config_h2 as cfg

# --- Focal Loss Class ---
class FocalLoss(nn.Module):
    """
        Implements Focal Loss to address extreme class imbalance.
        Down-weights easy negatives to focus gradients on hard positive examples.
        """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
                Computes the focal loss between `inputs` (logits) and `targets` (binary).
                """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# --- Helper Functions ---

def compute_pos_weight(graphs, sample_size=100):
    """
        Estimates the positive class weight from a random sample of training graphs.
        Used to initialize the alpha parameter for Focal Loss.

        Args:
            graphs (List[List[Data]]): The full training dataset.
            sample_size (int): Number of sequences to sample for estimation.

        Returns:
            torch.Tensor: Scalar weight representing (Negatives / Positives).
        """
    print("  Calculating class weights (Representative Sample)...")
    if len(graphs) == 0: return torch.tensor(1.0)

    # Take random indices or last N if len < sample_size
    indices = np.random.choice(len(graphs), min(len(graphs), sample_size), replace=False)

    total_pos = 0
    total_neg = 0

    for idx in indices:
        seq = graphs[idx]
        snap = seq[-1]
        labels = snap.y
        num_master = labels.shape[0] // 2
        labels = labels[:num_master]

        valid_mask = labels != cfg.LABEL_MASK_VALUE
        valid_labels = labels[valid_mask]

        total_pos += (valid_labels == 1).sum().item()
        total_neg += (valid_labels == 0).sum().item()

    if total_pos == 0:
        return torch.tensor(1.0)

    weight = total_neg / total_pos
    print(f"   Pos Weight: {weight:.4f} (Avg based on {len(indices)} windows)")
    return torch.tensor(weight, dtype=torch.float)


def get_temporal_subgraph_batch(snapshot_sequence, node_indices):
    """
        Extracts a temporal k-hop subgraph for a batch of target nodes.
        Maintains consistency across the double-layer (Master + Ghost) structure.

        Args:
            snapshot_sequence (List[Data]): List of T graph snapshots.
            node_indices (Tensor): Global indices of target nodes for the batch.

        Returns:
            tuple: (List[Data] sub_sequence, Tensor target_mapping)
        """
    reference_graph = snapshot_sequence[-1]
    num_total_nodes = reference_graph.num_nodes
    num_master = num_total_nodes // 2
    node_indices = node_indices.cpu()

    # k-hop neighborhood extraction
    subset, _, _, _ = k_hop_subgraph(
        node_idx=node_indices,
        num_hops=cfg.NEIGHBOR_HOPS,
        edge_index=reference_graph.edge_index.cpu(),
        relabel_nodes=False
    )

    # Ensure symmetry for Master/Ghost nodes
    subset_master = subset % num_master
    nodes_l0 = torch.unique(subset_master)
    nodes_l1 = nodes_l0 + num_master
    symmetric_subset = torch.cat([nodes_l0, nodes_l1])
    symmetric_subset = torch.sort(symmetric_subset)[0]

    # Map global indices to local subgraph indices
    old_to_new = torch.full((num_total_nodes,), -1, dtype=torch.long)
    old_to_new[symmetric_subset] = torch.arange(len(symmetric_subset))
    target_mapping = old_to_new[node_indices]
    target_mapping = target_mapping[target_mapping >= 0]

    sub_sequence = []
    for data in snapshot_sequence:
        current_edge_index, _ = subgraph(
            subset=symmetric_subset,
            edge_index=data.edge_index.cpu(),
            relabel_nodes=True,
            num_nodes=num_total_nodes
        )
        sub_data = Data(
            x=data.x[symmetric_subset].cpu(),
            edge_index=current_edge_index,
            y=data.y[symmetric_subset].cpu()
        )
        sub_sequence.append(sub_data)

    return sub_sequence, target_mapping


def apply_node_isolation(sub_sequence, isolation_rate=0.5):
    """
        Applies Stochastic Node Isolation (edge dropout) to regularize training.
        Randomly removes connections for a subset of nodes in each snapshot.

        Args:
            sub_sequence (List[Data]): Temporal graph sequence.
            isolation_rate (float): Probability of isolating a node.

        Returns:
            List[Data]: Regularized sequence.
        """
    for snapshot in sub_sequence:
        if snapshot.edge_index.size(1) == 0: continue

        num_nodes = snapshot.num_nodes
        num_isolate = int(num_nodes * isolation_rate)
        if num_isolate == 0: continue

        perm = torch.randperm(num_nodes)
        isolate_indices = perm[:num_isolate]

        keep_mask = torch.ones(num_nodes, dtype=torch.bool)
        keep_mask[isolate_indices] = False

        src, dst = snapshot.edge_index
        edge_mask = keep_mask[src] & keep_mask[dst]
        snapshot.edge_index = snapshot.edge_index[:, edge_mask]

    return sub_sequence


class NodeBatchDataset(Dataset):
    """
        PyTorch Dataset wrapper for lists of PyG Data sequences.
        """
    def __init__(self, sequences, split_idx_range=None):
        if split_idx_range:
            start, end = split_idx_range
            self.sequences = sequences[start:end]
        else:
            self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


# --- Training Loop ---
def train_epoch(model, dataset, optimizer, criterion, device):
    """
        Executes one training epoch with gradient accumulation.
        """
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    num_batches_processed = 0
    accumulation_counter = 0

    seq_indices = torch.randperm(len(dataset)).tolist()

    for seq_idx in tqdm(seq_indices, desc="Training", leave=False):
        full_sequence = dataset[seq_idx]
        target_snapshot = full_sequence[-1]

        num_total_nodes = target_snapshot.num_nodes
        num_master = num_total_nodes // 2
        all_indices = torch.arange(num_total_nodes)

        # Identify valid training nodes (Active & Not Masked)
        valid_mask = (target_snapshot.y != cfg.LABEL_MASK_VALUE) & (all_indices < num_master)
        valid_node_indices = torch.where(valid_mask)[0]

        if len(valid_node_indices) == 0: continue
        valid_node_indices = valid_node_indices[torch.randperm(valid_node_indices.size(0))]

        # Mini-batch processing
        for i in range(0, len(valid_node_indices), cfg.BATCH_SIZE):
            batch_node_idx = valid_node_indices[i: i + cfg.BATCH_SIZE]

            try:
                sub_sequence, target_mapping = get_temporal_subgraph_batch(full_sequence, batch_node_idx)
            except Exception as e:
                continue

            sub_sequence = apply_node_isolation(sub_sequence, cfg.NODE_ISOLATION_RATE)
            sub_sequence = [d.to(device, non_blocking=True) for d in sub_sequence]
            target_mapping = target_mapping.to(device, non_blocking=True)

            num_subgraph_nodes = sub_sequence[-1].num_nodes
            num_logits = num_subgraph_nodes // 2

            # Sanity check mapping
            valid_map_mask = target_mapping < num_logits
            target_mapping = target_mapping[valid_map_mask]

            if len(target_mapping) == 0:
                del sub_sequence, target_mapping
                continue

            batch_labels = sub_sequence[-1].y[target_mapping]
            mask = (batch_labels != cfg.LABEL_MASK_VALUE)
            if mask.sum().item() == 0:
                del sub_sequence, target_mapping, batch_labels, mask
                continue

            # Forward Pass
            if accumulation_counter == 0:
                optimizer.zero_grad(set_to_none=True)

            logits = model(sub_sequence)
            target_logits = logits[target_mapping]
            masked_logits = target_logits[mask]
            masked_labels = batch_labels[mask].float().unsqueeze(1)
            masked_labels = torch.clamp(masked_labels, 0.0, 1.0)

            loss = criterion(masked_logits, masked_labels)
            loss = loss / cfg.ACCUMULATION_STEPS
            loss.backward()

            accumulation_counter += 1

            if accumulation_counter >= cfg.ACCUMULATION_STEPS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                accumulation_counter = 0

            total_loss += loss.item() * cfg.ACCUMULATION_STEPS
            preds_np = torch.sigmoid(masked_logits).detach().cpu().numpy().flatten()
            lbls_np = masked_labels.detach().cpu().numpy().flatten()
            all_preds.extend(preds_np.tolist())
            all_labels.extend(lbls_np.tolist())

            del sub_sequence, target_mapping, logits, target_logits, loss
            num_batches_processed += 1
            if num_batches_processed % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        if device.type == 'cuda': torch.cuda.empty_cache()

    if len(all_labels) == 0: return 0.0, 0.5, 0.0
    avg_loss = total_loss / max(num_batches_processed, 1)

    try:
        if len(np.unique(all_labels)) < 2:
            auc = 0.5
            f1 = 0.0
        else:
            auc = roc_auc_score(all_labels, all_preds)
            # Consistent 0.5 Threshold
            binary_preds = (np.array(all_preds) > 0.5).astype(int)
            f1 = f1_score(all_labels, binary_preds)
    except:
        auc, f1 = 0.5, 0.0

    return avg_loss, auc, f1


@torch.no_grad()
def validate_epoch(model, dataset, criterion, device):
    """
        Executes one validation epoch.
        """
    model.eval()
    all_preds, all_labels = [], []

    for seq_idx in tqdm(range(len(dataset)), desc="Validation", leave=False):
        full_sequence = dataset[seq_idx]
        target_snapshot = full_sequence[-1]

        num_total_nodes = target_snapshot.num_nodes
        num_master = num_total_nodes // 2
        all_indices = torch.arange(num_total_nodes)
        valid_mask = (target_snapshot.y != cfg.LABEL_MASK_VALUE) & (all_indices < num_master)
        valid_node_indices = torch.where(valid_mask)[0]

        if len(valid_node_indices) == 0: continue

        VAL_BATCH = cfg.BATCH_SIZE

        for i in range(0, len(valid_node_indices), VAL_BATCH):
            batch_node_idx = valid_node_indices[i: i + VAL_BATCH]
            try:
                sub_sequence, target_mapping = get_temporal_subgraph_batch(full_sequence, batch_node_idx)
            except:
                continue

            sub_sequence = [d.to(device, non_blocking=True) for d in sub_sequence]
            target_mapping = target_mapping.to(device, non_blocking=True)

            num_subgraph_nodes = sub_sequence[-1].num_nodes
            num_logits = num_subgraph_nodes // 2
            valid_map_mask = target_mapping < num_logits
            target_mapping = target_mapping[valid_map_mask]

            if len(target_mapping) == 0:
                del sub_sequence, target_mapping
                continue

            batch_labels = sub_sequence[-1].y[target_mapping]
            logits = model(sub_sequence)
            target_logits = logits[target_mapping]
            mask = (batch_labels != cfg.LABEL_MASK_VALUE)

            if mask.sum().item() == 0:
                del sub_sequence, target_mapping, logits
                continue

            preds_np = torch.sigmoid(target_logits[mask]).cpu().numpy().flatten()
            lbls_np = batch_labels[mask].cpu().numpy().flatten()
            all_preds.extend(preds_np.tolist())
            all_labels.extend(lbls_np.tolist())
            del sub_sequence, target_mapping, logits

        if device.type == 'cuda': torch.cuda.empty_cache()

    if len(all_labels) == 0: return 0.0, 0.5, 0.0
    try:
        if len(np.unique(all_labels)) < 2:
            auc = 0.5
            f1 = 0.0
        else:
            auc = roc_auc_score(all_labels, all_preds)
            # Consistent 0.5 Threshold
            binary_preds = (np.array(all_preds) > 0.5).astype(int)
            f1 = f1_score(all_labels, binary_preds)
    except:
        auc, f1 = 0.5, 0.0
    return 0.0, auc, f1


def main(override_save_dir=None, override_epochs=None):
    if override_save_dir:
        print(f"  DEBUG MODE: Using {override_save_dir}")
        cfg.GRAPH_DIR = override_save_dir
        cfg.CHECKPOINT_DIR = override_save_dir / "checkpoints"
        cfg.LOG_DIR = override_save_dir / "logs"

    epochs = override_epochs if override_epochs else cfg.NUM_EPOCHS

    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.empty_cache()
        print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")

    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    save_path = cfg.CHECKPOINT_DIR / "best_model_final.pt"

    print(" Loading data (Pre-packaged Windows)...")
    try:
        train_graphs = torch.load(cfg.GRAPH_DIR / "train_graphs.pt", map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f" Error: 'train_graphs.pt' not found in {cfg.GRAPH_DIR}")
        return

    if not train_graphs:
        print(" Warning: Train graphs are empty.")
        return

    # Chronological Split (80/20)
    total_sequences = len(train_graphs)
    train_count = int(total_sequences * 0.8)

    train_dataset = NodeBatchDataset(train_graphs, split_idx_range=(0, train_count))
    val_dataset = NodeBatchDataset(train_graphs, split_idx_range=(train_count, total_sequences))

    print(" Initializing Model...")
    model = DynamicTemporalGNN(
        num_features=cfg.NUM_FEATURES, gat_hidden=cfg.GAT_HIDDEN, gat_heads=cfg.GAT_HEADS,
        gat_out=cfg.GAT_OUT, lstm_hidden=cfg.LSTM_HIDDEN, lstm_layers=cfg.LSTM_LAYERS,
        mlp_hidden=cfg.MLP_HIDDEN, num_classes=cfg.NUM_CLASSES, dropout=cfg.DROPOUT
    ).to(cfg.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    # Calculate Alpha for Focal Loss
    pos_weight_tensor = compute_pos_weight(train_graphs).to(cfg.DEVICE)
    ratio = pos_weight_tensor.item()
    alpha_val = ratio / (1.0 + ratio)
    print(f" Ratio={ratio:.1f} -> Converted Alpha={alpha_val:.4f}")

    # implicit sigmoid by pytorch
    criterion = FocalLoss(alpha=alpha_val, gamma=2.0).to(cfg.DEVICE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    writer = SummaryWriter(cfg.LOG_DIR)

    best_val_auc = 0
    patience_counter = 0
    total_start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        train_loss, train_auc, train_f1 = train_epoch(model, train_dataset, optimizer, criterion, cfg.DEVICE)
        print(f"Train - Loss: {train_loss:.4f} | AUC: {train_auc:.4f} | F1: {train_f1:.4f}")

        val_loss, val_auc, val_f1 = validate_epoch(model, val_dataset, criterion, cfg.DEVICE)
        print(f"Val   - AUC: {val_auc:.4f} | F1: {val_f1:.4f}")

        epoch_time = time.time() - epoch_start_time
        print(f"⏱️ Epoch Time: {epoch_time:.2f} sec")
        writer.add_scalar("Time/Epoch", epoch_time, epoch)
        writer.add_scalars('AUC', {'train': train_auc, 'val': val_auc}, epoch)

        scheduler.step(val_auc)

        # Checkpointing
        if val_auc > best_val_auc + cfg.MIN_DELTA:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f" Best Model Saved!")
        else:
            patience_counter += 1
            print(f" Patience: {patience_counter}/{cfg.PATIENCE}")

        if patience_counter >= cfg.PATIENCE:
            print(" Early stopping.")
            break
        gc.collect()

    total_time = time.time() - total_start_time
    print("\n===============================")
    print(f" Total Training Time: {total_time:.2f} sec")
    print("===============================\n")

    import json
    results = {
        "total_training_time_sec": total_time,
        "best_val_auc": best_val_auc
    }
    with open(cfg.CHECKPOINT_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(" Results saved to training_results.json")

    writer.close()


if __name__ == "__main__":
    main()