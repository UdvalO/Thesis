# ==============================================================================
# attention_weights.py (H1)
#
# PURPOSE:
#   Load a trained DYMGNN model, extract temporal attention weights from a test graph,
#   and visualize attention patterns both in aggregate and for sample nodes.
#
# STEPS:
#   1. Setup device and checkpoint path.
#   2. Define hook mechanism to capture attention logits during forward pass.
#   3. Load test graph data, ensure proper feature shape, and attach static edges.
#   4. Initialize NeighborLoader for batching.
#   5. Load DYMGNN model weights from checkpoint and attach attention hook.
#   6. Forward pass through test loader to capture attention logits.
#   7. Apply softmax to logits to obtain attention weights.
#   8. Plot:
#       a. Aggregate attention curve (mean ± std across all nodes)
#       b. Individual attention profiles for 5 random sample nodes
#
# INPUTS:
#   - Test graph features: test_features.pt (on remote computer)
#   - Static edges: static_edge_index.pt (on remote computer)
#   - DYMGNN model checkpoint: best_model_h1.pt (on remote computer)
#   - Configuration: config_h1.py (on flash drive)
#
# OUTPUTS:
#   - Aggregate temporal attention plot: attention_plot.png (on flash drive)
#   - Sample node attention profiles: attention_samples.png (on flash drive)
# ==============================================================================

# --- 0. DEPENDENCIES ---
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import config_h1 as cfg
from model_architecture_h1 import DYMGNN

# --- 1. SETUP ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = cfg.GRAPH_SAVE_DIR / "best_model_h1.pt"

# --- 2. THE HOOK MECHANISM ---
attention_cache = {}


def get_attention_hook(name):
    """
        Returns a hook function to capture attention logits from the model during forward pass.

        Args:
            name (str): Key name to store logits in attention_cache.

        Returns:
            hook (function): Forward hook function
        """
    def hook(model, input, output):
        # Detach logits and store in attention_cache
        attention_cache[name] = output.detach()

    return hook

# --- 3. DATA AND MODEL LOADING FUNCTION ---
def load_data_and_model():
    """
        Loads test graph data, applies preprocessing, loads model weights,
        attaches attention hook, and prepares NeighborLoader.

        Returns:
            test_loader (NeighborLoader): Batched test graph loader
            model (DYMGNN): Loaded and ready-to-evaluate model
        """
    print("Loading test data...")
    test_graphs = torch.load(cfg.GRAPH_SAVE_DIR / "test_features.pt")
    target_graph = test_graphs[0]

    # --- Feature Shape Check ---
    # Transpose if Time dimension is first (expected: [Nodes, Time, Features])
    if target_graph.x.shape[0] == cfg.SEQUENCE_LENGTH and target_graph.x.shape[1] > 100:
        print(f"Transposing features from [Time, Nodes, F] to [Nodes, Time, F]...")
        target_graph.x = target_graph.x.permute(1, 0, 2)

    # --- Load Static Edges ---
    edge_path = cfg.GRAPH_SAVE_DIR / "static_edge_index.pt"
    if edge_path.exists():
        static_edge_index = torch.load(edge_path, weights_only=False).long()

        # Safety check: clip edges if they exceed node count
        num_nodes = target_graph.x.shape[0]
        if static_edge_index.max() >= num_nodes:
            print(f"Clipping edges to match node count ({num_nodes})...")
            mask = (static_edge_index[0] < num_nodes) & (static_edge_index[1] < num_nodes)
            static_edge_index = static_edge_index[:, mask]

        target_graph.edge_index = static_edge_index
    else:
        raise FileNotFoundError(f"Could not find {edge_path}")

    # --- Neighbor Loader for batching ---
    test_loader = NeighborLoader(
        target_graph,
        num_neighbors=cfg.NUM_NEIGHBORS,
        batch_size=4096,
        input_nodes=None,
        shuffle=False
    )

    # --- Load Model ---
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = DYMGNN(
        num_features=cfg.NUM_INPUT_FEATURES,
        hidden_dim=cfg.HIDDEN_DIM,
        num_heads=cfg.NUM_HEADS,
        dropout=0.0
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.eval()

    # Attach hook to attention_net to capture attention logits
    model.attention_net.register_forward_hook(get_attention_hook('attn'))

    return test_loader, model

# --- 4. EXTRACT ATTENTION WEIGHTS ---
def extract_attention_weights(loader, model):
    """
        Forward pass through loader to extract attention weights.

        Args:
            loader (NeighborLoader): Batched test graph loader
            model (DYMGNN): Loaded model

        Returns:
            all_alphas (np.ndarray): Attention weights [Nodes x Time]
        """
    all_alphas = []
    print("Extracting attention weights...")

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(DEVICE)
            x, edge_index = batch.x, batch.edge_index

            # Permute for model if needed: [Nodes, Time, F] -> [Time, Nodes, F]
            if x.dim() == 3 and x.shape[0] != cfg.SEQUENCE_LENGTH:
                x = x.permute(1, 0, 2)

            # Forward pass triggers attention hook
            _ = model(x, edge_index)

            if 'attn' in attention_cache:
                scores = attention_cache['attn']  # Raw Logits from hook

                alpha = F.softmax(scores, dim=1) # Convert logits to probability

                all_alphas.append(alpha.cpu().numpy())
            else:
                print("Hook failed to capture attention.")
                return None

    return np.concatenate(all_alphas, axis=0)

# --- 5. PLOT ATTENTION CURVES ---
def plot_attention_curve(alphas):
    """
        Visualize attention weights as:
            a) Aggregate curve (mean ± std)
            b) Individual profiles for sample nodes

        Args:
            alphas (np.ndarray): Attention weights [Nodes x Time]
        """
    if alphas is None:
        return

    # --- Prepare 2D Attention Array ---
    if alphas.ndim == 3 and alphas.shape[-1] == 1:
        alphas_2d = alphas.squeeze(-1)
    else:
        alphas_2d = alphas

    T = alphas_2d.shape[1]
    time_steps = list(range(T))
    labels = [f"T-{T-1-i}" if i < T-1 else "T" for i in range(T)]

    mean_weights = np.mean(alphas_2d, axis=0)
    std_weights = np.std(alphas_2d, axis=0)

    # ============================================================
    # PLOT 1: Aggregate Temporal Attention (Mean ± Std)
    # ============================================================
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    plt.plot(
        time_steps,
        mean_weights,
        marker="o",
        linewidth=3,
        label="Mean Attention"
    )

    plt.fill_between(
        time_steps,
        np.maximum(mean_weights - std_weights, 0),
        np.minimum(mean_weights + std_weights, 1),
        alpha=0.2,
        label="±1 Std Dev"
    )
    plt.title(
        "Temporal Attention Distribution (Mean ± Std Across Nodes)",
        fontsize=14,
        fontweight="bold"
    )
    plt.xlabel("Sequence Position")
    plt.ylabel("Attention Weight")
    plt.xticks(time_steps, labels)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    save_path = cfg.PROJECT_ROOT / "images" / "attention_plot.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Aggregate plot saved to {save_path}")
    plt.close()

    # ============================================================
    # PLOT 2: Individual Attention Profiles (5 Sample Nodes)
    # ============================================================
    plt.figure(figsize=(10, 6))

    num_nodes = alphas_2d.shape[0]
    sampled_indices = np.random.choice(num_nodes, 5, replace=False)

    for node_id in sampled_indices:
        plt.plot(
            time_steps,
            alphas_2d[node_id],
            marker="o",
            linewidth=2,
            alpha=0.8,
            label=f"Node {node_id}"
        )

    plt.title(
        "Temporal Attention Profiles (5 Sample Nodes)",
        fontsize=14,
        fontweight="bold"
    )
    plt.xlabel("Sequence Position")
    plt.ylabel("Attention Weight")
    plt.xticks(time_steps, labels)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    save_path = cfg.PROJECT_ROOT / "images" / "attention_samples.png"
    plt.savefig(save_path, dpi=300)
    print(f"Sample nodes plot saved to {save_path}")
    plt.close()

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    loader, model = load_data_and_model()
    alphas = extract_attention_weights(loader, model)
    plot_attention_curve(alphas)