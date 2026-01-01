# ==============================================================================
# model_architecture_h1.py (H1)
#
# PURPOSE:
#   Defines the DYMGNN architecture used for the H1 experiments.
#   This implementation strictly adheres to the Reference Model's core logic
#   (GAT -> LSTM -> Attention -> Decoder) but incorporates specific stability
#   modifications required for convergence on consumer hardware.
#
# ARCHITECTURE (Thesis Section 2.2 & 3.2.2):
#   1. Topological Layer (GAT): Captures spatial dependencies between borrowers.
#      - Strictness: No residual connections, consistent with Zandi et al. (2025).
#   2. Temporal Layer (LSTM): Models the sequential evolution of borrower states.
#   3. Temporal Attention: A "Node-Independent" mechanism (Thesis Eq. 11-13).
#      - Modification: Replaces the Reference Model's "Global Pooling" (Eq. 7)
#        with a Bahdanau-style MLP to prevent signal oversmoothing and handle
#        variable batch sizes.
#   4. Decoder: A 2-layer MLP projecting embeddings to default probability.
#
# INPUTS:
#   - train_features.pt
#   - test_features.pt
#   - static_edge_index.pt
#   - metadata.pt
#
# OUTPUTS:
#   - best_model_h1.pt
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class DYMGNN(nn.Module):
    """
        Implements the Dynamic Multilayer Graph Neural Network (H1 Configuration).

        Methodology:
            - Topological Layer (GAT): strictly follows the Reference Model's aggregation
              logic (averaging heads, no residuals) as defined in Section 2.2.1.
            - Temporal Layer (LSTM): models sequential evolution (Eq. 6).
            - Attention: Replaces the Reference Model's global pooling (Eq. 7) with the
              "Node-Independent" Bahdanau-style MLP (Eq. 11) to prevent oversmoothing
              and handle variable batch sizes.
            - Decoder: Replicates the 2-layer MLP structure (20->10->1) described in
              Section 2.2.4.

        Args:
            num_features (int): Dimensionality of input node features.
            hidden_dim (int): Dimensionality of hidden states (e.g., 32).
            num_heads (int): Number of attention heads for GAT layer (K=2).
            dropout (float): Dropout probability (default 0.5).
        """
    def __init__(self, num_features, hidden_dim, num_heads, dropout=0.5):
        super(DYMGNN, self).__init__()
        self.dropout = dropout

        # --- 1. Topological Embedding (GAT) ---
        # Thesis Sec 2.2.1: "Aggregated using averaging operations"
        self.gat = GATConv(
            in_channels=num_features,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=False,  # Averaging heads as per Reference Model logic
            dropout=dropout
        )

        # --- 2. Temporal Embedding (LSTM) ---
        # Thesis Sec 2.2.2: Models "sequential evolution of the borrower's state"
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # --- 3. Attention Mechanism (Node-Independent) ---
        # Thesis Sec 3.2.2: "Computes attention weights for each node individually
        # using a shared Bahdanau-style MLP" (Eq. 11).
        # Solves "Global Oversmoothing" of the Reference Model (Thesis Eq. 7).
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(), # The tanh activation ensures numerical stability
            nn.Linear(hidden_dim // 2, 1)
        )

        # --- 4. Decoder ---
        # Thesis Sec 2.2.4: "Dense layer reducing... 20 to 10... followed by 10 to 1"
        self.decoder = nn.Sequential(
            # Layer 1: Matches "Dense Layer (20, 10)" ratio
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2: Matches "Dense Layer (10, 1)"
            # This restores the Strict 2-layer structure of the paper.
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_sequence, edge_index):
        """
        Performs the forward pass: Spatial GAT -> Temporal LSTM -> Attention -> Decoder.

        Methodology:
            - Spatial Aggregation: Computes GAT embeddings for each snapshot $t$ (Eq. 5).
            - Temporal Evolution: Processes the sequence of spatial embeddings via LSTM.
            - Dynamic Attention: Computes node-specific weights $\alpha_{i}^{(t)}$ (Eq. 12)
              and aggregates context vectors $h_{i}^{att}$ (Eq. 13).

        Args:
            x_sequence (Tensor): Input features [Time, Nodes, Features] or [Batch, Time, Nodes, Feats].
            edge_index (Tensor): Graph connectivity [2, Edges].

        Returns:
            Tensor: Raw logits [Nodes, 1] (unnormalized risk scores).
        """

        # Handle batch dimension if present (e.g. from NeighborLoader)
        if x_sequence.dim() == 4:
            x_sequence = x_sequence.squeeze(0)

        T, NumNodes, F_in = x_sequence.shape
        embeddings = []

        # 1. Topological Loop (GAT)
        for t in range(T):
            x_t = x_sequence[t]
            h_t = self.gat(x_t, edge_index)
            h_t = F.elu(h_t)
            embeddings.append(h_t)

        # Stack: (Nodes, Time, Hidden)
        seq_tensor = torch.stack(embeddings, dim=1)

        # 2. Temporal (LSTM)
        lstm_out, _ = self.lstm(seq_tensor)

        # 3. Attention Aggregation (
        scores = self.attention_net(lstm_out) # Raw score (Eq. 11)
        alpha = F.softmax(scores, dim=1) # Softmax Normalization (Eq 12)
        h_att = torch.sum(alpha * lstm_out, dim=1) # Weighted Sum (Eq 13)

        # 4. Decoder
        logits = self.decoder(h_att)

        return logits