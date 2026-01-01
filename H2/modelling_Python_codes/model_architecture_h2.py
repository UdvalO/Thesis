# ==============================================================================
# model_architecture_h2.py (H2: Implicit Topology Architecture)
#
# PURPOSE:
#   Defines the Dynamic Temporal GNN used for Hypothesis 2.
#   This architecture is specifically optimized for "Implicit" topologies,
#   incorporating residual connections and supra-graph pooling to handle
#   the dynamic nature of the k-NN graphs.
#
# ARCHITECTURE (Thesis Sec 3.3 & Appendix B):
#   1. TemporalGAT (With Residuals):
#      - Enhances the standard GAT with residual skip connections (ResGAT)
#        to prevent gradient vanishing during the training of dynamic structures.
#      - "To stabilize training... residual connections were added to the GAT layer."
#
#   2. Unidirectional LSTM (Thesis Appendix B):
#      - Strictly matches the dimension constraints of the Reference Model's
#        Figure 5 (Output Dim=20).
#
#   3. Supra-Graph Pooling (Thesis Sec 3.3.1):
#      - Explicitly handles the dual-layer structure (Geo + Lender) by
#        averaging the latent representations of the two identity nodes
#        before the final classification step.
#
# INPUTS:
#   - snapshot_sequence: List of PyG Data objects (T=6 snapshots).
#
# OUTPUTS:
#   - logits: Raw risk scores [Batch_Size, 1].
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import config_h2 as cfg
from torch.utils.checkpoint import checkpoint


# --- 1. TemporalGAT (With Residuals) ---
class TemporalGAT(nn.Module):
    """
        GAT Module with Residual Skip Connections.

        Methodology:
            - Unlike the H1 implementation (which strictly followed the Reference Model's
              simpler GAT), H2 adopts a ResGAT structure.
            - Eq: h' = Activation(GAT(h, edge_index) + Linear(h))
            - Justification: Stabilizes convergence on dynamic graphs where edges change frequently.

        Args:
            in_channels (int): Input feature dimension.
            hidden_channels (int): Hidden GAT dimension.
            out_channels (int): Output dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dropout=0.2):
        super().__init__()
        # Layer 1
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.lin1 = nn.Linear(in_channels, hidden_channels * num_heads)

        # Layer 2
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.lin2 = nn.Linear(hidden_channels * num_heads, out_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # Block 1: Residual GAT
        x_in = x
        x = self.dropout(x)
        x = self.conv1(x, edge_index) + self.lin1(x_in)
        x = F.elu(x)

        # Block 2: Residual GAT
        x_in = x
        x = self.dropout(x)
        x = self.conv2(x, edge_index) + self.lin2(x_in)

        return x


# --- 2. Temporal Attention ---
class TemporalAttention(nn.Module):
    """
        Node-Independent Temporal Attention Mechanism.

        Methodology (Thesis Eq 11-13):
            - Identical to H1, utilizing the Bahdanau-style MLP to compute
              individualized temporal importance weights for each borrower.
        """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_outputs):
        # lstm_outputs: [Batch, Seq, Hidden]
        scores = self.attention_net(lstm_outputs)
        attention_weights = F.softmax(scores.squeeze(-1), dim=1)
        context = (lstm_outputs * attention_weights.unsqueeze(-1)).sum(dim=1)
        return context, attention_weights


# --- 3. Main Model ---
class DynamicTemporalGNN(nn.Module):
    """
        The IM-DYMGNN (Implicit Dynamic Multilayer GNN) for Hypothesis 2.

        Methodology:
            - Orchestrates the spatial-temporal learning pipeline.
            - Integrates 'Supra-Graph Pooling' in the forward pass to merge
              the dual-layer representations (Geo + Lender).

        Args:
            num_features (int): Input feature dimension.
            gat_hidden (int): GAT hidden dimension (32).
            gat_heads (int): Number of heads (2).
            gat_out (int): GAT output dimension (20).
            lstm_hidden (int): LSTM hidden dimension (20).
            lstm_layers (int): Number of LSTM layers (2).
            mlp_hidden (int): Decoder hidden dimension (10).
            num_classes (int): Output classes (1).
            dropout (float): Dropout rate (0.5).
            learn_init_states (bool): Whether to learn initial LSTM states.
        """
    def __init__(self, num_features, gat_hidden, gat_heads, gat_out, lstm_hidden,
                 lstm_layers, mlp_hidden, num_classes, dropout=0.5, learn_init_states=True):
        super().__init__()

        # 1. GAT (With Residuals)
        self.gat = TemporalGAT(num_features, gat_hidden, gat_out, gat_heads, dropout)
        self.gat_norm = nn.LayerNorm(gat_out)

        # 2. LSTM (Unidirectional)
        # Output dimension is strictly 20.
        self.lstm = nn.LSTM(
            input_size=gat_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False  # Strict Paper Alignment
        )

        lstm_output_dim = lstm_hidden  # 20

        self.lstm_norm = nn.LayerNorm(lstm_output_dim)

        if learn_init_states:
            self.h0 = nn.Parameter(torch.zeros(lstm_layers, 1, lstm_hidden))
            self.c0 = nn.Parameter(torch.zeros(lstm_layers, 1, lstm_hidden))
        else:
            self.register_buffer('h0', torch.zeros(lstm_layers, 1, lstm_hidden))
            self.register_buffer('c0', torch.zeros(lstm_layers, 1, lstm_hidden))

        # 3. Attention
        self.attention = TemporalAttention(lstm_output_dim)

        # 4. Classifier (Decoder)
        # Strictly matches Reference Model (Thesis Figure 5): Dense(20->10) -> ReLU -> Dropout -> Dense(10->1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 10),  # 20 -> 10
            nn.ReLU(),
            nn.Dropout(dropout),  # p=0.5
            nn.Linear(10, num_classes)  # 10 -> 1
        )

    def forward(self, snapshot_sequence, return_attention=False):
        """
                Forward pass with Supra-Graph Pooling.

                Methodology:
                    - Iterate through dynamic snapshots (T=6).
                    - Compute Spatial Embeddings via ResGAT.
                    - Compute Temporal Embeddings via LSTM.
                    - Pool the Dual-Layer Representation (Geo + Lender).
                    - Classify.
                """
        embeddings = []

        # --- Pass 1: GAT ---
        for snapshot in snapshot_sequence:
            node_emb = self.gat(snapshot.x, snapshot.edge_index)
            node_emb = self.gat_norm(node_emb)
            embeddings.append(node_emb)

        sequence = torch.stack(embeddings, dim=1)  # [Batch, Seq, Features]
        batch_size = sequence.size(0)

        # Expand hidden states for batch
        h0 = self.h0.repeat(1, batch_size, 1).contiguous()
        c0 = self.c0.repeat(1, batch_size, 1).contiguous()

        # --- Pass 2: LSTM ---
        def lstm_forward(seq, h, c):
            out, _ = self.lstm(seq, (h, c))
            return out

        if self.training and sequence.requires_grad:
            # Gradient Checkpointing for memory efficiency
            lstm_out = checkpoint(lstm_forward, sequence, h0, c0, use_reentrant=False)
        else:
            lstm_out, _ = self.lstm(sequence, (h0, c0))

        lstm_out = self.lstm_norm(lstm_out)

        # --- Pass 3: Attention ---
        context, attn_weights = self.attention(lstm_out)

        # --- Pass 4: Supra-Graph Pooling (H2 Specific) ---
        # The input x had shape [2N, F] because we stacked Geo and Lender nodes.
        # Now we separate them and average their embeddings.
        # This "fuses" the two implicit views of the borrower.
        num_master = context.shape[0] // 2
        context_layer0 = context[:num_master]
        context_layer1 = context[num_master:]

        # Average the representation from both layers (Geo + Lender)
        context_pooled = (context_layer0 + context_layer1) / 2

        # --- Pass 5: Classification ---
        logits = self.classifier(context_pooled)

        if return_attention:
            return logits, attn_weights
        return logits


if __name__ == "__main__":
    # Test instantiation
    model = DynamicTemporalGNN(
        num_features=cfg.NUM_FEATURES, gat_hidden=cfg.GAT_HIDDEN, gat_heads=cfg.GAT_HEADS,
        gat_out=cfg.GAT_OUT, lstm_hidden=cfg.LSTM_HIDDEN, lstm_layers=cfg.LSTM_LAYERS,
        mlp_hidden=cfg.MLP_HIDDEN, num_classes=cfg.NUM_CLASSES, dropout=cfg.DROPOUT
    )
    print(model)