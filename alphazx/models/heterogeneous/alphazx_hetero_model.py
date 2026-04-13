"""
Heterogeneous Graph Neural Network model for ZX-calculus simplification.

Uses HGTConv (Heterogeneous Graph Transformer) for type-aware message passing,
replacing the GPS (Graph-GPS) layers in the homogeneous model. The match diagram's
22 node types and 171 edge types are explicitly modeled, allowing the network to
learn different message functions for different structural relationships.

Architecture overview:
    1. FeatureEmbeddingLayer (shared with homogeneous model)
       - Embeds (node_type, phase) → dense vector, edge features → dense vector
       - Concatenates random-walk positional encoding
    2. Input projection: (emb_dim + pe_dim) → hidden_channels
    3. Flat → HeteroData conversion (using node_type and edge_type tensors)
    4. Shared HGTConv encoder (type-aware message passing)
    5. HeteroData → Flat conversion
    6. Policy branch: policy HGTConv → selectors (reused from homogeneous)
    7. Value branch: value HGTConv → attention aggregation → MLP

The model accepts the SAME flat tensor interface as AlphaZXModel:
    forward(x, edge_index, edge_attr, node_type, batch, pe, graph_ids, edge_type)
This ensures compatibility with the existing MCTS and training infrastructure.
The only addition is the edge_type parameter, which carries the edge type indices
needed for heterogeneous message routing.

Key advantages over the homogeneous (GPS) model:
    - Type-aware message passing: different transformations for Z-spiders vs X-spiders
      vs boundary nodes vs super nodes, without manual masking
    - Relation-aware edges: 'simple', 'inclusion', 'inclusion_super', 'simple_super'
      edges each get distinct attention parameters
    - Better structural inductive bias: the 22 node types and 4 relation categories
      provide strong prior knowledge about ZX-calculus structure

Trade-offs:
    - Does not use edge features (edge_size). HGTConv operates on edge types only.
      The edge_size information (number of ZX edges between match nodes) is lost.
    - More parameters per layer due to per-type projections (22 types × hidden²)
    - Conversion overhead from flat → hetero dictionaries at each forward pass
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import HGTConv

from alphazx.diagram.match import METADATA
from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.models.homogeneous.gps import FeatureEmbeddingLayer
from alphazx.models.homogeneous.rewrite_type_selector import RewriteTypeSelector
from alphazx.models.homogeneous.node_selector import NodeSelector
from alphazx.models.homogeneous.new_phase_selector import NewPhaseSelector
from alphazx.models.homogeneous.new_edge_selector import NewEdgeSelector
from alphazx.models.homogeneous.transfer_edge_selector import TransferEdgeSelector


# ---------------------------------------------------------------------------
# HGTConv building block
# ---------------------------------------------------------------------------

class HGTBlock(nn.Module):
    """Single HGTConv layer with per-type LayerNorm and dropout.

    Wraps PyG's HGTConv (which includes a learned skip connection) with:
    - Per-type LayerNorm for stable training
    - Dropout for regularization
    - Graceful handling of empty node types (passthrough)
    """

    def __init__(
        self,
        channels: int,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = HGTConv(channels, channels, metadata, heads=heads)
        # Per-type LayerNorm (each node type may have different statistics)
        self.norms = nn.ModuleDict({
            nt: nn.LayerNorm(channels) for nt in metadata[0]
        })
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        x_out = self.conv(x_dict, edge_index_dict)
        result = {}
        for nt in x_dict:
            if nt in x_out and x_out[nt] is not None and x_out[nt].size(0) > 0:
                result[nt] = self.dropout(self.norms[nt](x_out[nt]))
            else:
                # Passthrough for empty types or types that received no messages
                result[nt] = x_dict[nt]
        return result


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AlphaZXHeteroModel(nn.Module):
    """Heterogeneous AlphaZero model for ZX-calculus simplification.

    Drop-in replacement for AlphaZXModel. Uses HGTConv instead of GPS
    for type-aware message passing, while reusing the same policy selectors
    and value head architecture.

    The constructor accepts the same parameters as AlphaZXModel plus
    HGT-specific hyperparameters (num layers, heads, dropout).
    """

    NUM_ACTION_TYPES = 10  # Matches PolicyNetwork.NUM_ACTION_TYPES

    def __init__(
        self,
        num_node_types: int,
        num_possible_phases: int,
        num_possible_new_edges: int,
        node_embedding_channels: int,
        num_edge_embeddings: int,
        edge_embedding_channels: int,
        pe_in_channels: int,
        pe_out_channels: int,
        # HGT-specific hyperparameters
        hgt_num_shared_layers: int = 2,
        hgt_num_policy_layers: int = 2,
        hgt_num_value_layers: int = 2,
        hgt_heads: int = 4,
        hgt_dropout: float = 0.1,
        # Selector hyperparameters (match homogeneous defaults)
        rts_num_layers: int = 2,
        ns_num_layers: int = 2,
        nps_num_layers: int = 2,
        nes_num_layers: int = 2,
        tes_num_pooling_encoder_blocks: int = 1,
        tes_num_pooling_heads: int = 1,
        tes_pooling_layer_norm: bool = True,
        selector_dropout: float = 0.1,
        num_scoring_heads: int = 8,
    ):
        super().__init__()

        self.num_node_types = num_node_types
        self.num_possible_phases = num_possible_phases
        self.num_possible_new_edges = num_possible_new_edges
        hidden_channels = node_embedding_channels  # e.g. 64

        # --- Build HGT metadata from the match diagram METADATA ---
        self._node_type_names = list(METADATA.node_type_abbrevs)
        self._edge_type_triples = list(METADATA.edge_types)
        self._hgt_metadata = (self._node_type_names, self._edge_type_triples)

        # Reverse mapping: edge_type index → (src, rel, dst) triple
        self._idx_to_triple: list[tuple[str, str, str] | None] = [None] * (
            max(METADATA.edge_type_to_index_dict.values()) + 1
        )
        for triple, idx in METADATA.edge_type_to_index_dict.items():
            self._idx_to_triple[idx] = triple

        # --- Feature embedding (same as homogeneous model) ---
        emb_dim = node_embedding_channels + pe_out_channels  # e.g. 64 + 20 = 84
        self.emb = FeatureEmbeddingLayer(
            num_node_embeddings=num_node_types * num_possible_phases,
            node_embedding_out_channels=node_embedding_channels,
            num_edge_embeddings=num_edge_embeddings,
            edge_embedding_out_channels=edge_embedding_channels,
            pe_in_channels=pe_in_channels,
            pe_out_channels=pe_out_channels,
        )

        # --- Input projection: emb_dim → hidden_channels ---
        # Reduces dimensionality before HGTConv to control parameter count
        # (HGTConv params scale as O(num_node_types * channels²))
        self.input_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
        )

        # --- Shared HGTConv encoder ---
        self.shared_hgt = nn.ModuleList([
            HGTBlock(hidden_channels, self._hgt_metadata, hgt_heads, hgt_dropout)
            for _ in range(hgt_num_shared_layers)
        ])

        # --- Policy-specific HGTConv ---
        self.policy_hgt = nn.ModuleList([
            HGTBlock(hidden_channels, self._hgt_metadata, hgt_heads, hgt_dropout)
            for _ in range(hgt_num_policy_layers)
        ])

        # --- Value-specific HGTConv ---
        self.value_hgt = nn.ModuleList([
            HGTBlock(hidden_channels, self._hgt_metadata, hgt_heads, hgt_dropout)
            for _ in range(hgt_num_value_layers)
        ])

        # --- Policy selectors (reused from homogeneous architecture) ---
        # These operate on flat [num_nodes, C] tensors with node_type masking.
        # The selectors are architecture-agnostic — they just need good node
        # embeddings. The HGTConv encoder produces type-enriched embeddings.
        T = self.NUM_ACTION_TYPES

        self.rewrite_type_selector = RewriteTypeSelector(
            hidden_channels, T, rts_num_layers, selector_dropout, num_scoring_heads,
        )
        self.node_selector = NodeSelector(
            hidden_channels, T, ns_num_layers, selector_dropout, num_scoring_heads,
        )
        self.new_phase_selector = NewPhaseSelector(
            hidden_channels, T, num_possible_phases, nps_num_layers, selector_dropout,
        )
        self.new_edge_selector = NewEdgeSelector(
            hidden_channels, T, num_possible_new_edges, nes_num_layers, selector_dropout,
        )
        self.transfer_edge_selector = TransferEdgeSelector(
            hidden_channels, num_node_types, T,
            tes_num_pooling_encoder_blocks, tes_num_pooling_heads,
            tes_pooling_layer_norm, selector_dropout,
        )

        # --- Value head ---
        self.value_attention = pyg.nn.aggr.AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
            ),
            nn=nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
            ),
        )
        self.value_ff = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    # -------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_type: torch.Tensor,
        batch: torch.Tensor,
        pe: torch.Tensor,
        graph_ids: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> tuple[AlphaZXDistributionParams, torch.Tensor]:
        """Forward pass with same interface as AlphaZXModel.

        :param x: [num_nodes] Node feature indices (combined type+phase index).
        :param edge_index: [2, num_edges] Edge connectivity.
        :param edge_attr: [num_edges] Edge feature indices (combined type+size index).
        :param node_type: [num_nodes] Node type indices (0-21).
        :param batch: [num_nodes] Graph membership for batched processing.
        :param pe: [num_nodes, pe_dim] Random-walk positional encoding.
        :param graph_ids: [num_graphs] Diagram IDs.
        :param edge_type: [num_edges] Edge type indices (0-170). Required for
                          heterogeneous message routing. Available from data.edge_type.
        :return: (AlphaZXDistributionParams, value) same as AlphaZXModel.
        """
        assert edge_type is not None, (
            "AlphaZXHeteroModel requires edge_type tensor for heterogeneous "
            "message routing. Ensure data.edge_type is passed through."
        )

        num_nodes = x.size(0)
        device = x.device

        # 1. Embed features (same as homogeneous)
        x_emb, edge_attr_emb = self.emb(x, edge_attr, pe)
        # x_emb: [num_nodes, emb_dim=84], edge_attr_emb: [num_edges, edge_emb_channels]

        # 2. Project to HGT dimension
        x_proj = self.input_proj(x_emb)
        # x_proj: [num_nodes, hidden_channels=64]

        # 3. Convert flat tensors to per-type dictionaries
        x_dict, edge_index_dict, local_indices = self._flat_to_hetero(x_proj, edge_index, node_type, edge_type)

        # 4. Shared HGTConv encoding
        for block in self.shared_hgt:
            x_dict = block(x_dict, edge_index_dict)

        # 5. Policy branch
        policy_x_dict = {k: v.clone() for k, v in x_dict.items()}
        for block in self.policy_hgt:
            policy_x_dict = block(policy_x_dict, edge_index_dict)
        policy_x = self._hetero_to_flat(policy_x_dict, node_type, num_nodes, device)

        # 6. Value branch
        value_x_dict = x_dict  # no clone needed — shared branch is done
        for block in self.value_hgt:
            value_x_dict = block(value_x_dict, edge_index_dict)
        value_x = self._hetero_to_flat(value_x_dict, node_type, num_nodes, device)

        # 7. Policy selectors (operate on flat embeddings, same as homogeneous)
        mixture_probs = self.rewrite_type_selector(policy_x, node_type, batch)
        node_probs = self.node_selector(policy_x, node_type, batch)
        phase_probs = self.new_phase_selector(policy_x, node_type, batch)
        edge_probs = self.new_edge_selector(policy_x, node_type, batch)
        transfer_probs = self.transfer_edge_selector(policy_x, edge_index, node_type, batch)

        policy = AlphaZXDistributionParams(
            graph_ids=graph_ids,
            mixture_dist_probs=mixture_probs,
            node_dist_probs=node_probs,
            phase_dist_probs=phase_probs,
            new_edge_dist_probs=edge_probs,
            transfer_edge_dist_probs=transfer_probs,
        )

        # 8. Value head
        value = self.value_attention(value_x, batch)
        value = self.value_ff(value)
        value = torch.tanh(value)

        return policy, value

    # -------------------------------------------------------------------
    # Flat ↔ Hetero conversions
    # -------------------------------------------------------------------

    def _flat_to_hetero(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_type: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[tuple[str, str, str], torch.Tensor],
        torch.Tensor,
    ]:
        """Convert flat batched tensors to per-type dictionaries for HGTConv.

        :param x: [num_nodes, C] Node embeddings.
        :param edge_index: [2, num_edges] Edge connectivity (global indices).
        :param node_type: [num_nodes] Node type indices (0-21).
        :param edge_type: [num_edges] Edge type indices (0-170).
        :return: (x_dict, edge_index_dict, local_indices)
            - x_dict: {type_name: [n_type, C]} per-type node features
            - edge_index_dict: {(src, rel, dst): [2, n_edges]} per-type edges
            - local_indices: [num_nodes] mapping global → local index within type
        """
        device = x.device
        num_nodes = x.size(0)
        C = x.size(1)

        # Build per-type node features and local index mapping
        local_indices = torch.zeros(num_nodes, dtype=torch.long, device=device)
        x_dict: dict[str, torch.Tensor] = {}

        for i, name in enumerate(self._node_type_names):
            mask = (node_type == i)
            count = mask.sum().item()
            if count > 0:
                x_dict[name] = x[mask]
                # Assign consecutive local indices within this type
                local_indices[mask] = torch.arange(count, device=device)
            else:
                x_dict[name] = x.new_zeros(0, C)

        # Build per-type edge indices (remap global → local)
        src_global = edge_index[0]
        dst_global = edge_index[1]
        src_local = local_indices[src_global]
        dst_local = local_indices[dst_global]

        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}

        if edge_type.numel() > 0:
            for t_idx in edge_type.unique().tolist():
                triple = self._idx_to_triple[t_idx]
                if triple is None:
                    continue
                mask = (edge_type == t_idx)
                edge_index_dict[triple] = torch.stack([
                    src_local[mask], dst_local[mask],
                ])

        return x_dict, edge_index_dict, local_indices

    def _hetero_to_flat(
        self,
        x_dict: dict[str, torch.Tensor],
        node_type: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert per-type embeddings back to flat tensor preserving original order.

        :param x_dict: {type_name: [n_type, C]} per-type node features.
        :param node_type: [num_nodes] original node type indices.
        :param num_nodes: total number of nodes.
        :param device: target device.
        :return: [num_nodes, C] flat node embedding tensor.
        """
        # Determine channels from any non-empty entry
        C = 0
        for name in x_dict:
            if x_dict[name].size(0) > 0:
                C = x_dict[name].size(1)
                break

        x_flat = torch.zeros(num_nodes, C, device=device, dtype=torch.float32)

        for i, name in enumerate(self._node_type_names):
            if name not in x_dict or x_dict[name].size(0) == 0:
                continue
            mask = (node_type == i)
            x_flat[mask] = x_dict[name]

        return x_flat
