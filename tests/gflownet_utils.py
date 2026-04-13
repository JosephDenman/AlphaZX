"""Shared utilities for GFlowNet tests."""

import torch

from alphazx.diagram import METADATA, POSSIBLE_PHASES, NUM_POSSIBLE_NEW_EDGES
from alphazx.models.homogeneous.alphazx_model import AlphaZXModel
from alphazx.shared.config import CircuitConfig
from alphazx.gflownet.config import GFlowNetConfig
from alphazx.gflownet.environment import ZXGFlowNetEnv
from alphazx.gflownet.policy import GFlowNetForwardPolicy

# Small model for fast tests
_NODE_EMB = 32
_EDGE_EMB = 8
_PE_DIM = 20


def make_model() -> AlphaZXModel:
    """Create a small AlphaZXModel suitable for testing."""
    return AlphaZXModel(
        num_node_types=len(METADATA.node_type_abbrevs),
        num_possible_phases=len(POSSIBLE_PHASES),
        num_possible_new_edges=NUM_POSSIBLE_NEW_EDGES,
        node_embedding_channels=_NODE_EMB,
        num_edge_embeddings=len(METADATA.edge_feat_to_index_dict),
        edge_embedding_channels=_EDGE_EMB,
        pe_in_channels=_PE_DIM,
        pe_out_channels=_PE_DIM,
    )


def make_circuit_config(**overrides) -> CircuitConfig:
    """Create a small CircuitConfig for fast testing."""
    defaults = dict(num_qubits=3, depth=3, min_initial_t_gates=1)
    defaults.update(overrides)
    return CircuitConfig(**defaults)


def make_gflownet_config(**overrides) -> GFlowNetConfig:
    """Create a GFlowNetConfig for fast testing."""
    defaults = dict(
        num_qubits=3, depth=3, min_initial_t_gates=1,
        trajectories_per_batch=2,
        max_trajectory_length=10,
    )
    defaults.update(overrides)
    return GFlowNetConfig(**defaults)


def make_env(config=None) -> ZXGFlowNetEnv:
    """Create a ZXGFlowNetEnv for testing."""
    if config is None:
        config = make_circuit_config()
    return ZXGFlowNetEnv(config)


def make_policy(model=None) -> GFlowNetForwardPolicy:
    """Create a GFlowNetForwardPolicy for testing."""
    if model is None:
        model = make_model()
    return GFlowNetForwardPolicy(model, pe_dim=_PE_DIM)
