"""
Comprehensive tests for alphazx.models.heterogeneous.alphazx_hetero_model.AlphaZXHeteroModel,
the heterogeneous Graph Transformer variant that uses HGTConv for type-aware message passing.

Tests mirror the structure of test_alphazx_model.py for the homogeneous model, and add
heterogeneous-specific tests for flat↔hetero conversion, edge type routing, and hparams
extraction/reconstruction.

Uses synthetic data to avoid the pre-existing pyzx/networkx incompatibility
(VertexType.BOUNDARY parsing error) that blocks clifford_zx_diagram.
"""

import pytest
import torch
import torch_geometric as pyg

from alphazx.diagram import METADATA, POSSIBLE_PHASES, NUM_POSSIBLE_NEW_EDGES
from alphazx.distributions import AlphaZXDistribution, AlphaZXDistributionParams
from alphazx.models.heterogeneous.alphazx_hetero_model import AlphaZXHeteroModel, HGTBlock
from alphazx.mcts.parallel_self_play import _extract_model_hparams, _build_model_from_hparams


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_NODE_TYPES = len(METADATA.node_type_abbrevs)
NUM_POSSIBLE_PHASES = len(POSSIBLE_PHASES)
# NUM_POSSIBLE_NEW_EDGES imported from alphazx.diagram
NODE_EMB_CHANNELS = 32
NUM_EDGE_EMBEDDINGS = len(METADATA.edge_feat_to_index_dict)
EDGE_EMB_CHANNELS = 8
PE_DIM = 20


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_model(**overrides):
    defaults = dict(
        num_node_types=NUM_NODE_TYPES,
        num_possible_phases=NUM_POSSIBLE_PHASES,
        num_possible_new_edges=NUM_POSSIBLE_NEW_EDGES,
        node_embedding_channels=NODE_EMB_CHANNELS,
        num_edge_embeddings=NUM_EDGE_EMBEDDINGS,
        edge_embedding_channels=EDGE_EMB_CHANNELS,
        pe_in_channels=PE_DIM,
        pe_out_channels=PE_DIM,
    )
    defaults.update(overrides)
    return AlphaZXHeteroModel(**defaults)


def _make_synthetic_data(num_nodes: int = 15, num_edges_per_type: int = 2, seed: int = 0):
    """Create a synthetic post-pre_process PyG Data object with valid edge_type routing.

    Builds a graph with nodes of diverse types and edges whose edge_type indices
    are consistent with the source/dest node types (matching METADATA triples).

    After pre_process, data has:
      - x: [N] int (combined node feature index)
      - edge_index: [2, E] long
      - edge_attr: [E] int (combined edge feature index)
      - node_type: [N] int
      - edge_type: [E] int
      - pe: [N, PE_DIM] float
      - id: int
    """
    rng = torch.Generator().manual_seed(seed)

    # Assign node types: spread across available types
    node_types = torch.randint(0, NUM_NODE_TYPES, (num_nodes,), generator=rng)

    # Build per-node-type index lists
    type_to_nodes: dict[int, list[int]] = {}
    for i in range(num_nodes):
        t = node_types[i].item()
        type_to_nodes.setdefault(t, []).append(i)

    # Build edges consistent with METADATA edge type triples
    src_list, dst_list, edge_type_list, edge_attr_list = [], [], [], []

    for triple, et_idx in METADATA.edge_type_to_index_dict.items():
        src_type_name, rel, dst_type_name = triple
        src_type_idx = METADATA.node_type_abbrev_index_dict[src_type_name]
        dst_type_idx = METADATA.node_type_abbrev_index_dict[dst_type_name]

        src_candidates = type_to_nodes.get(src_type_idx, [])
        dst_candidates = type_to_nodes.get(dst_type_idx, [])

        if not src_candidates or not dst_candidates:
            continue

        for _ in range(min(num_edges_per_type, len(src_candidates) * len(dst_candidates))):
            s = src_candidates[torch.randint(len(src_candidates), (1,), generator=rng).item()]
            d = dst_candidates[torch.randint(len(dst_candidates), (1,), generator=rng).item()]
            src_list.append(s)
            dst_list.append(d)
            edge_type_list.append(et_idx)
            # edge_attr: use a valid edge feature index
            # Find any valid (et_idx, size) pair in edge_feat_to_index_dict
            for (et, sz), feat_idx in METADATA.edge_feat_to_index_dict.items():
                if et == et_idx:
                    edge_attr_list.append(feat_idx)
                    break
            else:
                edge_attr_list.append(0)

    if not src_list:
        # Fallback: at least one edge
        src_list, dst_list = [0], [min(1, num_nodes - 1)]
        edge_type_list = [0]
        edge_attr_list = [0]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(edge_type_list, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)

    # Sort by destination node (required by TransferEdgeSelector's SetTransformer)
    perm = edge_index[1].argsort(stable=True)
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    # Node features: valid indices into node_feat_to_index_dict
    # Use the first valid feature for each node type
    x = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(num_nodes):
        nt = node_types[i].item()
        # Find any (nt, phase) -> feat_idx mapping
        for (nti, phase), feat_idx in METADATA.node_feat_to_index_dict.items():
            if nti == nt:
                x[i] = feat_idx
                break

    # Positional encoding
    pe = torch.randn(num_nodes, PE_DIM)

    data = pyg.data.Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=node_types,
        edge_type=edge_type,
        pe=pe,
        id=torch.tensor(seed),
    )
    return data


def _make_batch(batch_size=2, num_nodes=15, seed=0):
    data_list = [_make_synthetic_data(num_nodes=num_nodes, seed=seed + i) for i in range(batch_size)]
    return pyg.data.Batch.from_data_list(data_list)


def _run_model(model, batch):
    """Run the full model forward pass including edge_type."""
    return model(
        batch.x, batch.edge_index, batch.edge_attr, batch.node_type,
        batch.batch, batch.pe, batch.id,
        edge_type=batch.edge_type,
    )


# ===========================================================================
# TestHGTBlock
# ===========================================================================


class TestHGTBlock:
    """Tests for the HGTBlock building block."""

    def test_instantiation(self):
        hgt_metadata = (list(METADATA.node_type_abbrevs), list(METADATA.edge_types))
        block = HGTBlock(64, hgt_metadata, heads=4, dropout=0.1)
        assert isinstance(block, torch.nn.Module)

    def test_empty_type_passthrough(self):
        """HGTBlock should pass through embeddings for empty node types.

        Populates two node types with a self-edge between them so HGTConv
        has something to process, then verifies that all other types get
        their empty tensors passed through unchanged.
        """
        hgt_metadata = (list(METADATA.node_type_abbrevs), list(METADATA.edge_types))
        block = HGTBlock(64, hgt_metadata, heads=4, dropout=0.0)
        block.eval()

        type_names = list(METADATA.node_type_abbrevs)

        x_dict = {}
        for name in type_names:
            x_dict[name] = torch.zeros(0, 64)
        # Populate two types so at least one edge triple can exist
        first_type = type_names[0]
        x_dict[first_type] = torch.randn(3, 64)

        # Find a self-edge triple for first_type (e.g. ('b', 'simple', 'b'))
        edge_index_dict = {}
        for triple in METADATA.edge_types:
            if triple[0] == first_type and triple[2] == first_type:
                edge_index_dict[triple] = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
                break

        with torch.no_grad():
            out_dict = block(x_dict, edge_index_dict)

        assert out_dict[first_type].shape == (3, 64)
        for name in type_names:
            if name != first_type:
                assert out_dict[name].shape[0] == 0


# ===========================================================================
# TestAlphaZXHeteroModelForward
# ===========================================================================


class TestAlphaZXHeteroModelForward:
    """Basic forward pass tests for AlphaZXHeteroModel."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_model()
        self.model.eval()

    def test_returns_params_and_value(self):
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            result = _run_model(self.model, batch)
        assert len(result) == 2
        params, value = result
        assert isinstance(params, AlphaZXDistributionParams)
        assert isinstance(value, torch.Tensor)

    def test_value_shape(self):
        batch = _make_batch(batch_size=3)
        with torch.no_grad():
            params, value = _run_model(self.model, batch)
        assert value.shape == (3, 1)

    def test_value_is_finite(self):
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            _, value = _run_model(self.model, batch)
        assert torch.isfinite(value).all()

    def test_single_batch(self):
        batch = _make_batch(batch_size=1)
        with torch.no_grad():
            params, value = _run_model(self.model, batch)
        assert params.mixture_dist_probs.shape[0] == 1
        assert value.shape == (1, 1)

    def test_requires_edge_type(self):
        """Forward should raise AssertionError if edge_type is None."""
        batch = _make_batch(batch_size=1)
        with pytest.raises(AssertionError, match="edge_type"):
            self.model(
                batch.x, batch.edge_index, batch.edge_attr, batch.node_type,
                batch.batch, batch.pe, batch.id,
                edge_type=None,
            )


# ===========================================================================
# TestAlphaZXHeteroModelOutputConstraints
# ===========================================================================


class TestAlphaZXHeteroModelOutputConstraints:
    """Verify that model outputs satisfy probability constraints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_model()
        self.model.eval()
        self.batch = _make_batch(batch_size=2)
        with torch.no_grad():
            self.params, self.value = _run_model(self.model, self.batch)

    def test_mixture_probs_non_negative(self):
        assert (self.params.mixture_dist_probs >= 0).all()

    def test_mixture_probs_sum_to_one_or_zero(self):
        sums = self.params.mixture_dist_probs.sum(dim=-1)
        for s in sums:
            assert torch.isclose(s, torch.tensor(1.0), atol=1e-5) or s < 1e-5

    def test_node_probs_non_negative(self):
        assert (self.params.node_dist_probs >= 0).all()

    def test_node_probs_sum_to_one_or_zero(self):
        B, T, N = self.params.node_dist_probs.shape
        for b in range(B):
            for t in range(T):
                s = self.params.node_dist_probs[b, t].sum()
                assert torch.isclose(s, torch.tensor(1.0), atol=1e-5) or s < 1e-5

    def test_phase_probs_non_negative(self):
        assert (self.params.phase_dist_probs >= 0).all()

    def test_new_edge_probs_non_negative(self):
        assert (self.params.new_edge_dist_probs >= 0).all()

    def test_transfer_edge_probs_in_zero_one(self):
        te = self.params.transfer_edge_dist_probs
        assert (te >= 0).all() and (te <= 1).all()

    def test_no_nan_in_any_param(self):
        for field in self.params._fields:
            t = getattr(self.params, field)
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                assert not torch.isnan(t).any(), f"NaN in {field}"

    def test_value_in_tanh_range(self):
        assert (self.value >= -1).all() and (self.value <= 1).all()


# ===========================================================================
# TestAlphaZXHeteroModelSamplingRoundtrip
# ===========================================================================


class TestAlphaZXHeteroModelSamplingRoundtrip:
    """Test model → distribution → sample → log_prob pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_model()
        self.model.eval()

    def test_sample_from_model_output(self):
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            params, value = _run_model(self.model, batch)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(1)
        B = batch.batch.max().item() + 1
        assert samples.shape[0] == B
        assert samples.shape[1] == 1

    def test_log_prob_of_own_samples_is_finite(self):
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(1)
        lp = dist.log_prob(samples)
        assert torch.isfinite(lp).all()

    def test_log_prob_is_negative(self):
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(1)
        lp = dist.log_prob(samples)
        assert (lp <= 0).all()

    def test_entropy_is_non_negative(self):
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        dist = AlphaZXDistribution(params)
        entropy = dist.entropy()
        assert entropy.item() >= 0


# ===========================================================================
# TestAlphaZXHeteroModelGradients
# ===========================================================================


class TestAlphaZXHeteroModelGradients:
    """Test that gradients flow through the full heterogeneous model."""

    def test_value_gradient_flow(self):
        model = _make_model()
        batch = _make_batch(batch_size=1)
        params, value = _run_model(model, batch)
        loss = value.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.shared_hgt.parameters()
        )
        assert has_grad, "No gradients in shared HGT from value loss"

    def test_policy_gradient_flow(self):
        model = _make_model()
        batch = _make_batch(batch_size=3)
        params, value = _run_model(model, batch)
        loss = (params.mixture_dist_probs.sum()
                + params.node_dist_probs.sum()
                + params.phase_dist_probs.sum()
                + params.new_edge_dist_probs.sum()
                + params.transfer_edge_dist_probs.sum())
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.policy_hgt.parameters()
        )
        assert has_grad, "No gradients in policy HGT from policy loss"

    def test_log_prob_gradient_flow(self):
        model = _make_model()
        batch = _make_batch(batch_size=1)
        params, value = _run_model(model, batch)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(1)
        lp = dist.log_prob(samples)
        loss = -lp.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients from log_prob loss"

    def test_shared_receives_gradients_from_both_heads(self):
        """Shared HGT encoder should receive gradients from both policy and value."""
        model = _make_model()
        batch = _make_batch(batch_size=2)
        params, value = _run_model(model, batch)
        loss = value.sum() + params.mixture_dist_probs.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.shared_hgt.parameters()
        )
        assert has_grad, "Shared HGT should receive gradients from combined loss"


# ===========================================================================
# TestFlatToHeteroConversion
# ===========================================================================


class TestFlatToHeteroConversion:
    """Test the _flat_to_hetero and _hetero_to_flat conversion correctness."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_model()
        self.model.eval()

    def test_flat_to_hetero_preserves_node_count(self):
        """Total nodes across all types should equal original node count."""
        batch = _make_batch(batch_size=2)

        x_emb, _ = self.model.emb(batch.x, batch.edge_attr, batch.pe)
        x_proj = self.model.input_proj(x_emb)

        x_dict, edge_index_dict, local_indices = self.model._flat_to_hetero(
            x_proj, batch.edge_index, batch.node_type, batch.edge_type,
        )

        total_nodes = sum(v.shape[0] for v in x_dict.values())
        assert total_nodes == batch.x.size(0)

    def test_flat_to_hetero_preserves_edge_count(self):
        """Total edges across all triples should equal original edge count."""
        batch = _make_batch(batch_size=1)

        x_emb, _ = self.model.emb(batch.x, batch.edge_attr, batch.pe)
        x_proj = self.model.input_proj(x_emb)

        x_dict, edge_index_dict, local_indices = self.model._flat_to_hetero(
            x_proj, batch.edge_index, batch.node_type, batch.edge_type,
        )

        total_edges = sum(ei.shape[1] for ei in edge_index_dict.values())
        assert total_edges == batch.edge_index.shape[1]

    def test_hetero_roundtrip_preserves_embeddings(self):
        """flat→hetero→flat should recover the original embedding tensor."""
        batch = _make_batch(batch_size=2)

        x_emb, _ = self.model.emb(batch.x, batch.edge_attr, batch.pe)
        x_proj = self.model.input_proj(x_emb)

        x_dict, _, _ = self.model._flat_to_hetero(
            x_proj, batch.edge_index, batch.node_type, batch.edge_type,
        )

        x_recovered = self.model._hetero_to_flat(
            x_dict, batch.node_type, batch.x.size(0), batch.x.device,
        )

        assert torch.allclose(x_proj, x_recovered, atol=1e-6), \
            "flat→hetero→flat roundtrip should recover original embeddings"

    def test_local_indices_are_consecutive(self):
        """Local indices within each type should be 0, 1, 2, ..., n-1."""
        batch = _make_batch(batch_size=1)

        x_emb, _ = self.model.emb(batch.x, batch.edge_attr, batch.pe)
        x_proj = self.model.input_proj(x_emb)

        x_dict, _, local_indices = self.model._flat_to_hetero(
            x_proj, batch.edge_index, batch.node_type, batch.edge_type,
        )

        for i, name in enumerate(self.model._node_type_names):
            mask = (batch.node_type == i)
            if mask.sum() == 0:
                continue
            type_local = local_indices[mask]
            expected = torch.arange(mask.sum(), device=type_local.device)
            assert torch.equal(type_local, expected), \
                f"Local indices for type {name} are not consecutive"

    def test_edge_type_routing_correctness(self):
        """Edges should be routed to the correct (src_type, rel, dst_type) triple."""
        batch = _make_batch(batch_size=1)

        x_emb, _ = self.model.emb(batch.x, batch.edge_attr, batch.pe)
        x_proj = self.model.input_proj(x_emb)

        _, edge_index_dict, _ = self.model._flat_to_hetero(
            x_proj, batch.edge_index, batch.node_type, batch.edge_type,
        )

        for triple, ei in edge_index_dict.items():
            if ei.shape[1] == 0:
                continue

            src_type_name, rel, dst_type_name = triple
            src_type_idx = METADATA.node_type_abbrev_index_dict[src_type_name]
            dst_type_idx = METADATA.node_type_abbrev_index_dict[dst_type_name]

            et_idx = METADATA.edge_type_to_index_dict[triple]
            mask = (batch.edge_type == et_idx)
            global_src = batch.edge_index[0, mask]
            global_dst = batch.edge_index[1, mask]

            src_types = batch.node_type[global_src]
            assert (src_types == src_type_idx).all(), \
                f"Edge triple {triple}: source nodes have wrong type"

            dst_types = batch.node_type[global_dst]
            assert (dst_types == dst_type_idx).all(), \
                f"Edge triple {triple}: dest nodes have wrong type"


# ===========================================================================
# TestHparamsExtractionAndReconstruction
# ===========================================================================


class TestHparamsExtractionAndReconstruction:
    """Test hparams extraction and model reconstruction for the hetero model."""

    def test_extract_hparams_returns_correct_values(self):
        model = _make_model()
        hparams = _extract_model_hparams(model)

        assert hparams['model_type'] == 'heterogeneous'
        assert hparams['num_node_types'] == NUM_NODE_TYPES
        assert hparams['num_possible_phases'] == NUM_POSSIBLE_PHASES
        assert hparams['num_possible_new_edges'] == NUM_POSSIBLE_NEW_EDGES
        assert hparams['node_embedding_channels'] == NODE_EMB_CHANNELS
        assert hparams['num_edge_embeddings'] == NUM_EDGE_EMBEDDINGS
        assert hparams['edge_embedding_channels'] == EDGE_EMB_CHANNELS
        assert hparams['pe_in_channels'] == PE_DIM
        assert hparams['pe_out_channels'] == PE_DIM

    def test_reconstruct_model_from_hparams(self):
        model = _make_model()
        hparams = _extract_model_hparams(model)
        state_dict = model.state_dict()

        reconstructed = _build_model_from_hparams(hparams)
        assert isinstance(reconstructed, AlphaZXHeteroModel)
        reconstructed.load_state_dict(state_dict)

    def test_hparams_roundtrip_identical_parameters(self):
        model = _make_model()
        model.eval()
        hparams = _extract_model_hparams(model)
        state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        reconstructed = _build_model_from_hparams(hparams)
        reconstructed.load_state_dict(state_dict)
        reconstructed.eval()

        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), reconstructed.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Parameter {n1} differs after reconstruction"

    def test_hparams_roundtrip_identical_outputs(self):
        torch.manual_seed(42)
        model = _make_model()
        model.eval()

        hparams = _extract_model_hparams(model)
        state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        reconstructed = _build_model_from_hparams(hparams)
        reconstructed.load_state_dict(state_dict)
        reconstructed.eval()

        batch = _make_batch(batch_size=2, seed=99)
        with torch.no_grad():
            params1, value1 = _run_model(model, batch)
            params2, value2 = _run_model(reconstructed, batch)

        assert torch.allclose(value1, value2, atol=1e-6)
        assert torch.allclose(
            params1.mixture_dist_probs, params2.mixture_dist_probs, atol=1e-6,
        )

    def test_homogeneous_model_still_works_with_build(self):
        """_build_model_from_hparams still correctly builds homogeneous models."""
        from alphazx.models.homogeneous.alphazx_model import AlphaZXModel

        model = AlphaZXModel(
            NUM_NODE_TYPES, NUM_POSSIBLE_PHASES, NUM_POSSIBLE_NEW_EDGES,
            NODE_EMB_CHANNELS, NUM_EDGE_EMBEDDINGS, EDGE_EMB_CHANNELS,
            PE_DIM, PE_DIM,
        )
        hparams = _extract_model_hparams(model)
        assert hparams['model_type'] == 'homogeneous'

        reconstructed = _build_model_from_hparams(hparams)
        assert isinstance(reconstructed, AlphaZXModel)
        reconstructed.load_state_dict(model.state_dict())

    def test_hparams_roundtrip_non_default_layers(self):
        """Roundtrip with non-default HGT layer counts (regression test).

        When the model is trained with e.g. 3 shared layers instead of 2,
        the hparams must capture those counts so workers reconstruct the
        correct architecture.
        """
        model = _make_model(
            hgt_num_shared_layers=3,
            hgt_num_policy_layers=1,
            hgt_num_value_layers=3,
        )
        hparams = _extract_model_hparams(model)

        assert hparams['hgt_num_shared_layers'] == 3
        assert hparams['hgt_num_policy_layers'] == 1
        assert hparams['hgt_num_value_layers'] == 3

        reconstructed = _build_model_from_hparams(hparams)
        assert len(reconstructed.shared_hgt) == 3
        assert len(reconstructed.policy_hgt) == 1
        assert len(reconstructed.value_hgt) == 3

        # State dict should load cleanly (no unexpected/missing keys)
        reconstructed.load_state_dict(model.state_dict())


# ===========================================================================
# TestAlphaZXHeteroModelParameterManagement
# ===========================================================================


class TestAlphaZXHeteroModelParameterManagement:
    """Test parameter counts, structure, and submodules."""

    def test_model_has_parameters(self):
        model = _make_model()
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0

    def test_parameter_count_is_reasonable(self):
        """With default hyperparameters, model should have ~10M params."""
        model = _make_model()
        num_params = sum(p.numel() for p in model.parameters())
        assert 1_000_000 < num_params < 50_000_000, \
            f"Parameter count {num_params} is outside expected range"

    def test_all_parameters_are_float(self):
        model = _make_model()
        for name, p in model.named_parameters():
            assert p.is_floating_point(), f"Parameter {name} is not floating point"

    def test_has_expected_submodules(self):
        model = _make_model()
        expected = [
            'emb', 'input_proj', 'shared_hgt', 'policy_hgt', 'value_hgt',
            'rewrite_type_selector', 'node_selector', 'new_phase_selector',
            'new_edge_selector', 'transfer_edge_selector',
            'value_attention', 'value_ff',
        ]
        for attr in expected:
            assert hasattr(model, attr), f"Missing submodule: {attr}"

    def test_num_action_types_is_10(self):
        model = _make_model()
        assert model.NUM_ACTION_TYPES == 10

    def test_custom_hgt_layers(self):
        model = _make_model(
            hgt_num_shared_layers=3,
            hgt_num_policy_layers=1,
            hgt_num_value_layers=1,
        )
        assert len(model.shared_hgt) == 3
        assert len(model.policy_hgt) == 1
        assert len(model.value_hgt) == 1


# ===========================================================================
# TestAlphaZXHeteroModelConsistency
# ===========================================================================


class TestAlphaZXHeteroModelConsistency:
    """Test that output params are consistent with AlphaZXDistribution expectations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_model()
        self.model.eval()

    def test_all_param_fields_present(self):
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        expected_fields = {
            'graph_ids', 'mixture_dist_probs', 'node_dist_probs',
            'phase_dist_probs', 'new_edge_dist_probs', 'transfer_edge_dist_probs',
        }
        assert set(params._fields) == expected_fields

    def test_batch_dimension_consistent_across_params(self):
        batch = _make_batch(batch_size=3)
        with torch.no_grad():
            params, value = _run_model(self.model, batch)
        B = 3
        assert params.mixture_dist_probs.shape[0] == B
        assert params.node_dist_probs.shape[0] == B
        assert params.phase_dist_probs.shape[0] == B
        assert params.new_edge_dist_probs.shape[0] == B
        assert params.transfer_edge_dist_probs.shape[0] == B
        assert value.shape[0] == B

    def test_mixture_probs_dim_is_10(self):
        batch = _make_batch(batch_size=1)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        assert params.mixture_dist_probs.shape[1] == 10

    def test_node_probs_second_dim_is_10(self):
        batch = _make_batch(batch_size=1)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        assert params.node_dist_probs.shape[1] == 10


# ===========================================================================
# TestInterfaceCompatibility
# ===========================================================================


class TestInterfaceCompatibility:
    """Test that AlphaZXHeteroModel is a true drop-in for AlphaZXModel."""

    def test_homogeneous_ignores_edge_type_none(self):
        """Homogeneous model should accept edge_type=None without error."""
        from alphazx.models.homogeneous.alphazx_model import AlphaZXModel

        model = AlphaZXModel(
            NUM_NODE_TYPES, NUM_POSSIBLE_PHASES, NUM_POSSIBLE_NEW_EDGES,
            NODE_EMB_CHANNELS, NUM_EDGE_EMBEDDINGS, EDGE_EMB_CHANNELS,
            PE_DIM, PE_DIM,
        )
        model.eval()
        batch = _make_batch(batch_size=1)
        with torch.no_grad():
            params, value = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.node_type,
                batch.batch, batch.pe, batch.id,
                edge_type=None,
            )
        assert isinstance(params, AlphaZXDistributionParams)
        assert value.shape == (1, 1)

    def test_both_models_produce_same_output_format(self):
        """Both models should produce identical output structure and shapes."""
        from alphazx.models.homogeneous.alphazx_model import AlphaZXModel

        hetero_model = _make_model()
        homo_model = AlphaZXModel(
            NUM_NODE_TYPES, NUM_POSSIBLE_PHASES, NUM_POSSIBLE_NEW_EDGES,
            NODE_EMB_CHANNELS, NUM_EDGE_EMBEDDINGS, EDGE_EMB_CHANNELS,
            PE_DIM, PE_DIM,
        )
        hetero_model.eval()
        homo_model.eval()

        batch = _make_batch(batch_size=2, seed=77)
        with torch.no_grad():
            h_params, h_value = _run_model(hetero_model, batch)
            o_params, o_value = homo_model(
                batch.x, batch.edge_index, batch.edge_attr, batch.node_type,
                batch.batch, batch.pe, batch.id,
            )

        assert set(h_params._fields) == set(o_params._fields)
        assert h_params.mixture_dist_probs.shape == o_params.mixture_dist_probs.shape
        assert h_params.node_dist_probs.shape == o_params.node_dist_probs.shape
        assert h_params.phase_dist_probs.shape == o_params.phase_dist_probs.shape
        assert h_params.new_edge_dist_probs.shape == o_params.new_edge_dist_probs.shape
        assert h_params.transfer_edge_dist_probs.shape == o_params.transfer_edge_dist_probs.shape
        assert h_value.shape == o_value.shape


# ===========================================================================
# TestSyntheticDataValidity
# ===========================================================================


class TestSyntheticDataValidity:
    """Verify that the synthetic data generator produces valid data."""

    def test_edge_types_consistent_with_node_types(self):
        """Every edge's edge_type triple should match the src/dst node types."""
        data = _make_synthetic_data(num_nodes=20, seed=42)
        for i in range(data.edge_index.shape[1]):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            et = data.edge_type[i].item()

            # Find the triple for this edge type index
            triple = None
            for t, idx in METADATA.edge_type_to_index_dict.items():
                if idx == et:
                    triple = t
                    break
            assert triple is not None, f"Edge type index {et} not in METADATA"

            src_type_name, _, dst_type_name = triple
            expected_src = METADATA.node_type_abbrev_index_dict[src_type_name]
            expected_dst = METADATA.node_type_abbrev_index_dict[dst_type_name]
            assert data.node_type[src].item() == expected_src, \
                f"Edge {i}: src type mismatch"
            assert data.node_type[dst].item() == expected_dst, \
                f"Edge {i}: dst type mismatch"

    def test_edge_index_sorted_by_dst(self):
        """Edge index should be sorted by destination node."""
        data = _make_synthetic_data(seed=123)
        dst = data.edge_index[1]
        assert torch.equal(dst, dst.sort()[0]), "edge_index not sorted by dst"

    def test_batch_has_edge_type(self):
        batch = _make_batch(batch_size=3)
        assert hasattr(batch, 'edge_type')
        assert batch.edge_type.shape[0] == batch.edge_index.shape[1]
