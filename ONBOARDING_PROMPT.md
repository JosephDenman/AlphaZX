# AlphaZX Onboarding Prompt

This document provides complete context for understanding the AlphaZX project. After reading this, you should be able to navigate the codebase, understand all design decisions, identify known weaknesses, and contribute meaningfully to any part of the system.

## What AlphaZX Is

AlphaZX applies AlphaZero-style Monte Carlo Tree Search (MCTS) to the problem of simplifying quantum circuits represented as ZX-calculus diagrams. The optimization target is reducing the number of non-Clifford (T) gates in a circuit, which is the primary cost metric in fault-tolerant quantum computing. T-gate count reduction is an NP-hard problem in general, making it a natural fit for learned search.

The system generates random quantum circuits, converts them to ZX diagrams, and uses MCTS guided by a graph neural network to discover sequences of ZX-calculus rewrite rules that reduce T-gate count. Training data comes from self-play: the network plays against itself, and the resulting (state, policy, value) triples train the next iteration of the network.

## Architecture Overview

The system has five layers, each in its own package:

```
alphazx/
├── diagram/       # ZX diagram representation and pattern matching
├── game/          # Game environment (reward, termination, action decoding)
├── rewriting/     # ZX-calculus rewrite rules and incremental match updates
├── models/        # GNN policy/value network (GPS encoder + hierarchical heads)
├── distributions/ # Structured action distribution (hierarchical sampling)
└── mcts/          # MCTS search, self-play, training loop, evaluation
```

The entry point is `train_alphazx.py` at the project root.

---

## Layer 1: Diagram Representation (`alphazx/diagram/`)

### ZXDiagram (`zx_diagram.py`)

`ZXDiagram` extends `networkx.MultiGraph`. Nodes are integers. Each node has attributes stored in a parallel dict (`node_attrs`): `basis` (z, x, or b for boundary), `phase` (rational multiple of π, stored as a float), and boundary flag. Edges can be parallel (MultiGraph), which matters for the ZX-calculus. The class provides accessors like `z_nodes()`, `x_nodes()`, `b_nodes()`, `phases()`, `num_edges()`, and `copy()`.

Key function: `base_match_from_node(diagram, node_id)` creates the appropriate `SimpleMatchNode` subclass (BasisZMatch, BasisXMatch, or BoundaryMatch) for a given node.

Phase representation: phases are stored as floats representing multiples of π. A phase of 0.25 means π/4 (a T-gate). Non-Clifford gates are those where `phase % 0.5 != 0`.

### Match Nodes (`match.py`)

This file defines the type hierarchy for nodes in the match diagram. The base class `ZXMatchDiagramNode` has two branches:

**SimpleMatchNode** (1:1 with ZX diagram nodes):
- `BasisZMatch(node_id)` — represents a single Z-basis node
- `BasisXMatch(node_id)` — represents a single X-basis node
- `BoundaryMatch(node_id)` — represents a boundary node

**MatchNode** (represents a rewrite rule match pattern):
- `FRightZMatch(node)` / `FRightXMatch(node)` — single-node F-right (spider splitting)
- `FLeftZMatch(n1, n2)` / `FLeftXMatch(n1, n2)` — adjacent pair F-left (spider fusion)
- `BRightMatch(z, x)` — adjacent Z-X pair, both degree 3, phase 0 (bialgebra right)
- `BLeftMatch(z, x, m, n)` — 4-node cycle pattern (bialgebra left)
- `YRightZMatch(bottom, center, tl, tr)` / `YRightXMatch(...)` — Y-rule right patterns
- `YLeftZMatch(bottom, center, tl, tr)` / `YLeftXMatch(...)` — Y-rule left patterns

**SuperNode** (one per match type, used for type-level graph structure):
- e.g., `FRightZSuperNode`, `BLeftSuperNode`, etc.

**METADATA** is a module-level singleton that stores all type registries: `match_node_type_abbrevs` (list of abbreviation strings like 'frz', 'frx', 'flz', etc.), `node_type_abbrev_index_dict`, `edge_type_to_index_dict`, etc. These are used pervasively throughout the codebase.

**Important quirk**: `MatchNode.__init__` accepts `*match` with variable arguments and has special handling: if a single int is passed, it creates `{node: 0}`. If a tuple is passed, it creates `{node: i for i, node in enumerate(match)}`. The `__hash__` is based on `.name`, which is a string property. MatchNodes are used as dictionary keys and set members throughout.

**Known issue**: The `@staticmethod` and `@property` decorators are combined on abstract methods (`index`, `name`, `abbrev`, `meta_neighbors`). This is technically invalid in Python — `@staticmethod @property` doesn't work as intended. It functions because subclasses override these as class-level attributes, so the descriptor protocol never actually activates. It's ugly but not broken.

### ZXMatchDiagram (`zx_match_diagram.py`)

`ZXMatchDiagram` extends `networkx.DiGraph`. It's a hierarchical graph where:
- Simple nodes (BasisZ/X/Boundary matches) correspond 1:1 to ZX diagram nodes
- Match nodes represent detected rewrite rule patterns
- Super nodes represent match types (one per type)
- Edges encode both the original ZX diagram connectivity and type-hierarchy relationships

**Overridden methods**: `add_node()` and `add_edge()` are overridden to:
- Automatically compute and attach node/edge attributes (phase, type, edge_size)
- Add the node to its type-specific set (e.g., `self.frz_nodes.add(match)`)
- Create the corresponding super node if it doesn't exist
- Add reverse edges (the DiGraph is stored as bidirectional)

This override is why cloning requires special care — naively calling `add_nodes_from` would trigger all this logic again. See the `_clone_match_diagram` function in `game_state.py`.

**`to_zx_match_diagram(diagram)`**: The factory function that detects all match patterns in a ZX diagram and builds the full match diagram. This is O(V²) in the worst case due to neighbor-pair enumeration and is the most expensive operation in the pipeline when called from scratch.

**`to_pyg_data(with_reverse_mapping)`**: Converts the match diagram to a PyTorch Geometric `Data` object for GNN input. Returns `(Data, DataIndexToMatch)` where `DataIndexToMatch` maps integer indices back to MatchNode objects (needed for action decoding).

**`compute_f_right_params(action, data, data_index, match_diagram)`**: Decodes the phase, new_edges, and transfer_edges components of an F-right action from the action tuple and PyG data. This is non-trivial because transfer edges are encoded as a binary mask over the node's neighbors in the PyG representation.

### Constants (`constants.py`)

Defines edge type indices: `S_ETYPE_INDEX=0` (simple), `I_ETYPE_INDEX=1` (inclusion), `SS_ETYPE_INDEX=2` (simple_super), `SI_ETYPE_INDEX=3` (inclusion_super). `ETYPE_COUNT=4`.

### Diagram Generators (`diagram_generators.py`)

Two generators:
- `clifford_zx_diagram(qubits, depth, t_gates)`: Generates random ZX graphs using `pyzx.generate.cliffords`. These are graph-level diagrams, not derived from circuits.
- `cnot_had_phase_zx_diagram(qubits, depth, p_had, p_t)`: Generates real quantum circuits (CNOT + Hadamard + T gates) via `pyzx.generate.CNOT_HAD_PHASE_circuit`, then converts to ZX diagram. These have more realistic structure and are recommended for training.

Both use `pyzx_to_nx_multigraph()` from `pyzx_nx_conv.py` to convert from PyZX's internal format to our NetworkX-based `ZXDiagram`.

---

## Layer 2: Game Environment (`alphazx/game/`)

### zx_game.py

This file serves dual purposes: it defines the game logic used by both the legacy `ZXGame` environment and the newer MCTS-based `GameState`.

**Key functions used by MCTS**:
- `num_non_clifford_gates(diagram)` — counts nodes where `phase % 0.5 != 0`
- `is_simplified(diagram)` — returns True when T-gate count is 0
- `tuple_to_match(match_diagram, data, action, data_index)` — decodes an action tuple into a `(MatchNode, params)` pair for rewriting
- `calculate_reward(old_stats, new_stats)` — shaped reward function
- `remove_isolated_nodes`, `remove_self_loop_edges`, `remove_isolated_components` — cleanup after rewrites
- `update_match_diagram_for_removed_nodes` — keeps match diagram in sync after cleanup

**`DiagramStats`**: Snapshot of diagram metrics (node/edge counts, match counts per type, T-gate count). Used for reward computation.

**`calculate_reward`**: The shaped reward function. Components:
- T-gate reduction: `Δt × 10.0` (primary signal)
- Node reduction: `+0.1` per node removed, `-0.2` per node added (asymmetric penalty)
- Edge reduction: `+0.05` per edge removed, `-0.1` per edge added
- Match reduction: `+0.2` per complex match removed (B and Y types)
- Returns `(total_reward, RewardBreakdown)`

**`tuple_to_match`**: Decodes the action tuple format `(graph_id, action_type, node_index, phase, new_edges, transfer_edges...)`. The `action_type` field (0-9) maps to match types via `action_type + 1 = match_type_index`. F-right matches require additional parameters (phase, new_edges, transfer_edges); all others need only the match node.

**`ZXGame`**: Legacy Gymnasium-style environment. Still functional but not used in the MCTS training loop. Kept for potential PPO/A2C comparisons. Uses `step(action) → (obs, reward, done, info)` interface.

---

## Layer 3: Rewrite Rules (`alphazx/rewriting/`)

The ZX-calculus has a small set of graph rewrite rules that preserve the quantum circuit semantics. AlphaZX implements six rule families (each with Z and X variants where applicable):

### F-rules (Spider Fusion/Fission)

**F-left** (`f_left_rewrite`): Fuses two adjacent same-basis spiders into one. The resulting phase is the sum mod 2. This is the most common simplification move — it merges redundant nodes.

**F-right** (`f_right_rewrite`): Splits a single spider into two. This is parameterized: the agent chooses the new phases, the number of parallel edges between the new nodes, and which neighbors transfer to the right node. F-right is the only rule that increases diagram complexity, but it can enable subsequent simplifications.

### B-rules (Bialgebra)

**B-right** (`b_right_rewrite`): Takes an adjacent Z-X pair (both degree 3, phase 0) and replaces them with 4 nodes arranged in a bialgebra pattern. Increases local complexity but can enable non-local simplifications.

**B-left** (`b_left_rewrite`): Takes a 4-node Z-X-M-N cycle (alternating bases, all degree 3, all phase 0) and collapses it to 2 nodes. This is the simplifying direction of bialgebra.

### Y-rules (Y-triangle)

**Y-right** and **Y-left** (`y_right_rewrite`, `y_left_rewrite`): Transform Y-shaped subgraphs (a center node with 3 neighbors of opposite basis, all degree 2, with specific phase constraints). These rules handle the interaction between Clifford and non-Clifford phases.

### UpdateSet (`update_set.py`)

Named tuple `(removed_nodes, added_nodes, original_match)` returned by every rewrite. Used by the incremental match update system.

### Efficient Rewrite (`efficient_rewrite.py`)

This is a critical optimization. After applying a rewrite, the match diagram needs to be updated to reflect new/removed pattern matches. The naive approach (recompute all matches from scratch) is O(V²). The efficient approach:

1. Compute the k-hop neighborhood of affected nodes BEFORE the rewrite
2. Record all existing matches in that neighborhood
3. Apply the rewrite
4. Extend the neighborhood to include newly created nodes
5. Remove old matches from the match diagram
6. Detect new matches in the post-rewrite neighborhood
7. Add new matches to the match diagram

Default `neighborhood_hops=4`. This makes updates O(local) instead of O(global).

Each match type has a corresponding `detect_*_in_neighborhood()` function that searches for that pattern within a given set of nodes. These are essentially local pattern matching functions.

**`verify_match_diagram_consistency`**: Debug function that compares the incrementally-maintained match diagram against a fresh recomputation. Returns `(is_consistent, error_list)`. Used in tests.

---

## Layer 4: Neural Network (`alphazx/models/`)

### Architecture

The network follows the AlphaZero pattern: a shared representation trunk feeds separate policy and value heads.

```
Input (PyG Data) → FeatureEmbeddingLayer → GPS Encoder → RepresentationNetwork
                                                              ↓
                                    ┌──────────────────────────┴──────────────────────────┐
                                    ↓                                                      ↓
                              PolicyNetwork                                          ValueNetwork
                              (5 sub-heads)                                    (GPS + AttentionalAggregation
                                    ↓                                           + MLP → scalar)
                    AlphaZXDistributionParams
```

### RepresentationNetwork (`representation_network.py`)

Composes `FeatureEmbeddingLayer` + `GPS`. Takes raw node types, edge types, and positional encodings. Outputs per-node embeddings and edge attributes.

### GPS Encoder (`gps.py`, `gps_layer.py`)

GPS = Graph Position & Structural encoder. Stacks multiple `GPSConv` layers with `ResGatedGraphConv` for message passing and multi-head attention. Uses BatchNorm (not LayerNorm). Includes residual connections.

`FeatureEmbeddingLayer` embeds discrete node/edge types into continuous vectors and projects positional encodings.

### Positional Encoding (`positional_encoding.py`)

Uses random-walk structural encoding (RWSE). `pre_process_single(data, pe_dim)` computes PE by taking powers of the random walk matrix (D⁻¹A). This is computed fresh for each graph since the graph topology changes at every step.

**Optimization note**: The self-play worker caches preprocessed data on the GameState object (`_cached_preprocessed_data`) to avoid recomputing PE twice (once during MCTS evaluation, once when storing to replay buffer).

### PolicyNetwork (`policy_network.py`)

Orchestrates 5 sub-heads that decompose the action into components:

1. **RewriteTypeSelector** (`rewrite_type_selector.py`): Outputs P(action_type) as a [B, 10] distribution over 10 rewrite types. Filters for super nodes (indices 12-21), ensures action types only get positive probability if valid match nodes exist.

2. **NodeSelector** (`node_selector.py`): Outputs P(node | action_type) as a [B, T, N] tensor. Uses masked softmax to zero out nodes that don't match the selected action type.

3. **NewPhaseSelector** (`new_phase_selector.py`): Outputs P(phase | node) as [B, N, num_phases]. Only active for F-right matches; other match types get a delta distribution at phase 0.

4. **NewEdgeSelector** (`new_edge_selector.py`): Outputs P(new_edges | node) as [B, N, num_new_edges]. Only active for F-right matches.

5. **TransferEdgeSelector** (`transfer_edge_selector.py`): Outputs P(transfer_edges | node) as [B, N, max_degree] independent Bernoullis. Uses `SetTransformerAggregation` (Graph Multiset Transformer) for neighbor-aware edge selection.

### ValueNetwork (`value_network.py`)

GPS layer → `AttentionalAggregation` (learned attention-weighted global pooling) → feed-forward MLP → scalar. Outputs a single value estimate per graph.

### AlphaZXModel

The top-level model class (in `alphazx/models/homogeneous/`). Composes `RepresentationNetwork` + `PredictionNetwork`. The `forward()` method takes PyG data and returns `(AlphaZXDistributionParams, value)`.

---

## Layer 5: Distributions (`alphazx/distributions/`)

### AlphaZXDistribution (`alpha_zx_dist.py`)

The structured action distribution. An action is a tuple: `(graph_id, action_type, node, phase, new_edges, transfer_edges...)`.

The joint probability factorizes as:
```
P(action) = P(type) × P(node|type) × P(phase|node) × P(new_edges|node) × P(transfer_edges|node)
```

Key methods:
- `sample(k)` → `[B, K, L]` tensor of k action samples per batch element
- `log_prob(actions)` → `[B, K]` log probabilities (sum of component log probs)
- `entropy()` → upper bound on entropy (sum of component entropies weighted by mixture)

Transfer edges use `MultivariateBernoulli` (independent Bernoullis over the neighbor set).

### Helper Distributions

- `MultivariateBernoulli` (`bernoulli_mixture.py`): Wraps `Independent(Bernoulli, 1)` for edge selection.
- `MultivariateBernoulliMixture`: `MixtureSameFamily(Categorical, Independent(Bernoulli, 1))`. Not currently used in the main training path.
- `CategoricalDistribution` (`categorical.py`): Thin wrapper over PyTorch's Categorical.

---

## Layer 6: MCTS and Training (`alphazx/mcts/`)

### MCTSConfig (`config.py`)

Dataclass with all hyperparameters. Key defaults:
- `num_simulations = 100` per search
- `c_puct = 1.5` (PUCT exploration constant)
- `pw_alpha = 0.5`, `pw_c = 1.0` (progressive widening)
- `temperature = 1.0` for training, `0.1` for evaluation
- `gamma = 1.0` (undiscounted)
- `pe_dim = 20`
- `max_episode_length = 100`
- `max_t_gate_increase = 10` (early termination threshold)
- `num_qubits = 5`, `depth = 5`
- `circuit_type = 'cnot_had_phase'`

### MCTSNode (`node.py`)

Tree node with `__slots__` for memory efficiency. Stores state, parent, children dict, visit count, total value, prior, reward.

**PUCT formula**: `Q(s,a) + c_puct × P(s,a) × √N(parent) / (1 + N(s,a))`

**Progressive widening**: `N ≥ pw_c × |children|^(1/pw_alpha)`. With pw_alpha=0.5, pw_c=1.0, the threshold for the 11th child is exactly 100 (the simulation budget), so children cap at 10.

**Backpropagation**: Value propagates from leaf to root. At each step, `node.total_value += value`, `node.visit_count += 1`, then `value = value × gamma + node.reward` (adding the immediate transition reward).

### MCTS Search (`search.py`)

Implements sampled MCTS with progressive widening. Key difference from standard AlphaZero: actions are SAMPLED from the policy distribution, not enumerated.

Search loop:
1. **Select**: Walk down tree using PUCT until reaching a leaf or widening point
2. **Expand**: At widening points, sample a new action, clone state, apply action, create child. At leaves, evaluate with neural network.
3. **Backpropagate**: Propagate value up to root

**Dirichlet noise**: Applied to all root children. Because children are added progressively, noise is re-generated and re-applied whenever the child count changes. Original priors are stored in `_original_prior` to prevent compounding.

**`_compute_policy`**: Converts root visit counts to probabilities with temperature: `π(a) = N(a)^(1/τ) / Σ N(a')^(1/τ)`. Uses log-space softmax for numerical stability.

### Evaluate (`evaluate.py`)

Bridge between MCTS and the neural network.

**`evaluate_state(model, state, pe_dim, device)`**: Preprocesses the PyG data (computes positional encoding, ensures float32), runs the model, returns `(AlphaZXDistribution, value_scalar)`. Caches the preprocessed data on the state object for reuse.

**`compute_action_prior(distribution, action)`**: Computes P(action) by summing component log probs. Used for PUCT priors on progressively-widened children.

### GameState (`game_state.py`)

Lightweight, copyable snapshot of the game state for MCTS tree search. Wraps `ZXDiagram` + `ZXMatchDiagram` + lazily-computed PyG Data.

**`clone()`**: Creates an independent copy for tree branching. Uses `_clone_match_diagram()` which copies the nx.DiGraph internals directly via `object.__new__()` (bypasses `__init__`) and `nx.DiGraph.add_nodes_from()` (bypasses overridden `add_node`). This was a critical optimization — the original implementation called `to_zx_match_diagram()` from scratch, which was the dominant MCTS bottleneck (5-15s per search of 100 simulations). The fast clone reduced this to 0.4-0.8s.

**`apply_action(action)`**: Mutates state in place. Decodes action, applies rewrite via `efficient_rewrite`, runs cleanup (isolated nodes, self-loops, disconnected components), computes reward, checks termination. Invalidates all caches.

### Self-Play (`self_play.py`)

**`SelfPlayWorker`**: Plays a single episode. At each step: run MCTS → get policy → record (state, policy) → sample action → apply → continue. After episode ends, computes discounted returns from step rewards and fills value targets retroactively.

Value target computation: `v_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...`, normalized by `max(initial_t_gates, 1) × 12.0` and clamped to [-1, 1]. If the episode was early-terminated due to T-gate increase, a terminal penalty of `-max_increase × 10.0` is appended to step rewards.

**`SelfPlayManager`**: Orchestrates multiple sequential episodes and feeds examples into the replay buffer. Currently single-threaded.

### Replay Buffer (`replay_buffer.py`)

Circular buffer with fixed capacity (default 100k). Stores `TrainingExample(state_data, mcts_policy, value_target, game_id)`. Uniform random sampling.

**`collate_training_batch`**: Converts a list of examples into a PyG Batch. Policy dicts stay as-is (variable action sets per example).

### Trainer (`trainer.py`)

AlphaZero training loop. Each iteration:
1. Self-play phase: generate games, store examples
2. Training phase: sample minibatches, compute loss, update

**Policy loss**: Cross-entropy between MCTS visit distribution and model predictions: `-Σ_a π(a) × log p(a)`. The model's `log_prob` is decomposed into component log probs (type, node, phase, new_edges, transfer_edges).

**Value loss**: MSE between predicted value and discounted return target.

**Total loss**: `policy_loss + c_value × value_loss`

Optimizer: AdamW. LR schedule: cosine, step, or constant. Gradient clipping at `max_grad_norm=1.0`.

### Evaluator (`evaluator.py`)

Evaluates the trained model against PyZX's `full_reduce` baseline. Runs low-temperature MCTS (near-greedy) on test circuits. Reports wins/ties/losses vs PyZX, average T-gate reduction, and simplification rate.

**Known issue**: The PyZX baseline generates a fresh circuit with the same parameters rather than converting the exact same circuit. This makes the comparison approximate, not exact.

---

## Data Flow

### Training Loop
```
generate_circuit() → ZXDiagram → GameState.from_diagram()
    → [for each step in episode]:
        MCTS.search(state)
            → [for each simulation]:
                select → clone + apply_action → evaluate_state(model) → backpropagate
            → visit_count_policy
        record (preprocessed_state, policy, -)
        sample action from policy → state.apply_action(action)
    → compute discounted returns → fill value targets
    → add examples to ReplayBuffer
    → [for each training step]:
        sample minibatch from buffer
        model.forward(batch) → (distribution_params, values)
        policy_loss = cross_entropy(mcts_policy, model_log_probs)
        value_loss = MSE(predicted_value, target_value)
        optimizer.step()
```

### Action Encoding
```
Action tuple: (graph_id, action_type, node_index, phase, new_edges, transfer_edge_0, transfer_edge_1, ...)
                  ↓           ↓            ↓         ↓         ↓              ↓
              batch idx    0-9 maps    PyG node    phase     edge count    binary mask
                          to match     index in   bucket     bucket       over neighbors
                          type via     the Data    index      index
                          +1 offset    object
```

### Graph Representation Pipeline
```
ZXDiagram (nx.MultiGraph, integer nodes)
    → ZXMatchDiagram (nx.DiGraph, MatchNode/SuperNode nodes, bidirectional edges)
        → PyG Data (node features: type+phase, edge features: type+size, PE)
            → Model input (after FeatureEmbedding + positional encoding)
```

---

## Known Shortcomings and Trade-offs

### Architectural

1. **Single-threaded self-play**: Games are played sequentially. The main optimization opportunity is batching neural network evaluations across multiple concurrent MCTS searches, but this requires virtual loss support and async evaluation. Deferred to a future phase.

2. **Progressive widening caps children at 10**: With `pw_alpha=0.5` and `pw_c=1.0`, the widening threshold for the 11th child exactly equals the 100-simulation budget. This means the search explores at most 10 distinct actions from each node. Whether this is sufficient depends on the branching factor of useful rewrites — it may need tuning.

3. **Evaluator baseline is approximate**: PyZX comparison generates fresh random circuits with the same parameters rather than converting the exact test circuit to PyZX format. A fairer comparison would apply `full_reduce` to the same circuit.

4. **Match node hashing**: MatchNodes hash by `.name` (a string property), not by object identity. This is correct for deduplication but has subtle implications — two MatchNode objects with the same constituent nodes are considered equal even if they were created independently.

5. **`@staticmethod @property` on abstract methods**: These stacked decorators in `match.py` don't work as intended in Python. They function only because subclasses override them as plain class attributes. Should be refactored to `@classmethod` or plain class variables.

6. **ZXGame (legacy) vs GameState (MCTS)**: Two parallel code paths for playing the game. `ZXGame` is a Gymnasium-style environment; `GameState` is the MCTS-native representation. They share utility functions from `zx_game.py` but duplicate some logic. The legacy path should eventually be removed or unified.

### Performance

7. **PE computation is O(V³)**: Random-walk positional encoding requires matrix power computation, which is cubic in the number of nodes. For large diagrams this dominates preprocessing time. The caching optimization (computed once during MCTS eval, reused for replay buffer storage) helps but doesn't reduce the fundamental cost.

8. **`to_pyg_data` called multiple times**: The match diagram → PyG conversion happens every time `GameState.ensure_data()` is called after a cache invalidation. Since `apply_action` invalidates the cache, every MCTS simulation that reaches a new state triggers this conversion.

9. **Match diagram clone is shallow**: `_clone_match_diagram` does shallow copies of node/edge attribute dicts. This is safe because MatchNode objects are immutable and attribute values are primitives, but it's a correctness assumption that could break if someone adds mutable attributes.

### Correctness

10. **F-right action space is massive**: The agent can choose phase (discretized), number of new edges, AND which neighbors to transfer. This makes F-right the hardest action to learn correctly. The current discretization into buckets may be too coarse or too fine.

11. **Transfer edge encoding**: Transfer edges are encoded as a binary mask over the node's neighbors in the PyG representation. The mapping between PyG neighbor indices and actual ZX diagram neighbors goes through `compute_f_right_params`, which is complex and has been a source of bugs.

12. **Value target normalization**: Discounted returns are normalized by `max(initial_t_gates, 1) × 12.0`. The 12.0 factor includes ~20% headroom over the primary 10x T-gate reward multiplier to account for secondary rewards. This is a heuristic that may need tuning.

13. **Early termination penalty scale**: When T-gates increase beyond `max_t_gate_increase`, a terminal penalty of `-max_increase × 10.0` is appended. This matches the T-gate reward scale but interacts with the value normalization in ways that haven't been thoroughly analyzed.

### Testing

14. **Test coverage gaps**: No unit tests for the model forward pass, the distribution sampling/likelihood, the training loop, or the evaluator. Tests focus on diagram manipulation, match detection, and rewriting. The MCTS integration is tested only through `test_game_state_clone.py`.

15. **Random seed control**: Tests use random circuit generation without fixed seeds. This makes some tests non-deterministic. The `match_patterns.py` test helper generates reference patterns but actual test circuits vary between runs.

---

## Key Design Decisions and Rationale

### Why Sampled MCTS (not standard AlphaZero MCTS)?

Standard AlphaZero enumerates all legal actions and assigns a prior to each. In ZX-calculus, the action space is structured and enormous (especially F-right with its phase/edge/transfer parameters). Enumeration is infeasible. Instead, we sample actions from the policy distribution and use progressive widening to gradually expand the search tree. This trades completeness for tractability.

### Why discounted returns (not uniform episode outcome)?

Classic AlphaZero uses the game outcome (win/loss/draw) as the value target for all positions in the episode. Our game has a shaped reward signal at every step, so we use discounted cumulative future reward instead. This gives more informative gradients: steps near T-gate reductions get high values, steps early in degenerate episodes get less negative signal.

### Why `_clone_match_diagram` bypasses `__init__`?

`ZXMatchDiagram.__init__` takes a `ZXDiagram` parameter and recomputes all matches from scratch. The overridden `add_node`/`add_edge` methods add computed attributes and maintain type-specific sets. Cloning needs to copy the existing graph structure without triggering any of this logic. Using `object.__new__` + direct `nx.DiGraph` base class methods is the only way to achieve this.

### Why batch dimension is always 1 during MCTS?

Each MCTS simulation evaluates a single game state. Batching across simulations would require virtual loss (to prevent multiple simulations from exploring the same path) and async evaluation queues. This is planned but not yet implemented. The current single-threaded approach processes ~100 simulations in 2-6 seconds (dominated by neural network evaluation).

### Why GPS (not GAT/GCN/GIN)?

GPS (Graph Position & Structure) combines local message passing (ResGatedGraphConv) with global attention, giving the network both local structural awareness and long-range information flow. The positional encoding provides structural identity that pure message-passing GNNs lack.

---

## Running the Project

### Installation
```bash
cd AlphaZX
pip install -e ".[dev]"
```

### Training
```bash
python train_alphazx.py \
    --num-qubits 5 --depth 5 \
    --num-iterations 100 \
    --self-play-games 50 \
    --training-steps 200 \
    --batch-size 32 \
    --num-simulations 100 \
    --log-level INFO
```

Add `--log-level DEBUG` for per-step action logging including rewrite types and T-gate changes.

### Tests
```bash
pytest tests/ -v
```

### Key CLI Arguments
- `--circuit-type`: `cnot_had_phase` (recommended) or `clifford`
- `--max-t-gate-increase`: Early termination threshold (default 10)
- `--min-initial-t-gates`: Skip circuits with too few T-gates (default 2)
- `--resume PATH`: Resume training from checkpoint
- `--no-pyzx-comparison`: Skip PyZX baseline evaluation
- `--eval-interval N`: Evaluate every N iterations

---

## File Index

### Core Source (`alphazx/`)

| File | Purpose |
|------|---------|
| `diagram/constants.py` | Edge type constants (4 types) |
| `diagram/match.py` | MatchNode type hierarchy, METADATA registry |
| `diagram/zx_diagram.py` | ZXDiagram (nx.MultiGraph wrapper) |
| `diagram/zx_match_diagram.py` | ZXMatchDiagram, to_zx_match_diagram, to_pyg_data |
| `diagram/diagram_generators.py` | Circuit generation (clifford, cnot_had_phase) |
| `diagram/pyzx_nx_conv.py` | PyZX → NetworkX conversion |
| `diagram/nx_drawing.py` | Visualization utilities |
| `game/zx_game.py` | Reward function, action decoding, cleanup, legacy ZXGame |
| `rewriting/efficient_rewrite.py` | Incremental match updates, neighborhood detection |
| `rewriting/f_rule_rewriter.py` | F-left (fusion) and F-right (fission) rewrites |
| `rewriting/b_rule_rewriter.py` | B-left and B-right (bialgebra) rewrites |
| `rewriting/y_rule_rewriter.py` | Y-left and Y-right rewrites |
| `rewriting/rewrite_actions.py` | Rewrite rule definitions (parallel to above) |
| `rewriting/update_set.py` | UpdateSet named tuple |
| `rewriting/utils.py` | `rewrite()` dispatcher |
| `models/homogeneous/gps.py` | GPS encoder + FeatureEmbeddingLayer |
| `models/homogeneous/gps_layer.py` | GPSConv layer configuration |
| `models/homogeneous/representation_network.py` | Shared trunk (embedding + GPS) |
| `models/homogeneous/prediction_network.py` | PolicyNetwork + ValueNetwork composition |
| `models/homogeneous/policy_network.py` | 5-head policy network |
| `models/homogeneous/value_network.py` | AttentionalAggregation + MLP value head |
| `models/homogeneous/rewrite_type_selector.py` | P(action_type) head |
| `models/homogeneous/node_selector.py` | P(node\|type) head |
| `models/homogeneous/new_phase_selector.py` | P(phase\|node) head |
| `models/homogeneous/new_edge_selector.py` | P(new_edges\|node) head |
| `models/homogeneous/transfer_edge_selector.py` | P(transfer_edges\|node) head |
| `models/positional_encoding.py` | Random-walk PE computation |
| `distributions/alpha_zx_dist.py` | AlphaZXDistribution (hierarchical sampling) |
| `distributions/bernoulli_mixture.py` | MultivariateBernoulli for edge selection |
| `distributions/categorical.py` | Categorical distribution wrapper |
| `distributions/utils.py` | Random distribution generators (testing) |
| `mcts/config.py` | MCTSConfig dataclass |
| `mcts/node.py` | MCTSNode (PUCT, widening, backprop) |
| `mcts/search.py` | MCTS search loop |
| `mcts/evaluate.py` | Neural network evaluation bridge |
| `mcts/game_state.py` | GameState, _clone_match_diagram |
| `mcts/self_play.py` | SelfPlayWorker, SelfPlayManager |
| `mcts/replay_buffer.py` | ReplayBuffer, TrainingExample |
| `mcts/trainer.py` | Trainer (AlphaZero training loop) |
| `mcts/evaluator.py` | Evaluator (vs PyZX baseline) |

### Tests (`tests/`)

| File | Covers |
|------|--------|
| `test_efficient_rewrite.py` | Incremental match updates vs full recomputation |
| `test_game_state_clone.py` | GameState.clone() and _clone_match_diagram |
| `test_f_left_rewrite.py` | F-left rewrite correctness |
| `test_f_right_rewrite.py` | F-right rewrite correctness |
| `test_b_left_rewrite.py` | B-left rewrite correctness |
| `test_b_right_rewrite.py` | B-right rewrite correctness |
| `test_y_left_rewrite.py` | Y-left rewrite correctness |
| `test_y_right_rewrite.py` | Y-right rewrite correctness |
| `test_zx_diagram.py` | ZXDiagram operations |
| `test_zx_match_diagram.py` | Match diagram construction and queries |
| `test_match.py` | MatchNode creation and equality |
| `match_patterns.py` | Reference match patterns for testing |

### Root

| File | Purpose |
|------|---------|
| `train_alphazx.py` | CLI entry point |
| `pyproject.toml` | Package config (hatchling backend, Python ≥3.10) |
| `LICENSE` | MIT |
| `.gitignore` | Standard Python patterns |

---

## Glossary

- **T-gate**: The π/8 gate; the only non-Clifford gate in the standard Clifford+T gate set. T-gate count is THE optimization target.
- **Clifford gates**: Gates that can be efficiently simulated classically (Hadamard, CNOT, S, etc.). Reducing to only Clifford gates means the circuit is trivially simulable.
- **ZX-calculus**: A graphical language for quantum computation where circuits are represented as open graphs with Z-spiders (green nodes) and X-spiders (red nodes) connected by edges.
- **Spider**: A node in a ZX diagram with a basis (Z or X) and a phase.
- **Spider fusion (F-left)**: Merging two adjacent same-basis spiders, summing their phases.
- **Spider fission (F-right)**: Splitting one spider into two with specified phases and edge routing.
- **Bialgebra rule**: Interaction rule between Z and X spiders that rearranges connectivity.
- **Match diagram**: The hierarchical graph that indexes all currently available rewrite rule matches in the ZX diagram. This is the key data structure that enables efficient action enumeration.
- **Progressive widening**: MCTS technique where children are added to a node gradually as it accumulates visits, rather than enumerating all possible children upfront.
- **PUCT**: Polynomial Upper Confidence Trees — the selection formula used in AlphaZero MCTS that balances exploitation (Q-value) with exploration (prior × visit-count bonus).
- **PE/RWSE**: Positional Encoding / Random Walk Structural Encoding — a technique that gives each node a feature vector capturing its structural position in the graph.
