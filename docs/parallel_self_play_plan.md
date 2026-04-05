# Implementation Plan: Multi-Process Self-Play

## Problem

Self-play is the training bottleneck. Each iteration generates `num_self_play_games` (default 100) games **sequentially** in `SelfPlayManager.generate_games()`. Each game runs MCTS with 100 simulations per move, taking 1-3 seconds per move × ~10 moves = 10-30 seconds per game. An iteration of 100 games therefore takes 15-50 minutes on CPU — far too slow for practical training.

## Solution: Multi-Process Self-Play via `ProcessPoolExecutor`

Spawn N worker **processes** (not threads — GIL prevents true parallelism for CPU-bound PyTorch inference). Each worker gets a frozen copy of the model's `state_dict`, constructs its own `SelfPlayWorker`, plays games independently, and returns `EpisodeResult` objects to the main process for replay buffer insertion.

### Why `ProcessPoolExecutor` over raw `multiprocessing`

`ProcessPoolExecutor` (from `concurrent.futures`) provides a cleaner API with automatic result collection, exception propagation, and context manager cleanup. It uses `multiprocessing` under the hood but abstracts away the boilerplate of explicit queues, join/terminate logic, and process lifecycle management.

## Architecture

```
Main Process (Trainer)
│
├── Owns: model, optimizer, replay buffer, TensorBoard logger
├── Before self-play: model.eval(), serialize state_dict
│
├── ProcessPoolExecutor(max_workers=N)
│   ├── Worker 0: _worker_play_games(state_dict, config, game_indices, seed_0)
│   │   └── Creates local model, loads state_dict, plays K games → [EpisodeResult, ...]
│   ├── Worker 1: _worker_play_games(state_dict, config, game_indices, seed_1)
│   │   └── Creates local model, loads state_dict, plays K games → [EpisodeResult, ...]
│   └── ...
│
├── Collects all EpisodeResult lists
├── Inserts examples into replay buffer (sequential — buffer is not thread-safe)
└── Proceeds to training phase
```

## Detailed Design

### 1. New module: `alphazx/mcts/parallel_self_play.py`

This module contains:

- **`_worker_play_games()`** — A module-level function (must be picklable) that each worker process executes. It:
  1. Reconstructs the model from `state_dict` + architecture hyperparams.
  2. Seeds its own RNG (torch, random, numpy) with a per-worker seed derived from a base seed + worker index. This ensures different workers generate different random circuits and make different MCTS sampling decisions.
  3. Creates a `SelfPlayWorker(model, config, device='cpu')`.
  4. Plays its assigned games in a local serial loop.
  5. Returns a `list[EpisodeResult]`.

- **`ParallelSelfPlayManager`** — Drop-in replacement for `SelfPlayManager` with the same `generate_games(num_games, start_diagrams=None)` interface. Internally:
  1. Serializes `model.state_dict()` once.
  2. Partitions `num_games` across N workers (roughly equal, remainder goes to last worker).
  3. Submits N tasks to the `ProcessPoolExecutor`.
  4. Collects results and inserts into the replay buffer (sequentially).
  5. Updates lifetime statistics (total_games, total_examples, etc.).

### 2. Worker function signature

```python
def _worker_play_games(
    model_state_dict: dict,
    model_hparams: dict,       # Architecture hyperparams to reconstruct model
    mcts_config: MCTSConfig,
    num_games: int,
    worker_seed: int,
    start_diagrams: list | None = None,
) -> list[EpisodeResult]:
```

**Why pass `model_hparams` separately?** The model class (`AlphaZXModel`) requires constructor arguments (num_node_types, num_possible_phases, etc.) that aren't stored in the state_dict. We pass these as a simple dict so the worker can reconstruct the model without importing the Trainer.

### 3. Model serialization strategy

**Option A (chosen): Pass `state_dict` as a Python dict of CPU tensors.**

- `state_dict = {k: v.cpu() for k, v in model.state_dict().items()}`
- Workers reconstruct the model and call `model.load_state_dict(state_dict)`.
- This is clean, well-supported, and the serialization overhead is small (model is ~2-5 MB).
- ProcessPoolExecutor pickles the arguments when dispatching to workers, so the state_dict is copied once per worker. For N=4-8 workers and a 5 MB model, this is 20-40 MB — negligible.

**Why not shared memory?** `torch.multiprocessing` supports shared-memory tensors via `model.share_memory_()`, but this requires `fork` start method and adds complexity around read-only access guarantees. Since we only copy the state_dict once per iteration (not per game), the overhead is trivial and the simpler approach wins.

### 4. Start method: `fork` vs `spawn`

**Use `fork` (default on Linux).** Rationale:

- We're CPU-only (MPS ruled out), so there are no CUDA fork-safety concerns.
- `fork` is faster: child processes inherit the parent's memory space via copy-on-write, so there's no need to re-import all modules.
- `spawn` would require re-importing the entire alphazx package in each worker, adding ~2-5 seconds of startup overhead per worker.

**Safety note:** We must ensure no CUDA contexts are initialized in the main process before forking. Since training is CPU-only, this is already the case. Add an assertion in `ParallelSelfPlayManager.__init__()`:
```python
assert not torch.cuda.is_initialized(), "CUDA must not be initialized before forking workers"
```

### 5. Seeding strategy

Each worker must produce different random circuits and MCTS samples. Strategy:

```python
base_seed = int(time.time() * 1000) % (2**31)
for worker_idx in range(num_workers):
    worker_seed = base_seed + worker_idx * 1000
```

Inside `_worker_play_games`:
```python
torch.manual_seed(worker_seed)
random.seed(worker_seed)
np.random.seed(worker_seed % (2**32))
```

### 6. Curriculum support

The current `_generate_curriculum_games()` in Trainer assigns each game a `(num_qubits, depth)` pair from the curriculum scheduler, then calls `generate_games(1)` with a temporarily overridden config.

For parallel self-play, we pre-compute the full list of `(nq, depth)` pairs, partition them across workers, and pass each worker its own list of config overrides:

```python
def _worker_play_games(
    ...,
    difficulty_overrides: list[tuple[int, int]] | None = None,
) -> list[EpisodeResult]:
    for i in range(num_games):
        if difficulty_overrides and i < len(difficulty_overrides):
            config.num_qubits, config.depth = difficulty_overrides[i]
        result = worker.play_episode(start_diagram)
        ...
```

### 7. Logging

Worker processes should not write to TensorBoard or the main process's logger handlers directly. Strategy:

- Set workers' logging level to WARNING to suppress per-game INFO messages (these would interleave on stdout and be unreadable).
- Each worker collects its own timing stats and returns them in the `EpisodeResult`.
- The main process logs aggregated per-game summaries after collecting all results, exactly as `SelfPlayManager.generate_games()` does today.

### 8. Changes to existing code

#### `self_play.py`

No changes needed. `SelfPlayWorker` and `EpisodeResult` are used as-is by the worker function. The existing serial `SelfPlayManager` remains for testing and single-process fallback.

#### `trainer.py`

Minimal changes:

```python
class Trainer:
    def __init__(self, ..., num_self_play_workers: int = 1):
        ...
        if num_self_play_workers > 1:
            from alphazx.mcts.parallel_self_play import ParallelSelfPlayManager
            self.self_play_manager = ParallelSelfPlayManager(
                model=model,
                config=mcts_config,
                replay_buffer=replay_buffer,
                device=device,
                num_workers=num_self_play_workers,
            )
        else:
            self.self_play_manager = SelfPlayManager(...)
```

The rest of `_run_iteration()` doesn't change — it already calls `self.self_play_manager.generate_games(num_games)` and processes the returned `list[EpisodeResult]`.

#### `config.py` / `TrainerConfig`

Add one field:

```python
@dataclass
class TrainerConfig:
    ...
    num_self_play_workers: int = 1
    """Number of parallel worker processes for self-play.
    1 = serial (no multiprocessing overhead). >1 spawns worker processes.
    Recommended: cpu_count - 1 to leave one core for the main process."""
```

### 9. Picklability audit

All objects passed to/from workers must be picklable:

| Object | Picklable? | Notes |
|--------|-----------|-------|
| `model.state_dict()` | Yes | Dict of CPU tensors |
| `MCTSConfig` | Yes | Dataclass of primitives |
| `model_hparams` (dict) | Yes | Dict of ints |
| `EpisodeResult` | Yes | Dataclass of primitives + list[TrainingExample] |
| `TrainingExample` | Yes | Dataclass with PyG Data + dict + float |
| `PyG Data` | Yes | Inherits torch.Tensor serialization |
| `ZXDiagram` (nx.Graph) | Yes | NetworkX graphs are pickle-friendly |

No custom `__reduce__` or `__getstate__` methods needed.

### 10. Error handling

- If a worker process crashes (e.g., segfault in a C extension, OOM), `ProcessPoolExecutor` raises a `BrokenProcessPool` exception in the main process.
- Wrap the `executor.map()` / `executor.submit()` calls in a try/except that logs the error and falls back to serial self-play for that iteration.
- Individual game failures within a worker are already caught by `SelfPlayWorker.play_episode()` (it catches `ValueError`, `KeyError`, etc. during `state.apply_action()`).

### 11. Performance expectations

With N workers, self-play time should scale roughly as:

```
T_parallel ≈ T_serial / N + T_overhead
```

Where `T_overhead` includes:
- Model state_dict serialization: ~10-50 ms (one-time per iteration)
- Process dispatch + result collection via pickle: ~100-500 ms total
- Worker startup (fork): ~50-100 ms per worker

For N=4 on a 4-core machine, expected speedup: **~3.5-3.8x** (near-linear; self-play is embarrassingly parallel with no shared state).

For N=8 on an 8-core machine: **~6-7x** speedup.

## Implementation Order

### Step 1: `_worker_play_games` function
Write the standalone worker function in `parallel_self_play.py`. This is the core unit of work — everything else wraps it.

### Step 2: `ParallelSelfPlayManager` class
Implement with the same `generate_games()` interface as `SelfPlayManager`. Start with `ProcessPoolExecutor` and simple game partitioning.

### Step 3: Wire into `Trainer`
Add `num_self_play_workers` to `TrainerConfig`, conditionally construct `ParallelSelfPlayManager` in `Trainer.__init__()`.

### Step 4: Curriculum integration
Extend `ParallelSelfPlayManager.generate_games()` to accept and distribute difficulty overrides, matching the existing `_generate_curriculum_games()` logic in Trainer.

### Step 5: Testing
- Unit test: `_worker_play_games` runs independently and returns valid `EpisodeResult` objects.
- Integration test: `ParallelSelfPlayManager.generate_games(8)` with 2 workers produces 8 results with examples.
- Regression test: serial vs parallel produce statistically similar results (same seed → same output for a single worker).
- Performance test: wall-clock time with N=4 is < 35% of serial time.

### Step 6: Logging polish
Suppress worker-level logging, add aggregate timing to main process output (e.g., "Self-play: 100 games in 4.2s across 4 workers (23.8 games/s)").

## Files to create/modify

| File | Action | Description |
|------|--------|-------------|
| `alphazx/mcts/parallel_self_play.py` | **CREATE** | Worker function + ParallelSelfPlayManager |
| `alphazx/mcts/trainer.py` | MODIFY | Conditional use of ParallelSelfPlayManager |
| `alphazx/mcts/config.py` | MODIFY | Add `num_self_play_workers` to TrainerConfig (or keep in TrainerConfig) |
| `tests/test_parallel_self_play.py` | **CREATE** | Unit + integration tests |

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Fork + third-party C extensions = occasional crashes | Low | Catch `BrokenProcessPool`, fall back to serial |
| Memory pressure from N model copies | Low (model is ~5 MB) | Monitor RSS; reduce workers if needed |
| Non-deterministic test failures from timing | Medium | Use short games (depth=2, num_qubits=2) in tests |
| Interleaved stdout from workers | Medium | Suppress worker logging to WARNING |
| Curriculum state mutation during parallel dispatch | None | Difficulty levels are pre-computed before dispatch |
