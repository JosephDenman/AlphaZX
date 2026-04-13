"""
Shared infrastructure used by all training paradigms (MCTS, GFlowNet, etc.).

Contains paradigm-agnostic components:
- GameState: ZX diagram state representation
- CircuitConfig: circuit generation and episode parameters
- CurriculumScheduler: progressive difficulty scaling
- ReplayBuffer / TrainingExample: experience storage
- TBLogger: TensorBoard logging
- evaluate_state / compute_action_prior: model inference utilities
- ACTION_TYPE_NAMES: human-readable action type labels
"""
