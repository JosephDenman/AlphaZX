#!/usr/bin/env python3
"""
AlphaZX Training Entry Point.

Runs the full AlphaZero-style training loop:
1. Generate self-play games using MCTS
2. Train the policy + value network on replay buffer data
3. Evaluate against PyZX baseline periodically
4. Checkpoint the model

Usage:
    python train_alphazx.py                         # Default settings (small circuits)
    python train_alphazx.py --num-qubits 5 --depth 10 --num-iterations 200
    python train_alphazx.py --resume checkpoints/checkpoint_latest.pt

All hyperparameters can be set via command-line arguments.
"""

import argparse
import logging
import os
import sys
from typing import Optional

# Enable MPS fallback before importing torch.  PyG's attention aggregation uses
# scatter_reduce which is not yet implemented natively on Apple Silicon (MPS).
# This lets unsupported ops fall back to CPU while everything else runs on GPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from alphazx.diagram import METADATA, POSSIBLE_PHASES
from alphazx.models.homogeneous.alphazx_model import AlphaZXModel
from alphazx.mcts.config import MCTSConfig
from alphazx.mcts.curriculum import CurriculumScheduler, CurriculumConfig
from alphazx.mcts.trainer import Trainer, TrainerConfig
from alphazx.mcts.replay_buffer import ReplayBuffer
from alphazx.mcts.evaluator import Evaluator
from alphazx.mcts.tb_logger import TBLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train AlphaZX: AlphaZero for ZX-calculus simplification'
    )

    # --- Circuit parameters ---
    circuit = parser.add_argument_group('Circuit generation')
    circuit.add_argument('--num-qubits', type=int, default=10,
                         help='Number of qubits in training circuits (default: 10)')
    circuit.add_argument('--depth', type=int, default=10,
                         help='Depth of training circuits (default: 10)')
    circuit.add_argument('--max-episode-length', type=int, default=100,
                         help='Max steps per episode (default: 100)')
    circuit.add_argument('--circuit-type', choices=['cnot_had_phase', 'clifford'],
                         default='cnot_had_phase',
                         help='Circuit generator type (default: cnot_had_phase)')
    circuit.add_argument('--p-had', type=float, default=0.2,
                         help='Hadamard gate probability for CNOT_HAD_PHASE (default: 0.2)')
    circuit.add_argument('--p-t', type=float, default=0.4,
                         help='T-gate probability for CNOT_HAD_PHASE (default: 0.4)')
    circuit.add_argument('--max-t-gate-increase', type=int, default=5,
                         help='Terminate episode if T-gates exceed initial by this much (default: 5, 0=disable)')
    circuit.add_argument('--min-initial-t-gates', type=int, default=2,
                         help='Re-roll circuits with fewer T-gates than this (default: 2)')

    # --- Model architecture ---
    model_group = parser.add_argument_group('Model architecture')
    model_group.add_argument('--node-channels', type=int, default=64,
                             help='Node embedding dimension (default: 64)')
    model_group.add_argument('--edge-channels', type=int, default=64,
                             help='Edge embedding dimension (default: 64)')
    model_group.add_argument('--pe-dim', type=int, default=20,
                             help='Positional encoding dimension (default: 20)')

    # --- MCTS parameters ---
    mcts = parser.add_argument_group('MCTS')
    mcts.add_argument('--num-simulations', type=int, default=50,
                      help='MCTS simulations per move (default: 50)')
    mcts.add_argument('--c-puct', type=float, default=1.5,
                      help='PUCT exploration constant (default: 1.5)')
    mcts.add_argument('--pw-alpha', type=float, default=0.5,
                      help='Progressive widening exponent (default: 0.5)')
    mcts.add_argument('--temperature', type=float, default=1.0,
                      help='MCTS temperature for self-play (default: 1.0)')
    mcts.add_argument('--leaf-batch-size', type=int, default=8,
                      help='Batch size for leaf evaluation in MCTS (default: 8). '
                           'Higher values give more NN throughput but less accurate '
                           'visit counts within each wave. Set to 1 to disable batching.')

    # --- Training parameters ---
    train = parser.add_argument_group('Training')
    train.add_argument('--num-iterations', type=int, default=100,
                       help='Number of training iterations (default: 100)')
    train.add_argument('--self-play-games', type=int, default=50,
                       help='Self-play games per iteration (default: 50)')
    train.add_argument('--training-steps', type=int, default=200,
                       help='Gradient steps per iteration (default: 200)')
    train.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    train.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    train.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    train.add_argument('--c-value', type=float, default=1.0,
                       help='Value loss weight (default: 1.0)')
    train.add_argument('--lr-schedule', choices=['cosine', 'constant', 'step'],
                       default='constant', help='LR schedule (default: constant)')

    # --- Replay buffer ---
    buf = parser.add_argument_group('Replay buffer')
    buf.add_argument('--buffer-capacity', type=int, default=100000,
                     help='Replay buffer capacity (default: 100000)')
    buf.add_argument('--min-buffer-size', type=int, default=256,
                     help='Min buffer size before training starts (default: 256)')

    # --- Evaluation ---
    ev = parser.add_argument_group('Evaluation')
    ev.add_argument('--eval-interval', type=int, default=5,
                    help='Evaluate every N iterations (default: 5)')
    ev.add_argument('--eval-games', type=int, default=20,
                    help='Games per evaluation (default: 20)')
    ev.add_argument('--eval-temperature', type=float, default=0.1,
                    help='Temperature for evaluation (default: 0.1)')
    ev.add_argument('--no-pyzx-comparison', action='store_true',
                    help='Disable PyZX baseline comparison')

    # --- Curriculum learning ---
    cur = parser.add_argument_group('Curriculum learning')
    cur.add_argument('--curriculum', action='store_true',
                     help='Enable curriculum learning (start with small circuits, '
                          'progressively increase difficulty)')
    cur.add_argument('--curriculum-strategy', choices=['performance', 'linear'],
                     default='performance',
                     help='Advancement strategy (default: performance)')
    cur.add_argument('--curriculum-start-qubits', type=int, default=2,
                     help='Starting number of qubits (default: 2)')
    cur.add_argument('--curriculum-start-depth', type=int, default=3,
                     help='Starting circuit depth (default: 3)')
    cur.add_argument('--curriculum-advance-threshold', type=float, default=0.15,
                     help='T-gate reduction ratio required to advance (default: 0.15)')
    cur.add_argument('--curriculum-advance-window', type=int, default=2,
                     help='Consecutive iterations meeting threshold to advance (default: 2)')
    cur.add_argument('--curriculum-linear-every', type=int, default=10,
                     help='For linear strategy: advance every N iterations (default: 10)')
    cur.add_argument('--curriculum-qubit-step', type=int, default=1,
                     help='Qubits added per advancement (default: 1)')
    cur.add_argument('--curriculum-depth-step', type=int, default=2,
                     help='Depth added per advancement (default: 2)')
    cur.add_argument('--curriculum-mix-easier', type=float, default=0.1,
                     help='Fraction of games at easier difficulty (default: 0.1)')
    cur.add_argument('--curriculum-mix-harder', type=float, default=0.1,
                     help='Fraction of games at harder difficulty (default: 0.1)')

    # --- Checkpointing ---
    ckpt = parser.add_argument_group('Checkpointing')
    ckpt.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                      help='Checkpoint directory (default: checkpoints)')
    ckpt.add_argument('--checkpoint-interval', type=int, default=5,
                      help='Checkpoint every N iterations (default: 5)')
    ckpt.add_argument('--resume', type=str, default=None,
                      help='Resume from checkpoint path')

    # --- Device ---
    parser.add_argument('--device', type=str, default=None,
                        help='Device (auto-detected if not specified)')

    # --- TensorBoard ---
    tb = parser.add_argument_group('TensorBoard')
    tb.add_argument('--tensorboard-dir', type=str, default='runs/alphazx',
                    help='TensorBoard log directory (default: runs/alphazx)')
    tb.add_argument('--no-tensorboard', action='store_true',
                    help='Disable TensorBoard logging')

    # --- Logging ---
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING'],
                        default='INFO', help='Logging level')

    return parser.parse_args()


def build_model(args: argparse.Namespace) -> AlphaZXModel:
    """Build the AlphaZXModel from command-line arguments."""
    return AlphaZXModel(
        num_node_types=len(METADATA.node_type_abbrevs),
        num_possible_phases=len(POSSIBLE_PHASES),
        num_possible_new_edges=5,
        node_embedding_channels=args.node_channels,
        num_edge_embeddings=len(METADATA.edge_feat_to_index_dict),
        edge_embedding_channels=args.edge_channels,
        pe_in_channels=args.pe_dim,
        pe_out_channels=args.pe_dim,
    )


def build_mcts_config(args: argparse.Namespace) -> MCTSConfig:
    """Build MCTSConfig from command-line arguments."""
    return MCTSConfig(
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        pw_alpha=args.pw_alpha,
        temperature=args.temperature,
        pe_dim=args.pe_dim,
        num_qubits=args.num_qubits,
        depth=args.depth,
        max_episode_length=args.max_episode_length,
        circuit_type=args.circuit_type,
        p_had=args.p_had,
        p_t=args.p_t,
        max_t_gate_increase=args.max_t_gate_increase,
        min_initial_t_gates=args.min_initial_t_gates,
        leaf_batch_size=args.leaf_batch_size,
    )


def build_trainer_config(args: argparse.Namespace) -> TrainerConfig:
    """Build TrainerConfig from command-line arguments."""
    return TrainerConfig(
        num_self_play_games=args.self_play_games,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        c_value=args.c_value,
        lr_schedule=args.lr_schedule,
        num_iterations=args.num_iterations,
        min_buffer_size=args.min_buffer_size,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard=not args.no_tensorboard,
        tensorboard_dir=args.tensorboard_dir,
    )


def build_curriculum(args: argparse.Namespace) -> Optional[CurriculumScheduler]:
    """Build CurriculumScheduler from command-line arguments, or None if disabled."""
    if not args.curriculum:
        return None
    config = CurriculumConfig(
        enabled=True,
        start_num_qubits=args.curriculum_start_qubits,
        start_depth=args.curriculum_start_depth,
        target_num_qubits=args.num_qubits,
        target_depth=args.depth,
        strategy=args.curriculum_strategy,
        advance_threshold=args.curriculum_advance_threshold,
        advance_window=args.curriculum_advance_window,
        linear_advance_every=args.curriculum_linear_every,
        qubit_step=args.curriculum_qubit_step,
        depth_step=args.curriculum_depth_step,
        mix_easier_fraction=args.curriculum_mix_easier,
        mix_harder_fraction=args.curriculum_mix_harder,
    )
    return CurriculumScheduler(config)


def detect_device(requested: Optional[str]) -> torch.device:
    """Auto-detect the best available device."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('alphazx')

    device = detect_device(args.device)
    logger.info(f"Using device: {device}")

    # Build components
    model = build_model(args).to(device)
    mcts_config = build_mcts_config(args)
    trainer_config = build_trainer_config(args)
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity)
    curriculum = build_curriculum(args)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Evaluator
    evaluator = None
    if not args.no_pyzx_comparison:
        evaluator = Evaluator(
            mcts_config=mcts_config,
            device=device,
            compare_pyzx=True,
            eval_temperature=args.eval_temperature,
        )

    # TensorBoard logger
    tb_logger = None
    if trainer_config.tensorboard:
        tb_logger = TBLogger(
            log_dir=trainer_config.tensorboard_dir,
            enabled=True,
        )
        logger.info(f"TensorBoard: {trainer_config.tensorboard_dir}")
        logger.info(f"  View with: tensorboard --logdir {trainer_config.tensorboard_dir}")

    # Build or resume trainer
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer = Trainer.load_checkpoint(
            model, args.resume, mcts_config, trainer_config, replay_buffer, device
        )
        trainer.evaluator = evaluator
        trainer.tb_logger = tb_logger
        if curriculum and trainer.curriculum is None:
            trainer.curriculum = curriculum
    else:
        trainer = Trainer(
            model=model,
            mcts_config=mcts_config,
            trainer_config=trainer_config,
            replay_buffer=replay_buffer,
            device=device,
            evaluator=evaluator,
            tb_logger=tb_logger,
            curriculum=curriculum,
        )

    # Log configuration summary
    if curriculum:
        logger.info(
            f"Curriculum: {curriculum.config.start_num_qubits}q/d{curriculum.config.start_depth} → "
            f"{args.num_qubits}q/d{args.depth} "
            f"({curriculum.config.strategy}, {len(curriculum.levels)} levels)"
        )
    else:
        logger.info(f"Circuit: {args.num_qubits} qubits, depth {args.depth}")
    logger.info(f"MCTS: {args.num_simulations} simulations, c_puct={args.c_puct}")
    logger.info(f"Training: {args.num_iterations} iterations, "
                f"{args.self_play_games} games/iter, "
                f"{args.training_steps} steps/iter, "
                f"batch_size={args.batch_size}, lr={args.lr}")
    logger.info(f"Buffer: capacity={args.buffer_capacity}")

    # Run training
    try:
        metrics = trainer.train()
        logger.info("Training complete!")

        # Print final summary
        if metrics:
            final = metrics[-1]
            logger.info(
                f"Final iteration: "
                f"policy_loss={final.avg_policy_loss:.4f}, "
                f"value_loss={final.avg_value_loss:.4f}, "
                f"avg_t_reduced={final.avg_t_gates_reduced:.2f}, "
                f"simplification={final.simplification_rate:.1%}"
            )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)


if __name__ == '__main__':
    main()
