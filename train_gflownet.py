#!/usr/bin/env python3
"""
GFlowNet Training Entry Point for AlphaZX.

Trains a GFlowNet to sample high-reward ZX-calculus rewrite trajectories
using Trajectory Balance loss.

Usage:
    python train_gflownet.py                           # Default (small circuits)
    python train_gflownet.py --num-qubits 5 --depth 5 --num-iterations 100
    python train_gflownet.py --loss-type sub_trajectory_balance

All hyperparameters can be set via command-line arguments.
"""

import argparse
import logging
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from alphazx.diagram import METADATA, POSSIBLE_PHASES, NUM_POSSIBLE_NEW_EDGES
from alphazx.models.homogeneous.alphazx_model import AlphaZXModel
from alphazx.models.heterogeneous.alphazx_hetero_model import AlphaZXHeteroModel
from alphazx.gflownet.config import GFlowNetConfig
from alphazx.gflownet.environment import ZXGFlowNetEnv
from alphazx.gflownet.sampler import TrajectorySampler
from alphazx.gflownet.trainer import GFlowNetTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train GFlowNet for ZX-calculus simplification',
    )

    # --- Circuit parameters ---
    circuit = parser.add_argument_group('Circuit generation')
    circuit.add_argument('--num-qubits', type=int, default=10)
    circuit.add_argument('--depth', type=int, default=10)
    circuit.add_argument('--circuit-type', choices=['cnot_had_phase', 'clifford', 'cliffordT'],
                         default='cnot_had_phase')
    circuit.add_argument('--p-had', type=float, default=0.2)
    circuit.add_argument('--p-t', type=float, default=0.4)
    circuit.add_argument('--p-s', type=float, default=0.24,
                         help='S-gate probability for cliffordT circuits (default: 0.24)')
    circuit.add_argument('--p-hsh', type=float, default=0.25,
                         help='HSH gate probability for cliffordT circuits (default: 0.25)')
    circuit.add_argument('--min-initial-t-gates', type=int, default=2)
    circuit.add_argument('--max-t-gate-increase', type=int, default=5)
    circuit.add_argument('--max-episode-length', type=int, default=100)

    # --- Model architecture ---
    model = parser.add_argument_group('Model architecture')
    model.add_argument('--model-type', choices=['homogeneous', 'heterogeneous'],
                       default='heterogeneous')
    model.add_argument('--node-channels', type=int, default=16)
    model.add_argument('--edge-channels', type=int, default=4)
    model.add_argument('--pe-dim', type=int, default=16)
    # HGT-specific (heterogeneous model only)
    model.add_argument('--hgt-shared-layers', type=int, default=5,
                       help='Number of shared HGTConv layers (default: 1)')
    model.add_argument('--hgt-policy-layers', type=int, default=5,
                       help='Number of policy-specific HGTConv layers (default: 1)')
    model.add_argument('--hgt-value-layers', type=int, default=5,
                       help='Number of value-specific HGTConv layers (default: 1)')
    model.add_argument('--hgt-heads', type=int, default=4,
                       help='Number of attention heads in HGTConv (default: 2)')

    # --- GFlowNet parameters ---
    gfn = parser.add_argument_group('GFlowNet')
    gfn.add_argument('--loss-type',
                     choices=['trajectory_balance', 'sub_trajectory_balance',
                              'detailed_balance', 'flow_matching'],
                     default='trajectory_balance',
                     help='GFlowNet loss objective (default: trajectory_balance)')
    gfn.add_argument('--reward-exponent', type=float, default=4.0,
                     help='Target exponent for terminal reward R(x). Higher = more '
                          'concentrated on best trajectories (default: 4.0)')
    gfn.add_argument('--reward-exponent-initial', type=float, default=1.0,
                     help='Starting exponent for annealing (default: 1.0). '
                          'Set equal to --reward-exponent to disable annealing.')
    gfn.add_argument('--reward-exponent-warmup', type=int, default=200,
                     help='Iterations to anneal exponent from initial to target '
                          '(default: 200). 0 = no annealing.')
    gfn.add_argument('--min-reward', type=float, default=0.01,
                     help='Minimum reward floor after exponentiation (default: 0.01)')
    gfn.add_argument('--trajectories-per-batch', type=int, default=8,
                     help='Trajectories sampled per training step (default: 8)')
    gfn.add_argument('--max-trajectory-length', type=int, default=20,
                     help='Maximum rewrite steps per trajectory (default: 20)')
    gfn.add_argument('--sampling-temperature', type=float, default=1.0,
                     help='Temperature for forward policy sampling (default: 1.0)')
    gfn.add_argument('--eval-temperature', type=float, default=1.0,
                     help='Temperature for evaluation sampling (default: 1.0). '
                          'GFlowNets sample ∝ R(x); use training temp for eval.')
    gfn.add_argument('--epsilon-uniform', type=float, default=0.05,
                     help='Uniform exploration probability (default: 0.05)')
    gfn.add_argument('--reward-shaping-coeff', type=float, default=0.0,
                     help='DEPRECATED — leave at 0.0.  Intermediate reward shaping '
                          'via exponential bonus is redundant with SubTB flow '
                          'estimation and causes gradient explosion on small circuits.')
    gfn.add_argument('--subtb-lambda', type=float, default=0.9,
                     help='SubTB interpolation parameter (default: 0.9). '
                          '0=DB (local), 1=TB (global). '
                          'Only used with --loss-type sub_trajectory_balance.')
    gfn.add_argument('--grad-clip-max-norm', type=float, default=1.0,
                     help='Gradient clipping max norm (default: 1.0)')
    gfn.add_argument('--replay-ratio', type=float, default=0.0,
                     help='Fraction of batch from replay buffer (default: 0.0 = off). '
                          'Recommended: 0.25-0.5')
    gfn.add_argument('--replay-buffer-size', type=int, default=1000,
                     help='Max trajectories in replay buffer (default: 1000)')
    gfn.add_argument('--replay-min-reward', type=float, default=0.02,
                     help='(Deprecated) Min shaped reward for replay buffer admission. '
                          'Ignored when --replay-min-reduction is set (default: 0.02)')
    gfn.add_argument('--replay-min-reduction', type=float, default=0.02,
                     help='Min T-count reduction ratio for replay buffer admission. '
                          'Compared against (initial_T - final_T) / initial_T, '
                          'independent of reward exponent. 0.02 = 2%% reduction. '
                          '0.0 = disable (fall back to --replay-min-reward). '
                          '(default: 0.02)')
    gfn.add_argument('--replay-diversity-weight', type=float, default=0.1,
                     help='Diversity bonus weight in replay priority (default: 0.1)')

    # --- Training parameters ---
    train = parser.add_argument_group('Training')
    train.add_argument('--num-iterations', type=int, default=500,
                       help='Number of training iterations (default: 100)')
    train.add_argument('--lr', type=float, default=1e-3)
    train.add_argument('--log-z-lr', type=float, default=5e-2,
                       help='Learning rate for log_Z parameter (default: 0.05). '
                            'Should be ~50x the model LR (Malkin et al. 2022)')
    train.add_argument('--weight-decay', type=float, default=1e-4)
    train.add_argument('--lr-schedule', choices=['cosine', 'cosine_restarts', 'constant'],
                       default='cosine_restarts')
    train.add_argument('--lr-restart-period', type=int, default=50,
                       help='Period for cosine warm restarts (default: 50)')

    # --- Evaluation ---
    evl = parser.add_argument_group('Evaluation')
    evl.add_argument('--eval-interval', type=int, default=10,
                     help='Evaluate every N iterations (default: 10)')
    evl.add_argument('--eval-games', type=int, default=30,
                     help='Circuits to evaluate on (default: 10)')
    evl.add_argument('--eval-scales', type=str, default=None,
                     help='Comma-separated QxD specs for cross-scale evaluation. '
                          'E.g. "20x160,40x290,80x525". Evaluated at the same '
                          'interval as --eval-interval. Uses the training circuit '
                          'type. (default: off)')
    evl.add_argument('--eval-scale-games', type=int, default=20,
                     help='Circuits per cross-scale eval point (default: 20)')
    evl.add_argument('--eval-circuit-types', type=str, default=None,
                     help='Comma-separated circuit types to eval on in addition '
                          'to the training type.  E.g. "cliffordT,cnot_had_phase". '
                          'Uses the same num_qubits and depth. (default: off)')
    evl.add_argument('--eval-benchmarks', type=str, default=None,
                     help='Benchmark suite(s) to evaluate on.  Comma-separated '
                          'from: small, medium, large, all. E.g. "small,medium". '
                          'Uses fixed arithmetic/QFT benchmark circuits from the '
                          'ZX-calculus literature. (default: off)')
    evl.add_argument('--benchmark-dir', type=str, default=None,
                     help='Path to benchmark_circuits directory. '
                          '(default: <project_root>/benchmark_circuits/)')

    # --- Checkpointing ---
    ckpt = parser.add_argument_group('Checkpointing')
    ckpt.add_argument('--checkpoint-dir', type=str, default='checkpoints_gfn')
    ckpt.add_argument('--checkpoint-interval', type=int, default=20)

    # --- System ---
    sys_group = parser.add_argument_group('System')
    sys_group.add_argument('--device', type=str, default='cpu')
    sys_group.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING'],
                           default='INFO')
    sys_group.add_argument('--num-workers', type=int, default=0,
                           help='Parallel sampling workers (default: 0 = sequential). '
                                'Workers sample trajectories without gradients, then '
                                'the main process replays with gradients. '
                                'Recommended: 2-8 depending on CPU cores.')

    return parser.parse_args()


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Build the neural network model (same architecture as MCTS)."""
    num_node_types = len(METADATA.node_type_abbrevs)
    num_possible_phases = len(POSSIBLE_PHASES)
    num_possible_new_edges = NUM_POSSIBLE_NEW_EDGES
    num_edge_embeddings = len(METADATA.edge_feat_to_index_dict)

    if args.model_type == 'homogeneous':
        return AlphaZXModel(
            num_node_types=num_node_types,
            num_possible_phases=num_possible_phases,
            num_possible_new_edges=num_possible_new_edges,
            node_embedding_channels=args.node_channels,
            num_edge_embeddings=num_edge_embeddings,
            edge_embedding_channels=args.edge_channels,
            pe_in_channels=args.pe_dim,
            pe_out_channels=args.pe_dim,
        )
    else:
        return AlphaZXHeteroModel(
            num_node_types=num_node_types,
            num_possible_phases=num_possible_phases,
            num_possible_new_edges=num_possible_new_edges,
            node_embedding_channels=args.node_channels,
            num_edge_embeddings=num_edge_embeddings,
            edge_embedding_channels=args.edge_channels,
            pe_in_channels=args.pe_dim,
            pe_out_channels=args.pe_dim,
            hgt_num_shared_layers=args.hgt_shared_layers,
            hgt_num_policy_layers=args.hgt_policy_layers,
            hgt_num_value_layers=args.hgt_value_layers,
            hgt_heads=args.hgt_heads,
        )


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        force=True,
    )
    # Ensure output is unbuffered so lines appear immediately
    for handler in logging.root.handlers:
        if hasattr(handler, 'stream'):
            handler.stream = sys.stdout
    logger = logging.getLogger('train_gflownet')

    # Build config
    config = GFlowNetConfig(
        # Circuit
        num_qubits=args.num_qubits,
        depth=args.depth,
        circuit_type=args.circuit_type,
        p_had=args.p_had,
        p_t=args.p_t,
        p_s=args.p_s,
        p_hsh=args.p_hsh,
        min_initial_t_gates=args.min_initial_t_gates,
        max_t_gate_increase=args.max_t_gate_increase,
        max_episode_length=args.max_episode_length,
        pe_dim=args.pe_dim,
        # GFlowNet
        loss_type=args.loss_type,
        reward_exponent=args.reward_exponent,
        reward_exponent_initial=args.reward_exponent_initial,
        reward_exponent_warmup_iters=args.reward_exponent_warmup,
        min_reward=args.min_reward,
        trajectories_per_batch=args.trajectories_per_batch,
        max_trajectory_length=args.max_trajectory_length,
        sampling_temperature=args.sampling_temperature,
        eval_temperature=args.eval_temperature,
        epsilon_uniform=args.epsilon_uniform,
        reward_shaping_coeff=args.reward_shaping_coeff,
        # Replay buffer
        subtb_lambda=args.subtb_lambda,
        grad_clip_max_norm=args.grad_clip_max_norm,
        replay_ratio=args.replay_ratio,
        replay_buffer_size=args.replay_buffer_size,
        replay_min_reward=args.replay_min_reward,
        replay_min_reduction_ratio=args.replay_min_reduction,
        replay_diversity_weight=args.replay_diversity_weight,
        # Parallel sampling
        num_sampling_workers=args.num_workers,
        # Training
        learning_rate=args.lr,
        log_z_learning_rate=args.log_z_lr,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        lr_restart_period=args.lr_restart_period,
    )

    # Build model
    model = build_model(args)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {args.model_type} with {param_count:,} parameters")
    logger.info(f"Config: {args.num_qubits}q / d{args.depth} / {args.circuit_type}")
    anneal_str = (f"{args.reward_exponent_initial}→{args.reward_exponent}"
                  f"/{args.reward_exponent_warmup}it"
                  if args.reward_exponent_warmup > 0
                  else str(args.reward_exponent))
    subtb_str = f", λ={args.subtb_lambda}" if args.loss_type == 'sub_trajectory_balance' else ""
    logger.info(f"GFlowNet: loss={args.loss_type}{subtb_str}, "
                f"reward_exp={anneal_str}, "
                f"batch={args.trajectories_per_batch}, max_len={args.max_trajectory_length}, "
                + (f"shaping={args.reward_shaping_coeff} (DEPRECATED), " if args.reward_shaping_coeff > 0 else "")
                + f"replay={args.replay_ratio:.0%}×{args.replay_buffer_size}, "
                f"grad_clip={args.grad_clip_max_norm}")
    logger.info(f"Optimizer: model_lr={args.lr}, log_z_lr={args.log_z_lr}, "
                f"weight_decay={args.weight_decay}, schedule={args.lr_schedule}"
                + (f", T_0={args.lr_restart_period}"
                   if args.lr_schedule == 'cosine_restarts' else ""))
    if args.num_workers > 0:
        logger.info(f"Parallel sampling: {args.num_workers} workers "
                     f"(sample-then-replay architecture)")

    # Build trainer
    trainer = GFlowNetTrainer(model, config, device=args.device)

    # Update scheduler T_max for plain cosine (not needed for warm restarts)
    if (trainer.scheduler is not None
            and isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)):
        trainer.scheduler.T_max = args.num_iterations

    # Parse cross-scale evaluation specs
    eval_scales = []
    if args.eval_scales:
        for spec in args.eval_scales.split(','):
            spec = spec.strip()
            q, d = spec.split('x')
            eval_scales.append((int(q), int(d)))
        logger.info(f"Cross-scale eval: {eval_scales} "
                     f"({args.eval_scale_games} games each)")

    # Parse additional eval circuit types
    eval_circuit_types = []
    if args.eval_circuit_types:
        for ct in args.eval_circuit_types.split(','):
            ct = ct.strip()
            if ct and ct != args.circuit_type:
                eval_circuit_types.append(ct)
        if eval_circuit_types:
            logger.info(f"Cross-type eval: {eval_circuit_types} "
                         f"(in addition to training type '{args.circuit_type}')")

    # Load benchmark circuits for eval
    benchmark_circuits = []
    if args.eval_benchmarks:
        from alphazx.diagram.benchmark_circuits import (
            get_small_benchmarks, get_medium_benchmarks, get_large_benchmarks,
            load_all_benchmarks,
        )
        suites = [s.strip().lower() for s in args.eval_benchmarks.split(',')]
        for suite in suites:
            if suite == 'small':
                benchmark_circuits.extend(get_small_benchmarks())
            elif suite == 'medium':
                benchmark_circuits.extend(get_medium_benchmarks())
            elif suite == 'large':
                benchmark_circuits.extend(get_large_benchmarks())
            elif suite == 'all':
                benchmark_circuits = load_all_benchmarks()
                break
            else:
                logger.warning(f"Unknown benchmark suite: {suite}")
        # Deduplicate by name
        seen = set()
        deduped = []
        for bc in benchmark_circuits:
            if bc.name not in seen:
                seen.add(bc.name)
                deduped.append(bc)
        benchmark_circuits = sorted(deduped, key=lambda c: c.t_count)
        logger.info(f"Benchmark eval: {len(benchmark_circuits)} circuits "
                     f"({', '.join(s for s in suites)})")

    # Checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    logger.info(f"Starting GFlowNet training for {args.num_iterations} iterations")
    t_start = time.time()

    for iteration in range(1, args.num_iterations + 1):
        logger.info(f"iter={iteration:4d}  sampling {args.trajectories_per_batch} "
                     f"trajectories (max {args.max_trajectory_length} steps each)...")
        sys.stdout.flush()
        metrics = trainer.train_step()

        shaped_suffix = ""
        if args.reward_shaping_coeff > 0:
            shaped_suffix = f"  shaped_R={metrics.mean_shaped_reward:.4f}"
        replay_suffix = ""
        if args.replay_ratio > 0:
            replay_suffix = (
                f"  replay={metrics.num_replay_trajectories}"
                f"  buf={metrics.replay_buffer_size}"
                f"/{metrics.replay_buffer_unique_fps}fp"
            )

        exp_suffix = ""
        if args.reward_exponent_warmup > 0:
            exp_suffix = f"  exp={metrics.reward_exponent:.2f}"

        logger.info(
            f"iter={iteration:4d}  "
            f"TB_loss={metrics.tb_loss:8.4f}  "
            f"log_Z={metrics.log_Z:7.3f}  "
            f"T_reduced={metrics.mean_t_gate_reduction:+.2f}  "
            f"max_reduced={metrics.max_t_gate_reduction:+d}  "
            f"positive={metrics.frac_positive_reduction:.0%}  "
            f"rewrites={metrics.mean_rewrites:.1f}  "
            f"sub_steps={metrics.mean_sub_steps:.0f}  "
            f"reward={metrics.mean_terminal_reward:.4f}"
            f"{shaped_suffix}"
            f"{exp_suffix}"
            f"  grad={metrics.grad_norm:.3f}  "
            f"lr={metrics.learning_rate:.2e}  "
            f"time={metrics.step_time:.1f}s"
            f"{replay_suffix}"
        )

        # Evaluate
        if iteration % args.eval_interval == 0:
            eval_metrics = trainer.evaluate(num_games=args.eval_games)
            # Agent line
            logger.info(
                f"  EVAL agent:  T_reduced={eval_metrics.mean_t_gate_reduction:+.2f}  "
                f"ratio={eval_metrics.mean_reduction_ratio:.1%}  "
                f"max={eval_metrics.max_t_gate_reduction:+d}  "
                f"positive={eval_metrics.frac_positive_reduction:.0%}  "
                f"time={eval_metrics.eval_time:.1f}s"
            )
            # PyZX baseline line
            if eval_metrics.pyzx_mean_t_gate_reduction is not None:
                logger.info(
                    f"  EVAL pyzx:   T_reduced={eval_metrics.pyzx_mean_t_gate_reduction:+.2f}  "
                    f"ratio={eval_metrics.pyzx_mean_reduction_ratio:.1%}  "
                    f"vs_pyzx={eval_metrics.agent_wins}W/"
                    f"{eval_metrics.agent_ties}T/"
                    f"{eval_metrics.agent_losses}L"
                )

            # Cross-scale evaluation
            if eval_scales:
                for eq, ed in eval_scales:
                    from copy import deepcopy
                    scale_config = deepcopy(config)
                    scale_config.num_qubits = eq
                    scale_config.depth = ed
                    scale_env = ZXGFlowNetEnv(scale_config)
                    scale_sampler = TrajectorySampler(
                        env=scale_env,
                        policy=trainer.policy,
                        device=args.device,
                        temperature=config.eval_temperature,
                        epsilon_uniform=0.0,
                        max_trajectory_length=config.max_trajectory_length,
                        reward_exponent=config.reward_exponent,
                        min_reward=config.min_reward,
                    )
                    scale_trajs = scale_sampler.sample_batch_with_pyzx(
                        args.eval_scale_games,
                    )
                    # Compute metrics
                    t_reds = [t.t_gate_reduction for t in scale_trajs]
                    ratios = [t.t_gate_reduction / t.initial_t_gates
                              for t in scale_trajs if t.initial_t_gates > 0]
                    pos_frac = sum(1 for r in t_reds if r > 0) / max(1, len(t_reds))
                    # PyZX baseline
                    pyzx_reds = []
                    sw, st, sl = 0, 0, 0
                    for t in scale_trajs:
                        pf, pr = trainer._run_pyzx_baseline(t.pyzx_graph)
                        if pr is not None:
                            pyzx_reds.append(pr)
                            ar = t.t_gate_reduction
                            if ar > pr: sw += 1
                            elif ar == pr: st += 1
                            else: sl += 1
                    pyzx_mean = (sum(pyzx_reds) / len(pyzx_reds)
                                 if pyzx_reds else 0)
                    logger.info(
                        f"  SCALE {eq}x{ed}:  "
                        f"T_reduced={sum(t_reds)/max(1,len(t_reds)):+.2f}  "
                        f"ratio={sum(ratios)/max(1,len(ratios)):.1%}  "
                        f"positive={pos_frac:.0%}  "
                        f"pyzx={pyzx_mean:+.2f}  "
                        f"vs_pyzx={sw}W/{st}T/{sl}L"
                    )

            # Cross-type evaluation (e.g., train on cnot_had_phase, eval on cliffordT)
            for ct in eval_circuit_types:
                from copy import deepcopy
                ct_config = deepcopy(config)
                ct_config.circuit_type = ct
                # Use Riu et al. defaults for cliffordT
                if ct == 'cliffordT':
                    ct_config.p_t = 0.17
                    ct_config.p_s = 0.24
                    ct_config.p_hsh = 0.25
                ct_env = ZXGFlowNetEnv(ct_config)
                ct_sampler = TrajectorySampler(
                    env=ct_env,
                    policy=trainer.policy,
                    device=args.device,
                    temperature=config.eval_temperature,
                    epsilon_uniform=0.0,
                    max_trajectory_length=config.max_trajectory_length,
                    reward_exponent=config.reward_exponent,
                    min_reward=config.min_reward,
                )
                ct_trajs = ct_sampler.sample_batch_with_pyzx(args.eval_games)
                ct_reds = [t.t_gate_reduction for t in ct_trajs]
                ct_ratios = [t.t_gate_reduction / t.initial_t_gates
                             for t in ct_trajs if t.initial_t_gates > 0]
                ct_pos = sum(1 for r in ct_reds if r > 0) / max(1, len(ct_reds))
                # PyZX baseline
                ct_pyzx_reds = []
                cw, cti, cl = 0, 0, 0
                for t in ct_trajs:
                    pf, pr = trainer._run_pyzx_baseline(t.pyzx_graph)
                    if pr is not None:
                        ct_pyzx_reds.append(pr)
                        ar = t.t_gate_reduction
                        if ar > pr: cw += 1
                        elif ar == pr: cti += 1
                        else: cl += 1
                ct_pyzx_mean = (sum(ct_pyzx_reds) / len(ct_pyzx_reds)
                                if ct_pyzx_reds else 0)
                logger.info(
                    f"  TYPE {ct}:  "
                    f"T_reduced={sum(ct_reds)/max(1,len(ct_reds)):+.2f}  "
                    f"ratio={sum(ct_ratios)/max(1,len(ct_ratios)):.1%}  "
                    f"positive={ct_pos:.0%}  "
                    f"pyzx={ct_pyzx_mean:+.2f}  "
                    f"vs_pyzx={cw}W/{cti}T/{cl}L"
                )

            # Benchmark circuit evaluation (fixed arithmetic/QFT circuits)
            if benchmark_circuits:
                bench_results = trainer.evaluate_benchmarks(benchmark_circuits)
                evaluated = [r for r in bench_results if r['winner'] != 'skipped']
                skipped = [r for r in bench_results if r['winner'] == 'skipped']
                wins = sum(1 for r in evaluated if r['winner'] == 'agent')
                ties = sum(1 for r in evaluated if r['winner'] == 'tie')
                losses = sum(1 for r in evaluated if r['winner'] == 'pyzx')
                n_eval = len(evaluated)
                skip_str = f"  skipped={len(skipped)}" if skipped else ""
                if n_eval > 0:
                    total_agent_red = sum(r['agent_reduced'] for r in evaluated)
                    total_pyzx_red = sum(r['pyzx_reduced'] for r in evaluated)
                    logger.info(
                        f"  BENCH ({n_eval} circuits):  "
                        f"agent_red={total_agent_red/n_eval:+.1f}  "
                        f"pyzx_red={total_pyzx_red/n_eval:+.1f}  "
                        f"vs_pyzx={wins}W/{ties}T/{losses}L"
                        f"{skip_str}"
                    )
                    for r in evaluated:
                        if r['winner'] == 'agent':
                            logger.info(
                                f"    WIN  {r['name']}: "
                                f"{r['initial_t']}→{r['agent_final_t']} "
                                f"(agent -{r['agent_reduced']}) vs "
                                f"(pyzx -{r['pyzx_reduced']})"
                            )
                else:
                    logger.info(
                        f"  BENCH: all {len(skipped)} circuits skipped "
                        f"(unsupported phases)"
                    )
                for r in skipped:
                    logger.warning(
                        f"    SKIP {r['name']}: unsupported phases"
                    )

        # Checkpoint
        if iteration % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'gfn_iter_{iteration:04d}.pt')
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'log_Z': trainer.policy.log_Z.item(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'config': config,
            }, ckpt_path)
            logger.info(f"  Checkpoint saved: {ckpt_path}")

    elapsed = time.time() - t_start
    logger.info(f"Training complete in {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Final evaluation
    final_eval = trainer.evaluate(num_games=args.eval_games * 2)
    logger.info(
        f"FINAL agent:  T_reduced={final_eval.mean_t_gate_reduction:+.2f}  "
        f"ratio={final_eval.mean_reduction_ratio:.1%}  "
        f"positive={final_eval.frac_positive_reduction:.0%}"
    )
    if final_eval.pyzx_mean_t_gate_reduction is not None:
        logger.info(
            f"FINAL pyzx:   T_reduced={final_eval.pyzx_mean_t_gate_reduction:+.2f}  "
            f"ratio={final_eval.pyzx_mean_reduction_ratio:.1%}  "
            f"vs_pyzx={final_eval.agent_wins}W/"
            f"{final_eval.agent_ties}T/"
            f"{final_eval.agent_losses}L"
        )


if __name__ == '__main__':
    main()
