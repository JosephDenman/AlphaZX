#!/usr/bin/env python3
"""
Smoke-test script: verify the full AlphaZero loop is learning.

Runs multiple iterations of self-play → training → self-play, printing
per-iteration summaries so you can see whether:

  1. Loss decreases within each training phase (gradient mechanics work).
  2. Self-play performance improves across iterations (the model is learning
     to make better decisions): shorter episodes, fewer T-gate increases,
     more T-gate reductions.

This is a middle ground between unit tests and a full training run.
It exercises the complete loop but with smaller circuits, fewer simulations,
and fewer iterations than a real run.

Expected runtime: ~30-60 minutes on CPU (5 iterations × 10 games × 50 sims).

Usage:
    python smoke_train_alphazx.py
    python smoke_train_alphazx.py --iterations 8 --self-play-games 15
"""

import argparse
import logging
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from alphazx.diagram import METADATA, POSSIBLE_PHASES
from alphazx.models.homogeneous.alphazx_model import AlphaZXModel
from alphazx.mcts.config import MCTSConfig
from alphazx.mcts.trainer import Trainer, TrainerConfig
from alphazx.mcts.replay_buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-test: verify the full AlphaZero training loop is learning"
    )
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of self-play → train iterations (default: 5)")
    parser.add_argument("--self-play-games", type=int, default=10,
                        help="Games per iteration (default: 10)")
    parser.add_argument("--training-steps", type=int, default=200,
                        help="Gradient steps per iteration (default: 200)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size (default: 16)")
    parser.add_argument("--num-simulations", type=int, default=50,
                        help="MCTS sims per move (default: 50)")
    parser.add_argument("--num-qubits", type=int, default=5,
                        help="Qubits in generated circuits (default: 5)")
    parser.add_argument("--depth", type=int, default=5,
                        help="Circuit depth (default: 5)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Print losses every N training steps (default: 50)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (default: cpu)")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress per-step/per-game debug noise from the MCTS internals
    logging.getLogger("alphazx.mcts.search").setLevel(logging.WARNING)
    logging.getLogger("alphazx.mcts.self_play").setLevel(logging.WARNING)

    logger = logging.getLogger("smoke_test")

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ── Build model ──
    model = AlphaZXModel(
        num_node_types=len(METADATA.node_type_abbrevs),
        num_possible_phases=len(POSSIBLE_PHASES),
        num_possible_new_edges=5,
        node_embedding_channels=4,
        num_edge_embeddings=len(METADATA.edge_feat_to_index_dict),
        edge_embedding_channels=4,
        pe_in_channels=4,
        pe_out_channels=4,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # ── MCTS config — fast games ──
    mcts_config = MCTSConfig(
        num_simulations=args.num_simulations,
        c_puct=1.5,
        pw_alpha=0.5,
        temperature=1.0,
        pe_dim=4,
        num_qubits=args.num_qubits,
        depth=args.depth,
        max_episode_length=50,
        max_t_gate_increase=5,
        min_initial_t_gates=2,
        circuit_type="cnot_had_phase",
        p_had=0.2,
        p_t=0.4,
        leaf_batch_size=8,
    )

    # ── Trainer config ──
    trainer_config = TrainerConfig(
        num_self_play_games=args.self_play_games,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=1e-4,
        c_value=1.0,
        lr_schedule="constant",
        num_iterations=args.iterations,
        min_buffer_size=0,
        eval_interval=999,           # disable built-in evaluation
        checkpoint_interval=999,     # disable checkpointing
    )

    replay_buffer = ReplayBuffer(capacity=100_000)

    trainer = Trainer(
        model=model,
        mcts_config=mcts_config,
        trainer_config=trainer_config,
        replay_buffer=replay_buffer,
        device=device,
    )

    # ══════════════════════════════════════════════════════════════════
    # Run iterations: self-play → train → repeat
    # ══════════════════════════════════════════════════════════════════
    iteration_summaries = []
    total_start = time.time()

    for iteration in range(1, args.iterations + 1):
        logger.info(f"{'═' * 60}")
        logger.info(f"  ITERATION {iteration}/{args.iterations}")
        logger.info(f"{'═' * 60}")

        # ── Self-play ──
        model.eval()
        sp_start = time.time()
        results = trainer.self_play_manager.generate_games(args.self_play_games)
        sp_elapsed = time.time() - sp_start

        total_examples = sum(len(r.examples) for r in results)
        avg_steps = sum(r.num_steps for r in results) / max(1, len(results))
        avg_t_reduced = sum(r.t_gates_reduced for r in results) / max(1, len(results))
        avg_initial_t = sum(r.initial_t_gates for r in results) / max(1, len(results))
        avg_final_t = sum(r.final_t_gates for r in results) / max(1, len(results))
        num_simplified = sum(1 for r in results if r.simplified)
        num_increased = sum(1 for r in results if r.final_t_gates > r.initial_t_gates)

        logger.info(
            f"  Self-play: {len(results)} games, {total_examples} examples in {sp_elapsed:.1f}s"
        )
        logger.info(
            f"    avg_steps={avg_steps:.1f}, "
            f"avg_t_gates: {avg_initial_t:.1f} → {avg_final_t:.1f} "
            f"(reduced={avg_t_reduced:+.1f}), "
            f"simplified={num_simplified}, increased={num_increased}"
        )

        if total_examples < args.batch_size:
            logger.error(
                f"Only {total_examples} examples — less than batch_size "
                f"({args.batch_size}). Increase --self-play-games."
            )
            sys.exit(1)

        # ── Training ──
        model.train()
        train_start = time.time()

        policy_losses = []
        value_losses = []
        total_losses = []

        for step in range(1, args.training_steps + 1):
            metrics = trainer._train_step()
            policy_losses.append(metrics.policy_loss)
            value_losses.append(metrics.value_loss)
            total_losses.append(metrics.total_loss)

            if step % args.log_interval == 0:
                logger.info(
                    f"    train step {step:4d}/{args.training_steps}  |  "
                    f"policy={metrics.policy_loss:8.4f}  "
                    f"value={metrics.value_loss:8.4f}  "
                    f"total={metrics.total_loss:8.4f}"
                )

        train_elapsed = time.time() - train_start

        # Compute early vs late loss within this iteration
        window = max(1, args.training_steps // 5)
        early_total = sum(total_losses[:window]) / window
        late_total = sum(total_losses[-window:]) / window
        mean_policy = sum(policy_losses) / len(policy_losses)
        mean_value = sum(value_losses) / len(value_losses)

        logger.info(
            f"  Training: {args.training_steps} steps in {train_elapsed:.1f}s "
            f"({args.training_steps / train_elapsed:.1f} steps/s)"
        )
        logger.info(
            f"    avg policy={mean_policy:.4f}, avg value={mean_value:.4f}, "
            f"loss trend: {early_total:.4f} → {late_total:.4f} "
            f"({'↓' if late_total < early_total * 0.95 else '→'})"
        )

        iteration_summaries.append({
            "iteration": iteration,
            "num_examples": total_examples,
            "avg_steps": avg_steps,
            "avg_t_reduced": avg_t_reduced,
            "avg_initial_t": avg_initial_t,
            "avg_final_t": avg_final_t,
            "num_simplified": num_simplified,
            "num_increased": num_increased,
            "mean_policy_loss": mean_policy,
            "mean_value_loss": mean_value,
            "early_loss": early_total,
            "late_loss": late_total,
            "buffer_size": len(replay_buffer),
        })

    total_elapsed = time.time() - total_start

    # ══════════════════════════════════════════════════════════════════
    # Final summary — compare first iteration vs last
    # ══════════════════════════════════════════════════════════════════
    logger.info(f"\n{'═' * 60}")
    logger.info(f"  FINAL SUMMARY  ({total_elapsed:.0f}s total)")
    logger.info(f"{'═' * 60}")

    header = (
        f"{'iter':>4}  {'games':>5}  {'buf':>6}  "
        f"{'avg_steps':>9}  {'t_reduced':>9}  {'t_init→fin':>11}  "
        f"{'simpl':>5}  {'incr':>4}  "
        f"{'policy':>8}  {'value':>8}  {'loss_trend':>12}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    for s in iteration_summaries:
        trend = "↓" if s["late_loss"] < s["early_loss"] * 0.95 else "→"
        logger.info(
            f"{s['iteration']:4d}  "
            f"{s['num_examples']:5d}  "
            f"{s['buffer_size']:6d}  "
            f"{s['avg_steps']:9.1f}  "
            f"{s['avg_t_reduced']:+9.1f}  "
            f"{s['avg_initial_t']:4.1f} → {s['avg_final_t']:4.1f}  "
            f"{s['num_simplified']:5d}  "
            f"{s['num_increased']:4d}  "
            f"{s['mean_policy_loss']:8.4f}  "
            f"{s['mean_value_loss']:8.4f}  "
            f"{s['early_loss']:.3f}{trend}{s['late_loss']:.3f}"
        )

    first = iteration_summaries[0]
    last = iteration_summaries[-1]

    logger.info("")
    logger.info("  Cross-iteration trends (iteration 1 vs last):")
    logger.info(f"    T-gate reduction:  {first['avg_t_reduced']:+.1f} → {last['avg_t_reduced']:+.1f}  "
                f"{'IMPROVING' if last['avg_t_reduced'] > first['avg_t_reduced'] else 'NOT IMPROVING'}")
    logger.info(f"    Avg episode length: {first['avg_steps']:.1f} → {last['avg_steps']:.1f}")
    logger.info(f"    T-gate increases:   {first['num_increased']} → {last['num_increased']}  "
                f"{'IMPROVING' if last['num_increased'] < first['num_increased'] else 'NOT IMPROVING'}")
    logger.info(f"    Policy loss:        {first['mean_policy_loss']:.4f} → {last['mean_policy_loss']:.4f}")
    logger.info(f"    Value loss:         {first['mean_value_loss']:.4f} → {last['mean_value_loss']:.4f}")

    if last["avg_t_reduced"] > first["avg_t_reduced"] + 0.5:
        logger.info("\n  ✓ The model is improving at T-gate reduction across iterations.")
    elif last["mean_policy_loss"] < first["mean_policy_loss"] * 0.9:
        logger.info("\n  ~ Policy loss improved but self-play performance hasn't caught up yet.")
        logger.info("    This is normal — the model may need more iterations.")
    else:
        logger.info("\n  ? No clear improvement yet. This is expected with only "
                    f"{args.iterations} iterations on small circuits.")
        logger.info("    Try --iterations 10 or --self-play-games 20 for a stronger signal.")


if __name__ == "__main__":
    main()
