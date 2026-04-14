"""
train.py
========
MAPPO 학습 CLI 진입점.

사용법
------
    # 기본 학습 (2000 에피소드)
    python -m rl.train

    # 설정 변경
    python -m rl.train --episodes 5000 --lr 1e-4 --rollout 512 --device cuda

    # 체크포인트에서 이어서 학습
    python -m rl.train --resume checkpoints/mappo_ep01000.pt
"""

from __future__ import annotations

import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from .trainer import MAPPOTrainer, TrainerConfig
from .mappo import MAPPOConfig


def main():
    parser = argparse.ArgumentParser(description="Train MAPPO for LCL Consolidation")
    parser.add_argument("--episodes",   type=int,   default=2000)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--rollout",    type=int,   default=512,
                        help="Steps per rollout buffer")
    parser.add_argument("--epochs",     type=int,   default=8,
                        help="PPO update epochs per rollout")
    parser.add_argument("--hidden",     type=int,   default=128)
    parser.add_argument("--device",     type=str,   default="cpu")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--sim_hours",  type=int,   default=72)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Checkpoint path to resume from")
    args = parser.parse_args()

    mappo_cfg = MAPPOConfig(
        lr=args.lr,
        rollout_steps=args.rollout,
        n_epochs=args.epochs,
        hidden_dim=args.hidden,
    )
    trainer_cfg = TrainerConfig(
        n_episodes=args.episodes,
        env_seed=args.seed,
        sim_duration=args.sim_hours,
        mappo=mappo_cfg,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    trainer = MAPPOTrainer(trainer_cfg)

    if args.resume:
        print(f"[Train] Resuming from {args.resume}")
        trainer.mappo.load(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
