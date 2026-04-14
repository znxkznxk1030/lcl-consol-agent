"""
trainer.py
==========
MAPPO 학습 루프.

사용법
------
    from rl.trainer import MAPPOTrainer, TrainerConfig
    trainer = MAPPOTrainer(TrainerConfig(n_episodes=2000))
    trainer.train()

출력
----
  checkpoints/mappo_best.pt     — 최고 성능 체크포인트
  checkpoints/mappo_final.pt    — 학습 완료 체크포인트
  checkpoints/training_log.json — 에피소드별 지표 로그 (논문 그래프용)
"""

from __future__ import annotations

import os, json, time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

from simulator_v1.env import EnvConfig

from .env_wrapper import MultiAgentLCLEnv
from .mappo import MAPPO, MAPPOConfig
from .reward import compute_episode_return


@dataclass
class TrainerConfig:
    # 학습 에피소드 수
    n_episodes:      int   = 2000
    eval_every:      int   = 50      # N 에피소드마다 평가
    save_every:      int   = 200     # N 에피소드마다 체크포인트 저장
    log_every:       int   = 10      # N 에피소드마다 콘솔 출력

    # 환경
    env_seed:        int   = 42
    sim_duration:    int   = 72      # 시뮬레이션 시간 (h)

    # MAPPO
    mappo:           MAPPOConfig = field(default_factory=MAPPOConfig)

    # 저장 경로
    checkpoint_dir:  str  = "checkpoints"
    log_path:        str  = "checkpoints/training_log.json"

    # 디바이스
    device:          str  = "cpu"


class MAPPOTrainer:
    """
    Multi-Agent PPO 학습기.

    에피소드를 수행하면서 rollout을 수집하고,
    버퍼가 가득 차면 PPO 업데이트를 수행한다.
    """

    def __init__(self, cfg: TrainerConfig = TrainerConfig()):
        self.cfg = cfg
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        env_cfg = EnvConfig(
            seed=cfg.env_seed,
            sim_duration_hours=cfg.sim_duration,
        )
        self.env   = MultiAgentLCLEnv(config=env_cfg)
        self.mappo = MAPPO(cfg=cfg.mappo, device=cfg.device)

        self.episode_logs: List[dict] = []
        self.best_return  = -float("inf")

    # ── 메인 학습 루프 ────────────────────────────────────────────

    def train(self) -> None:
        print(f"[Trainer] Start MAPPO training  episodes={self.cfg.n_episodes}")
        t0 = time.time()

        for ep in range(1, self.cfg.n_episodes + 1):
            ep_log = self._run_episode(ep)
            self.episode_logs.append(ep_log)

            # 콘솔 출력
            if ep % self.cfg.log_every == 0:
                self._print_progress(ep, ep_log, time.time() - t0)

            # 정기 평가
            if ep % self.cfg.eval_every == 0:
                eval_ret = self._evaluate(n_episodes=5)
                print(f"  [Eval ep={ep}] avg_return={eval_ret:.4f}")
                if eval_ret > self.best_return:
                    self.best_return = eval_ret
                    self.mappo.save(os.path.join(self.cfg.checkpoint_dir, "mappo_best.pt"))

            # 정기 저장
            if ep % self.cfg.save_every == 0:
                self.mappo.save(
                    os.path.join(self.cfg.checkpoint_dir, f"mappo_ep{ep:05d}.pt")
                )

        # 최종 저장
        self.mappo.save(os.path.join(self.cfg.checkpoint_dir, "mappo_final.pt"))
        self._save_log()
        print(f"[Trainer] Done. best_return={self.best_return:.4f}  "
              f"elapsed={time.time()-t0:.1f}s")

    # ── 에피소드 실행 ─────────────────────────────────────────────

    def _run_episode(self, ep_idx: int) -> dict:
        obs_dict, _ = self.env.reset(seed=self.cfg.env_seed + ep_idx)
        ep_reward   = 0.0
        n_steps     = 0
        last_info   = {}

        while True:
            g, s, mask = obs_dict["timing"]

            # 행동 선택
            a_t, a_p, lp_t, lp_p, value = self.mappo.select_actions(g, s, mask)

            # 환경 step
            next_obs, rew_dict, done, _, info = self.env.step(
                {"timing": a_t, "planning": a_p}
            )
            reward = rew_dict["timing"]

            # 롤아웃 버퍼에 저장
            from .mappo import Transition
            self.mappo.buffer.add(Transition(
                global_feat=g, ship_feat=s, mask=mask,
                timing_act=a_t, plan_act=a_p,
                reward=reward, done=done,
                log_prob_t=lp_t, log_prob_p=lp_p,
                value=value,
            ))

            ep_reward += reward
            n_steps   += 1
            last_info  = info
            obs_dict   = next_obs

            # 롤아웃 가득 찬 경우 중간 업데이트
            if self.mappo.buffer.is_full():
                last_g, last_s, last_mask = obs_dict["timing"]
                _, _, _, _, last_val = self.mappo.select_actions(last_g, last_s, last_mask)
                self.mappo.update(last_value=last_val)

            if done:
                break

        # 에피소드 종료 후 남은 버퍼 업데이트
        if len(self.mappo.buffer) > 0:
            self.mappo.update(last_value=0.0)

        metrics = self.env.get_final_metrics()
        ep_return = compute_episode_return(metrics)

        return {
            "episode":      ep_idx,
            "ep_reward":    round(ep_reward, 4),
            "ep_return":    round(ep_return, 4),
            "n_steps":      n_steps,
            "fill_rate":    metrics.get("avg_fill_rate", 0),
            "sla_violation":metrics.get("sla_violation_rate", 0),
            "total_cost":   metrics.get("total_cost", 0),
            "n_mbls":       metrics.get("number_of_mbls", 0),
            "compat_viol":  metrics.get("compatibility_violations", 0),
        }

    # ── 평가 ─────────────────────────────────────────────────────

    def _evaluate(self, n_episodes: int = 5) -> float:
        """결정론적 행동으로 평가."""
        returns = []
        for i in range(n_episodes):
            obs_dict, _ = self.env.reset(seed=9000 + i)
            done = False
            while not done:
                g, s, mask = obs_dict["timing"]
                a_t, a_p, *_ = self.mappo.select_actions(g, s, mask, deterministic=True)
                obs_dict, _, done, _, _ = self.env.step({"timing": a_t, "planning": a_p})
            metrics = self.env.get_final_metrics()
            returns.append(compute_episode_return(metrics))
        return float(np.mean(returns))

    # ── 유틸리티 ─────────────────────────────────────────────────

    def _print_progress(self, ep: int, log: dict, elapsed: float) -> None:
        print(
            f"[ep {ep:5d}]  "
            f"ret={log['ep_return']:+.3f}  "
            f"fill={log['fill_rate']:.2%}  "
            f"sla_viol={log['sla_violation']:.2%}  "
            f"cost={log['total_cost']:.0f}  "
            f"mbls={log['n_mbls']}  "
            f"elapsed={elapsed:.0f}s"
        )

    def _save_log(self) -> None:
        with open(self.cfg.log_path, "w") as f:
            json.dump(self.episode_logs, f, indent=2)
        print(f"[Trainer] Log saved → {self.cfg.log_path}")
