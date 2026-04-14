"""
mappo.py
========
Multi-Agent PPO (MAPPO) 구현.

참고: Yu et al. "The Surprising Effectiveness of MAPPO in Cooperative,
      Multi-Agent Games." NeurIPS 2022.

구조:
  - TimingAgent   (actor_t) : Discrete(2)  — WAIT / DISPATCH
  - PlanningAgent (actor_p) : Discrete(5)  — fill-rate target
  - CentralCritic           : 공유 joint value 추정

학습:
  1. 환경에서 T-step rollout 수집
  2. GAE(λ) 이점 추정
  3. PPO-clip 업데이트 (K epochs)
  4. 두 actor + critic 동시 업데이트
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .network import Actor, CentralCritic
from .features import GLOBAL_DIM, PER_SHIP_DIM, MAX_SHIPMENTS, N_TIMING, N_PLAN


# ── 하이퍼파라미터 ────────────────────────────────────────────────

@dataclass
class MAPPOConfig:
    # PPO
    clip_eps:    float = 0.2
    entropy_coef: float = 0.01
    value_coef:  float  = 0.5
    max_grad_norm: float = 0.5

    # GAE
    gamma:  float = 0.99
    lam:    float = 0.95

    # 학습
    lr:          float = 3e-4
    n_epochs:    int   = 8        # PPO update epochs per rollout
    batch_size:  int   = 64
    hidden_dim:  int   = 128

    # 롤아웃
    rollout_steps: int = 512

    # 저장
    checkpoint_dir: str = "checkpoints"


# ── Rollout Buffer ────────────────────────────────────────────────

@dataclass
class Transition:
    global_feat: np.ndarray      # (GLOBAL_DIM,)
    ship_feat:   np.ndarray      # (MAX_SHIPMENTS, PER_SHIP_DIM)
    mask:        np.ndarray      # (MAX_SHIPMENTS,) bool
    timing_act:  int
    plan_act:    int
    reward:      float
    done:        bool
    log_prob_t:  float           # log π_timing(a_t | s)
    log_prob_p:  float           # log π_plan(a_p | s)
    value:       float           # V(s) from central critic


class RolloutBuffer:
    """고정 크기 롤아웃 버퍼."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.transitions: List[Transition] = []

    def add(self, t: Transition) -> None:
        self.transitions.append(t)

    def clear(self) -> None:
        self.transitions = []

    def __len__(self) -> int:
        return len(self.transitions)

    def is_full(self) -> bool:
        return len(self.transitions) >= self.capacity

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, lam: float
    ) -> Tuple[List[float], List[float]]:
        """GAE-λ 이점 및 λ-return 계산."""
        n = len(self.transitions)
        returns    = [0.0] * n
        advantages = [0.0] * n

        gae = 0.0
        for i in reversed(range(n)):
            t = self.transitions[i]
            next_val = last_value if i == n - 1 else self.transitions[i + 1].value
            if t.done:
                next_val = 0.0
            delta = t.reward + gamma * next_val - t.value
            gae   = delta + gamma * lam * (0.0 if t.done else gae)
            advantages[i] = gae
            returns[i]    = gae + t.value

        return returns, advantages


# ── MAPPO ─────────────────────────────────────────────────────────

class MAPPO:
    """
    Multi-Agent PPO for cooperative LCL consolidation.

    에이전트
    --------
    timing   : WAIT / DISPATCH
    planning : fill-rate target (0..4)
    """

    def __init__(self, cfg: MAPPOConfig = MAPPOConfig(), device: str = "cpu"):
        self.cfg    = cfg
        self.device = torch.device(device)

        # ── 네트워크 ────────────────────────────────────────────
        self.actor_t = Actor(
            global_dim=GLOBAL_DIM, hidden_dim=cfg.hidden_dim, action_dim=N_TIMING
        ).to(self.device)

        self.actor_p = Actor(
            global_dim=GLOBAL_DIM, hidden_dim=cfg.hidden_dim, action_dim=N_PLAN
        ).to(self.device)

        self.critic = CentralCritic(
            global_dim=GLOBAL_DIM, hidden_dim=cfg.hidden_dim
        ).to(self.device)

        # ── Optimizer ───────────────────────────────────────────
        self.opt = optim.Adam(
            list(self.actor_t.parameters()) +
            list(self.actor_p.parameters()) +
            list(self.critic.parameters()),
            lr=cfg.lr,
        )
        self.buffer = RolloutBuffer(cfg.rollout_steps)

        # 학습 통계
        self.train_steps = 0
        self.losses: List[dict] = []

    # ── 행동 선택 ────────────────────────────────────────────────

    @torch.no_grad()
    def select_actions(
        self,
        global_feat: np.ndarray,
        ship_feat:   np.ndarray,
        mask:        np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, int, float, float, float]:
        """
        Returns
        -------
        timing_act, plan_act, log_prob_t, log_prob_p, value
        """
        g = torch.FloatTensor(global_feat).unsqueeze(0).to(self.device)
        s = torch.FloatTensor(ship_feat).unsqueeze(0).to(self.device)
        m = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

        a_t, lp_t, _ = self.actor_t.get_action(g, s, m, deterministic)
        a_p, lp_p, _ = self.actor_p.get_action(g, s, m, deterministic)
        v             = self.critic(g, s, m).squeeze(-1)

        return (
            a_t.item(), a_p.item(),
            lp_t.item(), lp_p.item(),
            v.item(),
        )

    # ── PPO 업데이트 ─────────────────────────────────────────────

    def update(self, last_value: float = 0.0) -> dict:
        """
        롤아웃 버퍼 전체에 대해 PPO-clip 업데이트 수행.

        Returns
        -------
        loss_info : dict  (논문 Table 기록용)
        """
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.cfg.gamma, self.cfg.lam
        )
        # 이점 정규화
        adv = np.array(advantages)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # numpy 배치 구성
        N = len(self.buffer)
        g_arr  = np.stack([t.global_feat for t in self.buffer.transitions])
        s_arr  = np.stack([t.ship_feat   for t in self.buffer.transitions])
        m_arr  = np.stack([t.mask        for t in self.buffer.transitions])
        at_arr = np.array([t.timing_act  for t in self.buffer.transitions], np.int64)
        ap_arr = np.array([t.plan_act    for t in self.buffer.transitions], np.int64)
        lpt_arr= np.array([t.log_prob_t  for t in self.buffer.transitions], np.float32)
        lpp_arr= np.array([t.log_prob_p  for t in self.buffer.transitions], np.float32)
        ret_arr= np.array(returns,   dtype=np.float32)

        total_loss_sum = 0.0
        for _ in range(self.cfg.n_epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, self.cfg.batch_size):
                b = idx[start: start + self.cfg.batch_size]

                g = torch.FloatTensor(g_arr[b]).to(self.device)
                s = torch.FloatTensor(s_arr[b]).to(self.device)
                mk= torch.BoolTensor(m_arr[b]).to(self.device)
                at= torch.LongTensor(at_arr[b]).to(self.device)
                ap= torch.LongTensor(ap_arr[b]).to(self.device)
                old_lpt = torch.FloatTensor(lpt_arr[b]).to(self.device)
                old_lpp = torch.FloatTensor(lpp_arr[b]).to(self.device)
                ret = torch.FloatTensor(ret_arr[b]).to(self.device)
                adv_b = torch.FloatTensor(adv[b]).to(self.device)

                # ── Actor forward ──────────────────────────────
                logits_t = self.actor_t(g, s, mk)
                logits_p = self.actor_p(g, s, mk)
                dist_t   = Categorical(logits=logits_t)
                dist_p   = Categorical(logits=logits_p)
                new_lpt  = dist_t.log_prob(at)
                new_lpp  = dist_p.log_prob(ap)
                ent_t    = dist_t.entropy().mean()
                ent_p    = dist_p.entropy().mean()

                # ── Critic forward ─────────────────────────────
                values   = self.critic(g, s, mk).squeeze(-1)

                # ── PPO-clip loss ──────────────────────────────
                def ppo_loss(new_lp, old_lp):
                    ratio  = (new_lp - old_lp).exp()
                    surr1  = ratio * adv_b
                    surr2  = ratio.clamp(1 - self.cfg.clip_eps,
                                         1 + self.cfg.clip_eps) * adv_b
                    return -torch.min(surr1, surr2).mean()

                loss_t = ppo_loss(new_lpt, old_lpt)
                loss_p = ppo_loss(new_lpp, old_lpp)

                # ── Value loss ─────────────────────────────────
                loss_v = F.mse_loss(values, ret)

                # ── Total loss ─────────────────────────────────
                loss = (loss_t + loss_p
                        + self.cfg.value_coef * loss_v
                        - self.cfg.entropy_coef * (ent_t + ent_p))

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor_t.parameters()) +
                    list(self.actor_p.parameters()) +
                    list(self.critic.parameters()),
                    self.cfg.max_grad_norm,
                )
                self.opt.step()
                total_loss_sum += loss.item()

        self.buffer.clear()
        self.train_steps += 1
        loss_info = {
            "loss_t": loss_t.item(),
            "loss_p": loss_p.item(),
            "loss_v": loss_v.item(),
            "entropy_t": ent_t.item(),
            "entropy_p": ent_p.item(),
        }
        self.losses.append(loss_info)
        return loss_info

    # ── 저장 / 로드 ──────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "actor_t":     self.actor_t.state_dict(),
            "actor_p":     self.actor_p.state_dict(),
            "critic":      self.critic.state_dict(),
            "optimizer":   self.opt.state_dict(),
            "train_steps": self.train_steps,
            "cfg":         self.cfg,
        }, path)
        print(f"[MAPPO] Saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor_t.load_state_dict(ckpt["actor_t"])
        self.actor_p.load_state_dict(ckpt["actor_p"])
        self.critic.load_state_dict(ckpt["critic"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.train_steps = ckpt.get("train_steps", 0)
        print(f"[MAPPO] Loaded ← {path}  (step {self.train_steps})")
