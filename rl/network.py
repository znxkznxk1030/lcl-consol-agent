"""
network.py
==========
신경망 구성 요소.

ShipmentSetEncoder  : Multi-head Self-Attention 기반 가변 길이 화물 집합 인코더
Actor               : 정책 네트워크 (logits)
CentralCritic       : MAPPO용 중앙 critic  (joint state 입력)
ActorCritic         : 단일 에이전트용 Actor-Critic

가변 길이 화물 집합에 Attention을 쓰는 것은 학술적으로 의미 있는 기여
(집합 순서에 불변하며, 화물 수에 robust).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .features import GLOBAL_DIM, PER_SHIP_DIM, MAX_SHIPMENTS


# ── 하이퍼파라미터 기본값 ─────────────────────────────────────────
HIDDEN_DIM  = 128
ATTN_HEADS  = 4
ATTN_LAYERS = 2


class ShipmentSetEncoder(nn.Module):
    """
    Multi-head Self-Attention을 이용한 화물 집합 인코더.

    입력 : (batch, MAX_SHIPMENTS, PER_SHIP_DIM) + bool mask (True=padding)
    출력 : (batch, hidden_dim)  — 순서 불변(permutation-invariant) 표현

    논문 참고: Lee et al. "Set Transformer" (ICML 2019)
    """

    def __init__(self, input_dim: int = PER_SHIP_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 n_heads: int = ATTN_HEADS,
                 n_layers: int = ATTN_LAYERS):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in n_layers * [None]])
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, N, PER_SHIP_DIM)
        mask : (B, N)  — True = padding slot

        Returns
        -------
        (B, hidden_dim)
        """
        h = self.projection(x)                       # (B, N, H)

        for attn, norm in zip(self.layers, self.norms):
            attn_out, _ = attn(h, h, h, key_padding_mask=mask)
            h = norm(h + attn_out)

        # 패딩을 제외한 mean pooling
        valid = (~mask).unsqueeze(-1).float()        # (B, N, 1)
        count = valid.sum(1).clamp(min=1)            # (B, 1)
        pooled = (h * valid).sum(1) / count          # (B, H)

        return self.norm_out(self.ffn(pooled) + pooled)


class Actor(nn.Module):
    """정책 네트워크: 상태 → action logits."""

    def __init__(self, global_dim: int = GLOBAL_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 action_dim: int = 2):
        super().__init__()
        self.set_encoder = ShipmentSetEncoder(hidden_dim=hidden_dim)
        joint_dim = global_dim + hidden_dim

        self.net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self,
                global_feat: torch.Tensor,
                ship_feat:   torch.Tensor,
                mask:        torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        logits : (B, action_dim)
        """
        ship_enc = self.set_encoder(ship_feat, mask)        # (B, H)
        joint    = torch.cat([global_feat, ship_enc], dim=-1)  # (B, G+H)
        return self.net(joint)

    def get_action(self, global_feat, ship_feat, mask, deterministic=False):
        """추론용: action + log_prob 반환."""
        logits = self.forward(global_feat, ship_feat, mask)
        dist   = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class CentralCritic(nn.Module):
    """
    MAPPO 중앙 Critic.
    두 에이전트의 joint 관찰을 입력받아 joint value를 추정.

    입력: global + set_encoding  (두 에이전트 동일 obs 공유이므로 한 번만 인코딩)
    """

    def __init__(self, global_dim: int = GLOBAL_DIM,
                 hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.set_encoder = ShipmentSetEncoder(hidden_dim=hidden_dim)
        joint_dim = global_dim + hidden_dim

        self.net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self,
                global_feat: torch.Tensor,
                ship_feat:   torch.Tensor,
                mask:        torch.Tensor) -> torch.Tensor:
        """Returns (B, 1) value estimate."""
        ship_enc = self.set_encoder(ship_feat, mask)
        joint    = torch.cat([global_feat, ship_enc], dim=-1)
        return self.net(joint)


class ActorCritic(nn.Module):
    """단일 에이전트 Actor-Critic (PPO baseline 비교용)."""

    def __init__(self, global_dim=GLOBAL_DIM, hidden_dim=HIDDEN_DIM,
                 timing_dim=2, planning_dim=5):
        super().__init__()
        self.set_encoder = ShipmentSetEncoder(hidden_dim=hidden_dim)
        joint_dim = global_dim + hidden_dim

        # 두 개의 head: timing (2) + planning (5)
        shared = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.shared  = shared
        self.timing_head   = nn.Linear(hidden_dim // 2, timing_dim)
        self.planning_head = nn.Linear(hidden_dim // 2, planning_dim)
        self.value_head    = nn.Linear(hidden_dim // 2, 1)

    def forward(self, global_feat, ship_feat, mask):
        ship_enc = self.set_encoder(ship_feat, mask)
        joint    = torch.cat([global_feat, ship_enc], dim=-1)
        h        = self.shared(joint)
        return (self.timing_head(h),
                self.planning_head(h),
                self.value_head(h))
