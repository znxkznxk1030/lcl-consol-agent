"""
rl_agent.py
===========
학습된 MAPPO 모델을 기존 AgentBase 인터페이스로 래핑.

기존 agent.act(obs_dict) → action_dict 인터페이스와 완전 호환됨.
따라서 ConsolidationEnv.run(rl_agent) 으로 그대로 사용 가능.

사용법
------
    from rl.rl_agent import RLConsolidationAgent
    agent = RLConsolidationAgent("checkpoints/mappo_best.pt")
    result = env.run(agent)
"""

from __future__ import annotations

import torch
from typing import Optional

from simulator_v1.agents.base import AgentBase
from simulator_v1.volume_model import usable_container_cbm

from .mappo import MAPPO, MAPPOConfig
from .features import encode_observation, GLOBAL_DIM
from .env_wrapper import FILL_TARGETS, _PlanHelper
from simulator_v1.env import EnvConfig


class RLConsolidationAgent(AgentBase):
    """
    MAPPO 기반 LCL 통합 최적화 에이전트.

    act() 는 기존 AgentBase 인터페이스를 따른다.

    Parameters
    ----------
    checkpoint_path : str
        학습된 모델 체크포인트 경로 (.pt)
    deterministic : bool
        True = argmax 행동 (평가용), False = stochastic (탐색용)
    device : str
        "cpu" | "cuda"
    """

    agent_id = "mappo_rl"

    def __init__(
        self,
        checkpoint_path: str,
        cfg: Optional[EnvConfig] = None,
        deterministic: bool = True,
        device: str = "cpu",
    ):
        self.deterministic = deterministic
        self.device = device

        self.mappo = MAPPO(cfg=MAPPOConfig(), device=device)
        self.mappo.load(checkpoint_path)
        self.mappo.actor_t.eval()
        self.mappo.actor_p.eval()

        env_cfg = cfg or EnvConfig()
        self._helper = _PlanHelper(env_cfg)
        self._usable_cbm = usable_container_cbm(env_cfg.max_cbm_per_mbl)

        # 통계 (디버깅용)
        self._n_dispatch = 0
        self._n_wait     = 0

    # ── AgentBase 인터페이스 ──────────────────────────────────────

    def act(self, observation: dict) -> dict:
        """
        Parameters
        ----------
        observation : dict — ConsolidationEnv가 생성한 Observation.to_dict()

        Returns
        -------
        action_dict : DISPATCH / WAIT action dict
        """
        g, s_feat, mask = encode_observation(observation)

        a_t, a_p, *_ = self.mappo.select_actions(
            g, s_feat, mask, deterministic=self.deterministic
        )

        if a_t == 0:   # WAIT
            self._n_wait += 1
            return self._make_action("WAIT", [],
                                     f"rl_wait(fill_target={FILL_TARGETS[a_p]:.0%})")

        # DISPATCH
        self._n_dispatch += 1
        fill_target = FILL_TARGETS[a_p]
        ships  = observation["buffer"]["shipments"]
        if not ships:
            return self._make_action("WAIT", [], "rl_dispatch_empty_buffer")

        groups = self._helper.bin_pack(ships, self._usable_cbm)

        # fill_target 이상이거나 SLA CRITICAL 화물 포함 그룹만 dispatch
        critical_ids = {s["shipment_id"] for s in ships if s["time_to_due"] < 6}
        to_dispatch = []
        for g_ids in groups:
            g_set  = set(g_ids)
            g_ecbm = sum(s["effective_cbm"] for s in ships if s["shipment_id"] in g_set)
            fill   = g_ecbm / self._usable_cbm if self._usable_cbm > 0 else 0.0
            if fill >= fill_target or bool(g_set & critical_ids):
                to_dispatch.append(g_ids)

        if not to_dispatch:
            return self._make_action("WAIT", [],
                                     f"rl_below_fill_target({fill_target:.0%})")

        plans = self._make_plans(to_dispatch, ships, self._usable_cbm)
        reason = (f"rl_dispatch fill_target={fill_target:.0%}  "
                  f"groups={len(plans)}")
        return self._make_action("DISPATCH", plans, reason)
