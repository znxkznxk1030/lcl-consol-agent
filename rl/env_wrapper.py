"""
env_wrapper.py
==============
ConsolidationEnv 위에 step() 인터페이스를 추가하는 Gymnasium 호환 래퍼.

Multi-Agent 관점:
  TimingAgent   action ∈ {0=WAIT, 1=DISPATCH}
  PlanningAgent action ∈ {0..4}  →  fill-rate targets [0.40, 0.50, 0.60, 0.70, 0.80]

step(actions) 는 두 에이전트의 joint action dict를 받는다.
  actions = {"timing": int, "planning": int}

관찰은 두 에이전트 모두 동일한 글로벌 observation을 공유한다.
(Centralized observation, cooperative setting)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import numpy as np

from simulator_v1.env import ConsolidationEnv, EnvConfig
from simulator_v1.agents.base import AgentBase
from simulator_v1.volume_model import usable_container_cbm

from .features import encode_observation, GLOBAL_DIM, PER_SHIP_DIM, MAX_SHIPMENTS
from .reward import StepInfo, compute_reward

# ── 행동 공간 정의 ────────────────────────────────────────────────
TIMING_ACTIONS  = ["WAIT", "DISPATCH"]
FILL_TARGETS    = [0.40, 0.50, 0.60, 0.70, 0.80]   # PlanningAgent 행동 집합

N_TIMING  = len(TIMING_ACTIONS)    # 2
N_PLAN    = len(FILL_TARGETS)      # 5


class MultiAgentLCLEnv:
    """
    LCL 통합 최적화를 위한 Multi-Agent 환경.

    에이전트
    --------
    timing   : WAIT(0) / DISPATCH(1) 결정
    planning : fill-rate target 인덱스 결정 (dispatch 시에만 유효)

    관찰 공간  (두 에이전트 공유)
    ---------
    global_feat  : (GLOBAL_DIM,)
    ship_feat    : (MAX_SHIPMENTS, PER_SHIP_DIM)
    mask         : (MAX_SHIPMENTS,)   True = padding

    보상
    ----
    공유 scalar reward  (cooperative)
    """

    AGENTS = ["timing", "planning"]

    def __init__(self, config: Optional[EnvConfig] = None, seed: int = 42):
        self.cfg    = config or EnvConfig(seed=seed)
        self._inner = ConsolidationEnv(self.cfg)
        self._helper = _PlanHelper(self.cfg)

        self._last_obs: Optional[dict] = None
        self._prev_info: Optional[StepInfo] = None
        self._done = False

        # 에피소드 통계
        self._ep_compat_violations = 0
        self._ep_late_count = 0

    # ── Gym 인터페이스 ────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, dict]:
        """환경 초기화. 첫 번째 observation 반환."""
        if seed is not None:
            self.cfg.seed = seed
        self._inner = ConsolidationEnv(self.cfg)
        self._inner._reset()

        # 첫 tick: 화물 생성 후 첫 관찰
        self._advance_arrivals()
        self._last_obs = self._inner._build_observation().to_dict()
        self._done = False
        self._ep_compat_violations = 0
        self._ep_late_count = 0

        self._prev_info = self._make_step_info(dispatched_count=0,
                                                fill_rates=[], late_delta=0, compat_delta=0)
        obs_dict = self._build_ma_obs()
        return obs_dict, {}

    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict[str, float], bool, bool, dict]:
        """
        Parameters
        ----------
        actions : {"timing": int, "planning": int}

        Returns
        -------
        obs     : {"timing": ..., "planning": ...}
        rewards : {"timing": float, "planning": float}
        dones   : bool  (episode terminated)
        truncated : bool
        info    : dict
        """
        assert not self._done, "Call reset() before step()."

        timing_act  = int(actions.get("timing",  0))
        plan_act    = int(actions.get("planning", 2))   # default: fill=0.60

        # ── Action 실행 ───────────────────────────────────────────
        dispatch_count = 0
        fill_rates_this = []
        late_this = 0
        compat_this = 0

        if timing_act == 1:   # DISPATCH
            fill_target = FILL_TARGETS[plan_act]
            plans, n_dispatched, fills, late, compat = self._execute_dispatch(fill_target)
            dispatch_count   = n_dispatched
            fill_rates_this  = fills
            late_this        = late
            compat_this      = compat
            self._ep_late_count        += late
            self._ep_compat_violations += compat

        # ── 시간 진행 ─────────────────────────────────────────────
        curr_info = self._make_step_info(dispatch_count, fill_rates_this,
                                         late_this, compat_this)
        reward = compute_reward(self._prev_info, curr_info)

        self._inner.current_time += 1.0
        if self._inner.current_time >= self._inner.next_cutoff:
            self._inner.next_cutoff += self.cfg.cutoff_interval_hours

        # 화물 도착
        self._advance_arrivals()

        # ── 종료 조건 ─────────────────────────────────────────────
        self._done = self._inner.current_time >= self.cfg.sim_duration_hours
        if self._done:
            # 잔여 화물 강제 출고
            remaining = self._inner.buffer.ids()
            if remaining:
                self._inner._dispatch([remaining])

        self._last_obs = self._inner._build_observation().to_dict()
        self._prev_info = curr_info

        obs_dict     = self._build_ma_obs()
        reward_dict  = {"timing": reward, "planning": reward}   # shared
        info         = self._build_info()

        return obs_dict, reward_dict, self._done, False, info

    # ── 내부 헬퍼 ─────────────────────────────────────────────────

    def _advance_arrivals(self) -> None:
        """현재 tick의 화물 도착 처리."""
        t = self._inner.current_time
        new_ships = self._inner._generate_arrivals(t)
        for s in new_ships:
            self._inner.buffer.add(s)
            self._inner.all_shipments.append(s)

    def _execute_dispatch(
        self, fill_target: float
    ) -> Tuple[list, int, List[float], int, int]:
        """
        fill_target 기준으로 compatible bin packing 수행 후 dispatch.

        Returns
        -------
        plans, n_dispatched, fill_rates, late_count, compat_violations
        """
        ships = self._inner._build_observation().to_dict()["buffer"]["shipments"]
        if not ships:
            return [], 0, [], 0, 0

        usable = usable_container_cbm(self.cfg.max_cbm_per_mbl)
        groups = self._helper.bin_pack(ships, usable)

        # fill_target 이상인 그룹만 dispatch (SLA CRITICAL은 예외)
        critical_ids = {s["shipment_id"] for s in ships if s["time_to_due"] < 6}
        to_dispatch = []
        for g in groups:
            g_ecbm = sum(s["effective_cbm"]
                         for s in ships if s["shipment_id"] in set(g))
            fill = g_ecbm / usable if usable > 0 else 0.0
            has_critical = bool(set(g) & critical_ids)
            if fill >= fill_target or has_critical:
                to_dispatch.append(g)

        if not to_dispatch:
            return [], 0, [], 0, 0

        n_before_mbls   = len(self._inner.mbls)
        n_before_late   = self._ep_late_count
        n_before_compat = self._inner._compatibility_violations

        self._inner._dispatch(to_dispatch)

        new_mbls    = self._inner.mbls[n_before_mbls:]
        fill_rates  = []
        if new_mbls:
            usable_cbm = usable_container_cbm(self.cfg.max_cbm_per_mbl)
            for m in new_mbls:
                if usable_cbm > 0:
                    fill_rates.append(m.total_effective_cbm / usable_cbm)

        n_dispatched   = sum(len(m.shipment_ids) for m in new_mbls)
        late_delta     = sum(
            1 for s in self._inner.all_shipments
            if s.dispatched and s.is_late()
        ) - n_before_late
        compat_delta   = self._inner._compatibility_violations - n_before_compat

        return to_dispatch, n_dispatched, fill_rates, max(0, late_delta), compat_delta

    def _make_step_info(self, dispatched_count, fill_rates, late_delta, compat_delta) -> StepInfo:
        return StepInfo(
            time=self._inner.current_time,
            buffer_count=self._inner.buffer.count,
            total_effective_cbm=self._inner.buffer.total_effective_cbm,
            mbls_created=len(fill_rates),
            late_shipments=late_delta,
            fill_rates=fill_rates,
            compat_violations=compat_delta,
            holding_cost_per_hour=self.cfg.holding_cost_per_hour,
            late_penalty=self.cfg.late_penalty,
            dispatched_count=dispatched_count,
        )

    def _build_ma_obs(self) -> Dict[str, Tuple]:
        """두 에이전트가 공유하는 관찰 반환."""
        g, s, mask = encode_observation(self._last_obs)
        obs = (g, s, mask)
        return {"timing": obs, "planning": obs}

    def _build_info(self) -> dict:
        return {
            "current_time":          self._inner.current_time,
            "buffer_count":          self._inner.buffer.count,
            "buffer_cbm":            self._inner.buffer.total_effective_cbm,
            "ep_late":               self._ep_late_count,
            "ep_compat_violations":  self._ep_compat_violations,
            "n_mbls":                len(self._inner.mbls),
        }

    def get_final_metrics(self) -> dict:
        """에피소드 종료 후 최종 metrics 반환."""
        result = self._inner._build_result(_DummyAgent())
        return {k: v for k, v in vars(result.metrics).items()}


# ── 단일 에이전트 래퍼 ─────────────────────────────────────────────

class SingleAgentLCLEnv:
    """
    단일 에이전트 baseline 비교용 래퍼.
    action = [timing_int, planning_int]  (2원소 배열)
    """

    def __init__(self, config: Optional[EnvConfig] = None, seed: int = 42):
        self._ma = MultiAgentLCLEnv(config, seed)

    def reset(self, seed=None):
        obs_dict, info = self._ma.reset(seed)
        return self._flatten(obs_dict["timing"]), info

    def step(self, action):
        timing  = int(action[0])
        planning = int(action[1]) if len(action) > 1 else 2
        obs_d, rew_d, done, trunc, info = self._ma.step(
            {"timing": timing, "planning": planning}
        )
        return self._flatten(obs_d["timing"]), rew_d["timing"], done, trunc, info

    @staticmethod
    def _flatten(obs_tuple) -> np.ndarray:
        g, s, _ = obs_tuple
        return np.concatenate([g, s.flatten()])

    def get_final_metrics(self):
        return self._ma.get_final_metrics()


# ── 보조 클래스 ───────────────────────────────────────────────────

class _PlanHelper(AgentBase):
    """compatible_bin_pack을 호출하기 위한 최소 AgentBase 서브클래스."""
    agent_id = "_plan_helper"

    def __init__(self, cfg: EnvConfig):
        self._cfg = cfg

    def act(self, observation: dict) -> dict:
        raise NotImplementedError

    def bin_pack(self, ships: list, usable: float) -> List[List[str]]:
        return self._compatible_bin_pack(ships, usable)


class _DummyAgent:
    agent_id = "rl_agent"
