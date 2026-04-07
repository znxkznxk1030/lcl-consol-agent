"""
env.py
======
ConsolidationEnv: 시뮬레이터 코어
- JSON 인터페이스 (Observation / Action)
- Event log 생성 (시각화용)
- Agent와 완전히 분리됨
"""

from __future__ import annotations

import uuid
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .entities import (
    ItemType, Shipment, MBL,
    generate_shipment, create_mbl,
)
from .distributions import thinning_arrivals
from .compatibility import split_into_compatible_groups, count_violation_pairs
from .buffer import WarehouseBuffer
from .cost import CostEngine
from .schemas import (
    Observation, ConfigObservation, BufferObservation, ShipmentObservation,
    Action, Event, Metrics, SimulationResult,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    seed: int = 42
    sim_duration_hours: int = 72
    cutoff_interval_hours: int = 24
    max_cbm_per_mbl: float = 33.2  # 20ft 컨테이너 기본 (590×235×239cm)
    destination: str = "PORT_A"

    # ItemType별 시간당 평균 도착 화물 수 (기본값: 실측 통계 기반)
    arrival_rates: Dict[str, float] = field(default_factory=lambda: {
        "ELECTRONICS":   2.0,
        "CLOTHING":      1.5,
        "COSMETICS":     0.8,
        "FOOD_PRODUCTS": 0.7,
        "AUTO_PARTS":    0.5,
        "CHEMICALS":     0.3,
        "FURNITURE":     0.2,
        "MACHINERY":     0.15,
    })

    use_real_distributions: bool = True  # False 시 기존 균등분포 동작 유지

    fixed_cost_per_mbl: float = 100.0
    variable_cost_per_cbm: float = 10.0
    holding_cost_per_hour: float = 2.0
    late_penalty: float = 50.0

    sla_hours: float = 48.0
    _olist_dimension_samples: Optional[Dict[str, List[tuple[float, float, float]]]] = field(
        default=None,
        repr=False,
    )

    def to_dict(self) -> dict:
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ConsolidationEnv:
    def __init__(self, config: EnvConfig = EnvConfig()) -> None:
        self.cfg = config
        self.rng = random.Random(config.seed)

        self.buffer = WarehouseBuffer()
        self.cost_engine = CostEngine(
            fixed_cost_per_mbl=config.fixed_cost_per_mbl,
            variable_cost_per_cbm=config.variable_cost_per_cbm,
            holding_cost_per_hour=config.holding_cost_per_hour,
            late_penalty=config.late_penalty,
        )

        self.current_time: float = 0.0
        self.next_cutoff: float = config.cutoff_interval_hours

        self.all_shipments: List[Shipment] = []
        self.mbls: List[MBL] = []
        self.events: List[Event] = []
        self._event_counter: int = 0

        # 호환성 위반 카운터
        self._compatibility_violations: int = 0
        self._compatibility_extra_mbls: int = 0

    # ------------------------------------------------------------------
    # Public: run with agent
    # ------------------------------------------------------------------

    def run(self, agent) -> SimulationResult:
        """agent.act(observation: dict) -> dict 인터페이스를 따르는 agent 실행"""
        self._reset()

        while self.current_time < self.cfg.sim_duration_hours:
            self._tick_event()

            # 1. 화물 도착
            new_shipments = self._generate_arrivals(self.current_time)
            for s in new_shipments:
                self.buffer.add(s)
                self.all_shipments.append(s)
                self._log_event("SHIPMENT_ARRIVAL", {
                    "shipment_id": s.shipment_id,
                    "item_type": s.item_type.value,
                    "cargo_category": s.cargo_category.value,
                    "cbm": s.cbm,
                    "effective_cbm": s.effective_cbm,
                    "weight": s.weight,
                    "due_time": s.due_time,
                    "length_cm": s.length_cm,
                    "height_cm": s.height_cm,
                    "width_cm": s.width_cm,
                })

            # 2. Observation 생성 → Agent에 전달
            obs = self._build_observation()
            action_dict = agent.act(obs.to_dict())
            action = self._parse_action(action_dict)

            # 3. Agent 결정 로깅
            self._log_event("AGENT_DECISION", {
                "agent_id": action.agent_id,
                "action": action.action,
                "mbl_count": len(action.mbls),
                "reason": action.reason,
                "buffer_cbm": self.buffer.total_cbm,
                "buffer_effective_cbm": self.buffer.total_effective_cbm,
                "buffer_count": self.buffer.count,
            })

            # 4. Action 실행
            if action.action == "DISPATCH" and action.mbls:
                self._dispatch(action.mbls)

            # 5. Cutoff 갱신
            if self.current_time >= self.next_cutoff:
                self.next_cutoff += self.cfg.cutoff_interval_hours

            self.current_time += 1.0

        # 잔여 화물 강제 출고 (자동 CBM 분할)
        remaining = self.buffer.ids()
        if remaining:
            self._dispatch([remaining])

        return self._build_result(agent)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        shipments = self.buffer.all()
        return Observation(
            schema=Observation.version(),
            current_time=self.current_time,
            time_to_cutoff=self.next_cutoff - self.current_time,
            config=ConfigObservation(
                max_cbm_per_mbl=self.cfg.max_cbm_per_mbl,
                sla_hours=self.cfg.sla_hours,
            ),
            buffer=BufferObservation(
                count=self.buffer.count,
                total_cbm=self.buffer.total_cbm,
                total_effective_cbm=self.buffer.total_effective_cbm,
                total_weight=self.buffer.total_weight,
                shipments=[
                    ShipmentObservation(
                        shipment_id=s.shipment_id,
                        item_type=s.item_type.value,
                        cargo_category=s.cargo_category.value,
                        arrival_time=s.arrival_time,
                        waiting_time=round(self.current_time - s.arrival_time, 2),
                        cbm=s.cbm,
                        effective_cbm=s.effective_cbm,
                        weight=s.weight,
                        packages=s.packages,
                        due_time=s.due_time,
                        time_to_due=round(s.due_time - self.current_time, 2),
                        length_cm=s.length_cm,
                        height_cm=s.height_cm,
                        width_cm=s.width_cm,
                    )
                    for s in shipments
                ],
            ),
        )

    # ------------------------------------------------------------------
    # Action parser & executor
    # ------------------------------------------------------------------

    def _parse_action(self, action_dict: dict) -> Action:
        return Action(
            schema=action_dict.get("schema", "action/v1"),
            agent_id=action_dict.get("agent_id", "unknown"),
            action=action_dict.get("action", "WAIT"),
            mbls=action_dict.get("mbls", []),
            reason=action_dict.get("reason"),
        )

    def _dispatch(self, mbl_assignments: List[List[str]]) -> None:
        """mbl_assignments: 각 inner list = 하나의 MBL에 담을 shipment ID 목록."""
        # 전체 ID 한 번에 버퍼에서 제거
        all_ids = [sid for group in mbl_assignments for sid in group]
        all_ships = {s.shipment_id: s for s in self.buffer.remove(all_ids)}

        for group_ids in mbl_assignments:
            candidates = [all_ships[sid] for sid in group_ids if sid in all_ships]
            if not candidates:
                continue

            # 호환성 검사 → 위반 시 자동 분리
            violation_pairs = count_violation_pairs(candidates)
            if violation_pairs > 0:
                self._compatibility_violations += 1
                compat_groups = split_into_compatible_groups(candidates)
                extra = len(compat_groups) - 1
                self._compatibility_extra_mbls += extra

                self._log_event("COMPATIBILITY_VIOLATION", {
                    "original_count": len(candidates),
                    "violation_pairs": violation_pairs,
                    "split_into_groups": len(compat_groups),
                    "extra_mbls_created": extra,
                    "categories": [s.cargo_category.value for s in candidates],
                })
            else:
                compat_groups = [candidates]

            # 각 호환 그룹을 CBM 한도 내로 추가 분할 후 MBL 생성
            for group in compat_groups:
                for cbm_group in self._split_by_cbm(group):
                    self._create_mbl(cbm_group)

    def _split_by_cbm(self, shipments: List[Shipment]) -> List[List[Shipment]]:
        """effective_cbm 기준으로 max_cbm_per_mbl을 초과하지 않도록 분할."""
        result: List[List[Shipment]] = []
        current: List[Shipment] = []
        current_cbm = 0.0

        for s in shipments:
            ecbm = s.effective_cbm
            # 단일 화물 자체가 한도 초과인 경우: 단독 MBL로 처리
            if current and current_cbm + ecbm > self.cfg.max_cbm_per_mbl:
                result.append(current)
                current = [s]
                current_cbm = ecbm
            else:
                current.append(s)
                current_cbm += ecbm

        if current:
            result.append(current)

        return result or [[]]

    def _create_mbl(self, shipments: List[Shipment]) -> None:
        for s in shipments:
            s.dispatched = True
            s.dispatch_time = self.current_time

        mbl = create_mbl(shipments, self.current_time)
        self.mbls.append(mbl)

        # SLA violation 체크
        for s in shipments:
            if s.is_late():
                self._log_event("SLA_VIOLATION", {
                    "shipment_id": s.shipment_id,
                    "due_time": s.due_time,
                    "dispatch_time": s.dispatch_time,
                    "overdue_hours": round(s.dispatch_time - s.due_time, 2),
                })

        self._log_event("DISPATCH", {
            "shipment_ids": [s.shipment_id for s in shipments],
            "count": len(shipments),
        })
        self._log_event("MBL_CREATED", {
            "mbl_id": mbl.mbl_id,
            "shipment_count": len(mbl.shipment_ids),
            "total_cbm": mbl.total_cbm,
            "total_effective_cbm": mbl.total_effective_cbm,
            "total_weight": mbl.total_weight,
            "fill_rate": round(mbl.total_cbm / self.cfg.max_cbm_per_mbl, 4),
            "effective_fill_rate": round(mbl.total_effective_cbm / self.cfg.max_cbm_per_mbl, 4),
            "categories": list({h.cargo_category for h in mbl.hbls}),
            "hbl_ids": [h.hbl_id for h in mbl.hbls],
        })

    # ------------------------------------------------------------------
    # Arrival generator (Poisson)
    # ------------------------------------------------------------------

    def _generate_arrivals(self, hour: float) -> List[Shipment]:
        arrivals = []
        for item_type in ItemType:
            rate = self.cfg.arrival_rates.get(item_type.value, 0.0)
            if rate <= 0:
                continue

            if self.cfg.use_real_distributions:
                offsets = thinning_arrivals(self.rng, rate, hour)
            else:
                offsets = []
                t = 0.0
                while True:
                    t += self.rng.expovariate(rate)
                    if t > 1.0:
                        break
                    offsets.append(t)

            for t in offsets:
                dimensions_cm = self._sample_dimensions(item_type.value)
                s = generate_shipment(
                    self.rng, hour + t, item_type,
                    destination=self.cfg.destination,
                    sla_hours=self.cfg.sla_hours,
                    dimensions_cm=dimensions_cm,
                )
                arrivals.append(s)
        return arrivals

    def _sample_dimensions(self, item_type_value: str) -> Optional[tuple[float, float, float]]:
        samples = self.cfg._olist_dimension_samples or {}
        choices = samples.get(item_type_value)
        if not choices:
            return None
        return self.rng.choice(choices)

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, data: dict) -> None:
        self._event_counter += 1
        self.events.append(Event(
            event_id=f"EVT-{self._event_counter:05d}",
            event_type=event_type,
            timestamp=self.current_time,
            data=data,
        ))

    def _tick_event(self) -> None:
        self._log_event("TICK", {
            "buffer_count": self.buffer.count,
            "buffer_cbm": self.buffer.total_cbm,
            "buffer_effective_cbm": self.buffer.total_effective_cbm,
            "next_cutoff": self.next_cutoff,
        })

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _build_result(self, agent) -> SimulationResult:
        dispatched = [s for s in self.all_shipments if s.dispatched]
        n = len(dispatched)
        costs = self.cost_engine.compute(self.mbls, dispatched)

        avg_waiting = sum(s.waiting_time for s in dispatched) / n if n else 0.0
        late_count = sum(1 for s in dispatched if s.is_late())
        fill_rates = [m.total_cbm / self.cfg.max_cbm_per_mbl for m in self.mbls]

        return SimulationResult(
            schema=SimulationResult.version(),
            agent_id=getattr(agent, "agent_id", "unknown"),
            config=self.cfg.to_dict(),
            metrics=Metrics(
                total_shipments=len(self.all_shipments),
                dispatched_shipments=n,
                number_of_mbls=len(self.mbls),
                total_hbls=sum(len(m.hbls) for m in self.mbls),
                avg_waiting_time_hrs=round(avg_waiting, 2),
                sla_violation_rate=round(late_count / n if n else 0.0, 4),
                avg_fill_rate=round(sum(fill_rates) / len(fill_rates) if fill_rates else 0.0, 4),
                compatibility_violations=self._compatibility_violations,
                compatibility_extra_mbls=self._compatibility_extra_mbls,
                **costs,
            ),
            events=list(self.events),
        )

    def _reset(self) -> None:
        self.rng = random.Random(self.cfg.seed)
        self.buffer = WarehouseBuffer()
        self.current_time = 0.0
        self.next_cutoff = self.cfg.cutoff_interval_hours
        self.all_shipments = []
        self.mbls = []
        self.events = []
        self._event_counter = 0
        self._compatibility_violations = 0
        self._compatibility_extra_mbls = 0
