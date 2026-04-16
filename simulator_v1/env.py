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
    ItemType, Shipment, MBL, ContainerSlot,
    generate_shipment, create_mbl,
)
from .distributions import thinning_arrivals
from .compatibility import split_into_compatible_groups, count_violation_pairs
from .buffer import WarehouseBuffer
from .cost import CostEngine
from .volume_model import usable_container_cbm
from .planning import normalize_mbl_plans, shipment_ids_from_plan, build_loading_plan
from .schemas import (
    Observation, ConfigObservation, BufferObservation, ShipmentObservation,
    ContainerSlotObservation,
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
    destinations: List[str] = field(default_factory=list)
    destination_weights: Dict[str, float] = field(default_factory=dict)

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
    max_active_containers: int = 1   # 동시에 열 수 있는 최대 컨테이너(슬롯) 수
    _olist_dimension_samples: Optional[Dict[str, List[tuple[float, float, float]]]] = field(
        default=None,
        repr=False,
    )
    _olist_destination_samples: Optional[Dict[str, List[str]]] = field(
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
        self.active_slots: Dict[str, ContainerSlot] = {}   # 현재 열려 있는 컨테이너 슬롯
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
                    "destination": s.destination,
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

            # 5. Cutoff: 열려 있는 슬롯 모두 닫기 + cutoff 갱신
            if self.current_time >= self.next_cutoff:
                for slot_id in list(self.active_slots.keys()):
                    self._close_slot(slot_id)
                self.next_cutoff += self.cfg.cutoff_interval_hours

            self.current_time += 1.0

        # 잔여 슬롯 강제 닫기
        for slot_id in list(self.active_slots.keys()):
            self._close_slot(slot_id)

        # 잔여 버퍼 화물 강제 출고 (자동 CBM 분할)
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
                usable_cbm_per_mbl=usable_container_cbm(self.cfg.max_cbm_per_mbl),
                sla_hours=self.cfg.sla_hours,
                destinations=self._configured_destinations(),
                max_active_containers=self.cfg.max_active_containers,
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
                        destination=s.destination,
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
            containers=[
                ContainerSlotObservation(
                    slot_id=slot.slot_id,
                    max_cbm=slot.max_cbm,
                    usable_cbm=slot.usable_cbm,
                    current_cbm=slot.total_cbm,
                    current_effective_cbm=slot.total_effective_cbm,
                    fill_rate=slot.fill_rate,
                    shipment_count=len(slot.shipments),
                    shipment_ids=slot.shipment_ids,
                    opened_at=slot.opened_at,
                )
                for slot in self.active_slots.values()
            ],
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

    def _open_slot(self, slot_id: Optional[str] = None) -> ContainerSlot:
        """새 컨테이너 슬롯을 열고 active_slots에 등록한다."""
        if slot_id is None:
            slot_id = f"SLOT-{uuid.uuid4().hex[:6].upper()}"
        slot = ContainerSlot(
            slot_id=slot_id,
            max_cbm=self.cfg.max_cbm_per_mbl,
            opened_at=self.current_time,
        )
        self.active_slots[slot_id] = slot
        self._log_event("SLOT_OPENED", {"slot_id": slot_id, "max_cbm": slot.max_cbm})
        return slot

    def _close_slot(self, slot_id: str) -> None:
        """슬롯을 닫고 MBL을 생성한다. 슬롯이 비어 있으면 아무것도 하지 않는다."""
        slot = self.active_slots.pop(slot_id, None)
        if slot is None or not slot.shipments:
            return

        loading_plan = build_loading_plan(
            [
                {
                    "shipment_id": s.shipment_id,
                    "cargo_category": s.cargo_category.value,
                    "item_type": s.item_type.value,
                    "cbm": s.cbm,
                    "effective_cbm": s.effective_cbm,
                    "weight": s.weight,
                }
                for s in slot.shipments
            ],
            max_cbm_per_mbl=self.cfg.max_cbm_per_mbl,
        )
        self._log_event("SLOT_CLOSED", {
            "slot_id": slot_id,
            "shipment_count": len(slot.shipments),
            "total_effective_cbm": slot.total_effective_cbm,
            "fill_rate": slot.fill_rate,
        })
        self._create_mbl(slot.shipments, loading_plan=loading_plan)

    def _dispatch(self, mbl_assignments: List[object]) -> None:
        """mbl_assignments: MBL plan 목록 또는 shipment ID 그룹 목록.

        plan 딕셔너리에 추가 가능한 필드:
          - slot_id (str): 배정할 슬롯 ID. 없으면 새 슬롯 또는 기존 슬롯 선택.
          - close (bool): True(기본)면 즉시 MBL 생성, False면 슬롯 유지.
        """
        normalized_plans = normalize_mbl_plans(mbl_assignments)

        # 전체 ID 한 번에 버퍼에서 제거
        all_ids = [sid for plan in normalized_plans for sid in shipment_ids_from_plan(plan)]
        all_ships = {s.shipment_id: s for s in self.buffer.remove(all_ids)}

        for plan in normalized_plans:
            group_ids = shipment_ids_from_plan(plan)
            candidates = [all_ships[sid] for sid in group_ids if sid in all_ships]
            if not candidates:
                continue

            close = plan.get("close", True)   # 기본값 True → 기존 동작과 동일
            target_slot_id = plan.get("slot_id")

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

            # 각 호환 그룹을 CBM 한도 내로 추가 분할 후 처리
            for group in compat_groups:
                for cbm_group in self._split_by_cbm(group):
                    if close:
                        # 기존 동작: 즉시 MBL 생성
                        final_ids = [s.shipment_id for s in cbm_group]
                        if final_ids == group_ids and plan.get("loading_plan"):
                            loading_plan = plan["loading_plan"]
                        else:
                            loading_plan = build_loading_plan(
                                [
                                    {
                                        "shipment_id": s.shipment_id,
                                        "cargo_category": s.cargo_category.value,
                                        "item_type": s.item_type.value,
                                        "cbm": s.cbm,
                                        "effective_cbm": s.effective_cbm,
                                        "weight": s.weight,
                                    }
                                    for s in cbm_group
                                ],
                                max_cbm_per_mbl=self.cfg.max_cbm_per_mbl,
                                container_type=plan.get("container_type"),
                            )
                        self._create_mbl(cbm_group, loading_plan=loading_plan)
                    else:
                        # 슬롯에 배정만 하고 닫지 않음
                        if target_slot_id and target_slot_id in self.active_slots:
                            slot = self.active_slots[target_slot_id]
                        elif len(self.active_slots) < self.cfg.max_active_containers:
                            slot = self._open_slot(target_slot_id)
                        else:
                            # 슬롯이 꽉 찼으면 가장 채워진 슬롯을 닫고 새로 열기
                            fullest = max(self.active_slots.values(), key=lambda s: s.fill_rate)
                            self._close_slot(fullest.slot_id)
                            slot = self._open_slot(target_slot_id)

                        slot.shipments.extend(cbm_group)
                        self._log_event("SLOT_ASSIGNED", {
                            "slot_id": slot.slot_id,
                            "added_count": len(cbm_group),
                            "slot_fill_rate": slot.fill_rate,
                            "slot_effective_cbm": slot.total_effective_cbm,
                        })

                        # 슬롯이 usable CBM을 초과하면 자동으로 닫기
                        if slot.total_effective_cbm >= slot.usable_cbm:
                            self._close_slot(slot.slot_id)

    def _split_by_cbm(self, shipments: List[Shipment]) -> List[List[Shipment]]:
        """effective_cbm 기준으로 usable container CBM을 초과하지 않도록 분할."""
        result: List[List[Shipment]] = []
        current: List[Shipment] = []
        current_cbm = 0.0
        usable_cbm = usable_container_cbm(self.cfg.max_cbm_per_mbl)

        for s in shipments:
            ecbm = s.effective_cbm
            # 단일 화물 자체가 한도 초과인 경우: 단독 MBL로 처리
            if current and current_cbm + ecbm > usable_cbm:
                result.append(current)
                current = [s]
                current_cbm = ecbm
            else:
                current.append(s)
                current_cbm += ecbm

        if current:
            result.append(current)

        return result or [[]]

    def _create_mbl(self, shipments: List[Shipment], loading_plan: Optional[dict] = None) -> None:
        for s in shipments:
            s.dispatched = True
            s.dispatch_time = self.current_time

        mbl = create_mbl(shipments, self.current_time, loading_plan=loading_plan)
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
            "usable_cbm_per_mbl": usable_container_cbm(self.cfg.max_cbm_per_mbl),
            "fill_rate": round(mbl.total_effective_cbm / usable_container_cbm(self.cfg.max_cbm_per_mbl), 4),
            "nominal_fill_rate": round(mbl.total_cbm / self.cfg.max_cbm_per_mbl, 4),
            "effective_fill_rate": round(mbl.total_effective_cbm / self.cfg.max_cbm_per_mbl, 4),
            "categories": list({h.cargo_category for h in mbl.hbls}),
            "hbl_ids": [h.hbl_id for h in mbl.hbls],
            "has_loading_plan": bool(mbl.loading_plan),
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
                    destination=self._sample_destination(item_type.value),
                    sla_hours=self.cfg.sla_hours,
                    dimensions_cm=dimensions_cm,
                )
                arrivals.append(s)
        return arrivals

    def _configured_destinations(self) -> List[str]:
        olist_samples = self.cfg._olist_destination_samples or {}
        observed = sorted({dest for values in olist_samples.values() for dest in values if dest})
        if observed:
            return observed
        configured = [d for d in self.cfg.destinations if isinstance(d, str) and d.strip()]
        return configured or [self.cfg.destination]

    def _sample_destination(self, item_type_value: Optional[str] = None) -> str:
        olist_samples = self.cfg._olist_destination_samples or {}
        if item_type_value:
            choices = [dest for dest in olist_samples.get(item_type_value, []) if dest]
            if choices:
                return self.rng.choice(choices)

        destinations = self._configured_destinations()
        if len(destinations) == 1:
            return destinations[0]

        weights = [max(0.0, self.cfg.destination_weights.get(dest, 0.0)) for dest in destinations]
        if any(weight > 0 for weight in weights):
            return self.rng.choices(destinations, weights=weights, k=1)[0]
        return self.rng.choice(destinations)

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
        usable_cbm = usable_container_cbm(self.cfg.max_cbm_per_mbl)
        fill_rates = [m.total_effective_cbm / usable_cbm for m in self.mbls] if usable_cbm > 0 else []

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
        self.active_slots = {}
        self.events = []
        self._event_counter = 0
        self._compatibility_violations = 0
        self._compatibility_extra_mbls = 0
