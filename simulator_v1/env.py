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
from dataclasses import dataclass
from typing import List, Optional

from .entities import (
    ItemType, Shipment, MBL,
    generate_shipment, create_mbl,
)
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
    max_cbm_per_mbl: float = 10.0
    destination: str = "PORT_A"

    arrival_rate_A: float = 2.0
    arrival_rate_B: float = 0.8
    arrival_rate_C: float = 0.2

    fixed_cost_per_mbl: float = 100.0
    variable_cost_per_cbm: float = 10.0
    holding_cost_per_hour: float = 2.0
    late_penalty: float = 50.0

    sla_hours: float = 48.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


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
                    "cbm": s.cbm,
                    "weight": s.weight,
                    "due_time": s.due_time,
                })

            # 2. Observation 생성 → Agent에 전달
            obs = self._build_observation()
            action_dict = agent.act(obs.to_dict())
            action = self._parse_action(action_dict)

            # 3. Agent 결정 로깅
            self._log_event("AGENT_DECISION", {
                "agent_id": action.agent_id,
                "action": action.action,
                "selected_ids": action.selected_ids,
                "reason": action.reason,
                "buffer_cbm": self.buffer.total_cbm,
                "buffer_count": self.buffer.count,
            })

            # 4. Action 실행
            if action.action == "DISPATCH" and action.selected_ids:
                self._dispatch(action.selected_ids)

            # 5. Cutoff 갱신
            if self.current_time >= self.next_cutoff:
                self.next_cutoff += self.cfg.cutoff_interval_hours

            self.current_time += 1.0

        # 잔여 화물 강제 출고
        remaining = self.buffer.ids()
        if remaining:
            self._dispatch(remaining)

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
                total_weight=self.buffer.total_weight,
                shipments=[
                    ShipmentObservation(
                        shipment_id=s.shipment_id,
                        item_type=s.item_type.value,
                        arrival_time=s.arrival_time,
                        waiting_time=round(self.current_time - s.arrival_time, 2),
                        cbm=s.cbm,
                        weight=s.weight,
                        packages=s.packages,
                        due_time=s.due_time,
                        time_to_due=round(s.due_time - self.current_time, 2),
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
            selected_ids=action_dict.get("selected_ids", []),
            reason=action_dict.get("reason"),
        )

    def _dispatch(self, shipment_ids: List[str]) -> None:
        dispatched = self.buffer.remove(shipment_ids)
        for s in dispatched:
            s.dispatched = True
            s.dispatch_time = self.current_time

        mbl = create_mbl(dispatched, self.current_time)
        self.mbls.append(mbl)

        # SLA violation 체크
        for s in dispatched:
            if s.is_late():
                self._log_event("SLA_VIOLATION", {
                    "shipment_id": s.shipment_id,
                    "due_time": s.due_time,
                    "dispatch_time": s.dispatch_time,
                    "overdue_hours": round(s.dispatch_time - s.due_time, 2),
                })

        self._log_event("DISPATCH", {
            "shipment_ids": [s.shipment_id for s in dispatched],
            "count": len(dispatched),
        })
        self._log_event("MBL_CREATED", {
            "mbl_id": mbl.mbl_id,
            "shipment_count": len(mbl.shipment_ids),
            "total_cbm": mbl.total_cbm,
            "total_weight": mbl.total_weight,
            "fill_rate": round(mbl.total_cbm / self.cfg.max_cbm_per_mbl, 4),
            "hbl_ids": [h.hbl_id for h in mbl.hbls],
        })

    # ------------------------------------------------------------------
    # Arrival generator (Poisson)
    # ------------------------------------------------------------------

    def _generate_arrivals(self, hour: float) -> List[Shipment]:
        arrivals = []
        for item_type, rate in [
            (ItemType.A, self.cfg.arrival_rate_A),
            (ItemType.B, self.cfg.arrival_rate_B),
            (ItemType.C, self.cfg.arrival_rate_C),
        ]:
            t = 0.0
            while True:
                inter = self.rng.expovariate(rate)
                t += inter
                if t > 1.0:
                    break
                s = generate_shipment(
                    self.rng, hour + t, item_type,
                    destination=self.cfg.destination,
                    sla_hours=self.cfg.sla_hours,
                )
                arrivals.append(s)
        return arrivals

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
