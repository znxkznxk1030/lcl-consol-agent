"""
state_store.py
==============
시뮬레이터 공유 상태 (thread-safe)
- tick loop와 API 핸들러가 함께 접근
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from simulator_v1.env import ConsolidationEnv, EnvConfig
from simulator_v1.entities import Shipment, MBL
from simulator_v1.volume_model import effective_cbm_from_raw, usable_container_cbm
from simulator_v1.schemas import Event


class SimStatus(str, Enum):
    IDLE = "idle"               # 시작 전
    WAITING = "waiting"         # tick 처리 완료, next_tick 대기 중
    PROCESSING = "processing"   # tick 처리 중
    DONE = "done"               # 시뮬 종료


@dataclass
class SimState:
    status: SimStatus = SimStatus.IDLE
    current_time: float = 0.0
    next_cutoff: float = 0.0
    buffer_count: int = 0
    buffer_cbm: float = 0.0
    buffer_shipments: list = field(default_factory=list)
    total_mbls: int = 0
    events: List[dict] = field(default_factory=list)
    last_tick_arrivals: List[dict] = field(default_factory=list)  # 이번 tick에 도착한 화물


class SimulationStore:
    def __init__(self) -> None:
        self.env: Optional[ConsolidationEnv] = None
        self.status: SimStatus = SimStatus.IDLE
        self.tick_trigger: asyncio.Event = asyncio.Event()
        self._lock: asyncio.Lock = asyncio.Lock()
        self.last_metrics: Optional[dict] = None

    def get_state(self) -> dict:
        if self.env is None:
            return {"status": self.status, "current_time": 0.0, "buffer": {}}

        shipments = self.env.buffer.all()
        return {
            "schema": "observation/v3",
            "status": self.status,
            "current_time": self.env.current_time,
            "time_to_cutoff": round(self.env.next_cutoff - self.env.current_time, 2),
            "next_cutoff": self.env.next_cutoff,
            "sim_duration_hours": self.env.cfg.sim_duration_hours,
            "config": {
                "max_cbm_per_mbl": self.env.cfg.max_cbm_per_mbl,
                "usable_cbm_per_mbl": usable_container_cbm(self.env.cfg.max_cbm_per_mbl),
                "sla_hours": self.env.cfg.sla_hours,
                "destinations": self.env._configured_destinations(),
                "max_active_containers": self.env.cfg.max_active_containers,
            },
            "buffer": {
                "count": self.env.buffer.count,
                "total_cbm": self.env.buffer.total_cbm,
                "total_effective_cbm": self.env.buffer.total_effective_cbm,
                "total_weight": self.env.buffer.total_weight,
                "shipments": [
                    {
                        "shipment_id": s.shipment_id,
                        "item_type": s.item_type.value,
                        "destination": s.destination,
                        "cargo_category": s.cargo_category.value,
                        "arrival_time": s.arrival_time,
                        "waiting_time": round(self.env.current_time - s.arrival_time, 2),
                        "cbm": s.cbm,
                        "effective_cbm": s.effective_cbm,
                        "weight": s.weight,
                        "packages": s.packages,
                        "due_time": s.due_time,
                        "time_to_due": round(s.due_time - self.env.current_time, 2),
                        "length_cm": s.length_cm,
                        "height_cm": s.height_cm,
                        "width_cm": s.width_cm,
                    }
                    for s in shipments
                ],
            },
            "mbls": self._serialize_mbls(),
            "active_slots": self._serialize_active_slots(),
            "events": [e.to_dict() for e in self.env.events[-50:]],  # 최근 50개
        }

    def _serialize_active_slots(self) -> list:
        usable_cbm = usable_container_cbm(self.env.cfg.max_cbm_per_mbl)
        result = []
        for slot in self.env.active_slots.values():
            shipments = [
                {
                    "shipment_id": s.shipment_id,
                    "item_type": s.item_type.value,
                    "destination": s.destination,
                    "cargo_category": s.cargo_category.value,
                    "arrival_time": s.arrival_time,
                    "waiting_time": round(self.env.current_time - s.arrival_time, 2),
                    "cbm": s.cbm,
                    "effective_cbm": s.effective_cbm,
                    "weight": s.weight,
                    "packages": s.packages,
                    "due_time": s.due_time,
                    "time_to_due": round(s.due_time - self.env.current_time, 2),
                    "length_cm": s.length_cm,
                    "height_cm": s.height_cm,
                    "width_cm": s.width_cm,
                }
                for s in slot.shipments
            ]
            result.append({
                "slot_id": slot.slot_id,
                "max_cbm": slot.max_cbm,
                "usable_cbm": usable_cbm,
                "current_cbm": slot.total_cbm,
                "current_effective_cbm": slot.total_effective_cbm,
                "fill_rate": slot.fill_rate,
                "shipment_count": len(slot.shipments),
                "shipment_ids": slot.shipment_ids,
                "shipments": shipments,
                "opened_at": slot.opened_at,
            })
        return result

    def _serialize_mbls(self) -> list:
        ship_map = {s.shipment_id: s for s in self.env.all_shipments}
        max_cbm = self.env.cfg.max_cbm_per_mbl
        result = []
        for m in self.env.mbls:
            result.append(self._serialize_mbl(m, ship_map, max_cbm))
        return result

    def _serialize_mbl(self, m: MBL, ship_map: dict, max_cbm: float) -> dict:
        hbls = []
        for h in m.hbls:
            s = ship_map.get(h.shipment_id)
            effective_cbm = round(effective_cbm_from_raw(h.cbm, h.cargo_category), 3)
            hbls.append({
                "hbl_id": h.hbl_id,
                "shipment_id": h.shipment_id,
                "item_type": s.item_type.value if s else "—",
                "destination": s.destination if s else "—",
                "cargo_category": h.cargo_category,
                "cbm": h.cbm,
                "effective_cbm": effective_cbm,
                "weight": h.weight,
                "packages": h.packages,
                "arrival_time": s.arrival_time if s else None,
                "waiting_time": round(s.waiting_time, 2) if s else None,
                "is_late": s.is_late() if s else False,
                "length_cm": h.length_cm,
                "height_cm": h.height_cm,
                "width_cm": h.width_cm,
            })
        return {
            "mbl_id": m.mbl_id,
            "dispatch_time": m.dispatch_time,
            "total_cbm": m.total_cbm,
            "total_effective_cbm": m.total_effective_cbm,
            "total_weight": m.total_weight,
            "total_packages": m.total_packages,
            "shipment_count": len(m.shipment_ids),
            "usable_cbm_per_mbl": usable_container_cbm(max_cbm),
            "fill_rate": round(m.total_effective_cbm / usable_container_cbm(max_cbm), 4),
            "nominal_fill_rate": round(m.total_cbm / max_cbm, 4),
            "loading_plan": m.loading_plan,
            "hbls": hbls,
        }

    def get_serialized_mbl(self, mbl_id: str) -> Optional[dict]:
        if self.env is None:
            return None

        ship_map = {s.shipment_id: s for s in self.env.all_shipments}
        max_cbm = self.env.cfg.max_cbm_per_mbl
        for m in self.env.mbls:
            if m.mbl_id == mbl_id:
                return self._serialize_mbl(m, ship_map, max_cbm)
        return None

    def get_metrics(self) -> dict:
        if self.env is None:
            return {}
        dispatched = [s for s in self.env.all_shipments if s.dispatched]
        n = len(dispatched)
        costs = self.env.cost_engine.compute(self.env.mbls, dispatched)
        avg_waiting = sum(s.waiting_time for s in dispatched) / n if n else 0.0
        late_count = sum(1 for s in dispatched if s.is_late())
        usable_cbm = usable_container_cbm(self.env.cfg.max_cbm_per_mbl)
        fill_rates = [m.total_effective_cbm / usable_cbm for m in self.env.mbls] if usable_cbm > 0 else []

        return {
            "total_shipments": len(self.env.all_shipments),
            "dispatched_shipments": n,
            "number_of_mbls": len(self.env.mbls),
            "avg_waiting_time_hrs": round(avg_waiting, 2),
            "sla_violation_rate": round(late_count / n if n else 0.0, 4),
            "avg_fill_rate": round(sum(fill_rates) / len(fill_rates) if fill_rates else 0.0, 4),
            **costs,
        }


# 싱글톤
store = SimulationStore()
