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
            "status": self.status,
            "current_time": self.env.current_time,
            "time_to_cutoff": round(self.env.next_cutoff - self.env.current_time, 2),
            "next_cutoff": self.env.next_cutoff,
            "sim_duration_hours": self.env.cfg.sim_duration_hours,
            "config": {
                "max_cbm_per_mbl": self.env.cfg.max_cbm_per_mbl,
                "sla_hours": self.env.cfg.sla_hours,
            },
            "buffer": {
                "count": self.env.buffer.count,
                "total_cbm": self.env.buffer.total_cbm,
                "total_weight": self.env.buffer.total_weight,
                "shipments": [
                    {
                        "shipment_id": s.shipment_id,
                        "item_type": s.item_type.value,
                        "arrival_time": s.arrival_time,
                        "waiting_time": round(self.env.current_time - s.arrival_time, 2),
                        "cbm": s.cbm,
                        "weight": s.weight,
                        "packages": s.packages,
                        "due_time": s.due_time,
                        "time_to_due": round(s.due_time - self.env.current_time, 2),
                    }
                    for s in shipments
                ],
            },
            "mbls": [
                {
                    "mbl_id": m.mbl_id,
                    "dispatch_time": m.dispatch_time,
                    "total_cbm": m.total_cbm,
                    "total_weight": m.total_weight,
                    "shipment_count": len(m.shipment_ids),
                    "fill_rate": round(m.total_cbm / self.env.cfg.max_cbm_per_mbl, 4),
                }
                for m in self.env.mbls
            ],
            "events": [e.to_dict() for e in self.env.events[-50:]],  # 최근 50개
        }

    def get_metrics(self) -> dict:
        if self.env is None:
            return {}
        dispatched = [s for s in self.env.all_shipments if s.dispatched]
        n = len(dispatched)
        costs = self.env.cost_engine.compute(self.env.mbls, dispatched)
        avg_waiting = sum(s.waiting_time for s in dispatched) / n if n else 0.0
        late_count = sum(1 for s in dispatched if s.is_late())
        fill_rates = [m.total_cbm / self.env.cfg.max_cbm_per_mbl for m in self.env.mbls]

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
