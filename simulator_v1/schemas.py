"""
schemas.py
==========
JSON 스키마 정의 (Observation / Action / Event / Result)
모든 컴포넌트는 이 스키마를 통해 통신한다.
"""

from __future__ import annotations
from typing import Literal, List, Optional
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Observation Schema  (Simulator → Agent)
# ---------------------------------------------------------------------------

@dataclass
class ShipmentObservation:
    shipment_id: str
    item_type: str              # "A" | "B" | "C"
    arrival_time: float
    waiting_time: float
    cbm: float
    weight: float
    packages: int
    due_time: float
    time_to_due: float          # due_time - current_time


@dataclass
class BufferObservation:
    count: int
    total_cbm: float
    total_weight: float
    shipments: List[ShipmentObservation]


@dataclass
class ConfigObservation:
    max_cbm_per_mbl: float
    sla_hours: float


@dataclass
class Observation:
    schema: str
    current_time: float
    time_to_cutoff: float
    config: ConfigObservation
    buffer: BufferObservation

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def version(cls) -> str:
        return "observation/v1"


# ---------------------------------------------------------------------------
# Action Schema  (Agent → Simulator)
# ---------------------------------------------------------------------------

@dataclass
class Action:
    schema: str
    agent_id: str
    action: Literal["WAIT", "DISPATCH"]
    selected_ids: List[str]
    reason: Optional[str] = None    # 디버깅 / LLM 해석용

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def wait(cls, agent_id: str, reason: str = "") -> "Action":
        return cls(
            schema="action/v1",
            agent_id=agent_id,
            action="WAIT",
            selected_ids=[],
            reason=reason,
        )

    @classmethod
    def dispatch(cls, agent_id: str, selected_ids: List[str], reason: str = "") -> "Action":
        return cls(
            schema="action/v1",
            agent_id=agent_id,
            action="DISPATCH",
            selected_ids=selected_ids,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Event Schema  (시각화용 이벤트 로그)
# ---------------------------------------------------------------------------

EventType = Literal[
    "SHIPMENT_ARRIVAL",
    "AGENT_DECISION",
    "DISPATCH",
    "MBL_CREATED",
    "SLA_VIOLATION",
    "TICK",
]


@dataclass
class Event:
    event_id: str
    event_type: EventType
    timestamp: float
    data: dict

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Result Schema  (시뮬레이션 최종 결과)
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    total_shipments: int
    dispatched_shipments: int
    number_of_mbls: int
    total_hbls: int
    avg_waiting_time_hrs: float
    sla_violation_rate: float
    avg_fill_rate: float
    total_cost: float
    mbl_cost: float
    holding_cost: float
    late_cost: float


@dataclass
class SimulationResult:
    schema: str
    agent_id: str
    config: dict
    metrics: Metrics
    events: List[Event] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def version(cls) -> str:
        return "result/v1"
