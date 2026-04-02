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
    item_type: str              # "ELECTRONICS"|"CLOTHING"|"COSMETICS"|"FOOD_PRODUCTS"|"AUTO_PARTS"|"CHEMICALS"|"FURNITURE"|"MACHINERY"

    cargo_category: str         # "GENERAL" | "HAZMAT" | "FOOD" | "FRAGILE" | "OVERSIZED"
    arrival_time: float
    waiting_time: float
    cbm: float
    effective_cbm: float        # 패킹 시 실제 점유 CBM (FRAGILE은 cbm × 1.3)
    weight: float
    packages: int
    due_time: float
    time_to_due: float          # due_time - current_time
    length_cm: Optional[float] = None
    height_cm: Optional[float] = None
    width_cm: Optional[float] = None


@dataclass
class BufferObservation:
    count: int
    total_cbm: float
    total_effective_cbm: float  # 패킹 계산 기준 CBM
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
        return "observation/v2"


# ---------------------------------------------------------------------------
# Action Schema  (Agent → Simulator)
# ---------------------------------------------------------------------------

@dataclass
class Action:
    schema: str
    agent_id: str
    action: Literal["WAIT", "DISPATCH"]
    # 여러 MBL을 한 번에 제안: 각 inner list = 하나의 MBL에 담을 shipment ID 목록
    mbls: List[List[str]]
    reason: Optional[str] = None    # 디버깅 / LLM 해석용

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def wait(cls, agent_id: str, reason: str = "") -> "Action":
        return cls(
            schema="action/v1",
            agent_id=agent_id,
            action="WAIT",
            mbls=[],
            reason=reason,
        )

    @classmethod
    def dispatch(cls, agent_id: str, mbls: List[List[str]], reason: str = "") -> "Action":
        return cls(
            schema="action/v1",
            agent_id=agent_id,
            action="DISPATCH",
            mbls=mbls,
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
    "COMPATIBILITY_VIOLATION",  # 혼적 불가 그룹 감지 → 자동 분리
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
    # 호환성 관련
    compatibility_violations: int       # 혼적 불가 dispatch 발생 횟수
    compatibility_extra_mbls: int       # 분리로 인해 추가 생성된 MBL 수
    # 비용
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
        return "result/v2"
