"""
sla_analyzer.py
===============
SLA 위험도 분석 모듈.

observation["buffer"]["shipments"]의 time_to_due 필드를 기반으로
각 화물의 SLA 위반 위험도를 분류하고 즉시 출하 권고 목록을 생성.

위험 등급:
  CRITICAL : time_to_due < 6h   → 즉시 출하 필수
  HIGH     : time_to_due < 12h  → 이번 tick 출하 강력 권고
  MEDIUM   : time_to_due < 24h  → 다음 2~3 tick 내 출하 필요
  LOW      : time_to_due >= 24h → 여유 있음
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# 위험 등급 임계값 (시간)
# ---------------------------------------------------------------------------

RISK_THRESHOLDS: Dict[str, float] = {
    "CRITICAL": 6.0,
    "HIGH":    12.0,
    "MEDIUM":  24.0,
}

# SLA 위반 시 기본 패널티 (cost.py 기본값과 동기화)
DEFAULT_LATE_PENALTY = 50.0


# ---------------------------------------------------------------------------
# 결과 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class SLARisk:
    shipment_id: str
    item_type: str
    cargo_category: str
    time_to_due_hours: float
    waiting_time_hours: float
    risk_level: str          # CRITICAL / HIGH / MEDIUM / LOW
    recommended_action: str  # "DISPATCH_NOW" | "DISPATCH_SOON" | "MONITOR" | "OK"

    def to_dict(self) -> dict:
        return {
            "shipment_id": self.shipment_id,
            "item_type": self.item_type,
            "cargo_category": self.cargo_category,
            "time_to_due_hours": round(self.time_to_due_hours, 2),
            "waiting_time_hours": round(self.waiting_time_hours, 2),
            "risk_level": self.risk_level,
            "recommended_action": self.recommended_action,
        }


@dataclass
class SLARiskAssessment:
    at_risk_shipments: List[SLARisk] = field(default_factory=list)
    safe_shipments: List[str] = field(default_factory=list)         # LOW 등급 ID
    recommended_priority_dispatch: List[str] = field(default_factory=list)  # CRITICAL+HIGH ID
    sla_violation_probability: float = 0.0   # 1 tick 대기 시 위반 예상 비율
    expected_penalty_if_wait: float = 0.0    # 대기 시 예상 추가 페널티 (USD)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    total_shipments: int = 0

    def to_dict(self) -> dict:
        return {
            "at_risk_shipments": [r.to_dict() for r in self.at_risk_shipments],
            "safe_shipments": self.safe_shipments,
            "recommended_priority_dispatch": self.recommended_priority_dispatch,
            "sla_violation_probability": round(self.sla_violation_probability, 3),
            "expected_penalty_if_wait": round(self.expected_penalty_if_wait, 2),
            "risk_summary": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
                "total": self.total_shipments,
            },
        }


# ---------------------------------------------------------------------------
# SLA 분석기
# ---------------------------------------------------------------------------

class SLAAnalyzer:
    """
    각 화물의 SLA 위험도를 분석하고 즉시 출하 우선순위를 결정.
    """

    def __init__(self, late_penalty: float = DEFAULT_LATE_PENALTY):
        self.late_penalty = late_penalty

    def analyze(
        self,
        shipments: List[dict],
        tick_hours: float = 1.0,
    ) -> SLARiskAssessment:
        """
        Parameters
        ----------
        shipments : List[dict]
            observation["buffer"]["shipments"] 전체 목록
        tick_hours : float
            시뮬레이션 1 tick 단위 (기본 1시간)
            — 이 시간만큼 대기 시 위반 예상 계산에 사용

        Returns
        -------
        SLARiskAssessment
        """
        if not shipments:
            return SLARiskAssessment(total_shipments=0)

        risks: List[SLARisk] = []
        safe_ids: List[str] = []
        priority_ids: List[str] = []

        for s in shipments:
            time_to_due = s.get("time_to_due", 999.0)
            risk_level, action = _classify_risk(time_to_due)

            risk = SLARisk(
                shipment_id=s["shipment_id"],
                item_type=s.get("item_type", "UNKNOWN"),
                cargo_category=s.get("cargo_category", "GENERAL"),
                time_to_due_hours=time_to_due,
                waiting_time_hours=s.get("waiting_time", 0.0),
                risk_level=risk_level,
                recommended_action=action,
            )

            if risk_level == "LOW":
                safe_ids.append(s["shipment_id"])
            else:
                risks.append(risk)
                if risk_level in ("CRITICAL", "HIGH"):
                    priority_ids.append(s["shipment_id"])

        # 위험도 카운트
        c_count = sum(1 for r in risks if r.risk_level == "CRITICAL")
        h_count = sum(1 for r in risks if r.risk_level == "HIGH")
        m_count = sum(1 for r in risks if r.risk_level == "MEDIUM")

        # 1 tick 대기 시 추가 위반 예상
        # CRITICAL 중 time_to_due <= tick_hours → 확실한 위반
        # HIGH 중 time_to_due <= tick_hours * 2 → 높은 위반 확률 (0.7 적용)
        certain_violations = sum(
            1 for r in risks
            if r.risk_level == "CRITICAL" and r.time_to_due_hours <= tick_hours
        )
        probable_violations = sum(
            0.7 for r in risks
            if r.risk_level == "HIGH" and r.time_to_due_hours <= tick_hours * 2
        )
        total = len(shipments)
        violation_prob = (certain_violations + probable_violations) / total if total > 0 else 0.0
        expected_penalty = (certain_violations + probable_violations) * self.late_penalty

        # at_risk_shipments: time_to_due 오름차순 정렬 (급한 것 우선)
        risks.sort(key=lambda r: r.time_to_due_hours)

        return SLARiskAssessment(
            at_risk_shipments=risks,
            safe_shipments=safe_ids,
            recommended_priority_dispatch=priority_ids,
            sla_violation_probability=min(1.0, violation_prob),
            expected_penalty_if_wait=expected_penalty,
            critical_count=c_count,
            high_count=h_count,
            medium_count=m_count,
            low_count=len(safe_ids),
            total_shipments=total,
        )

    def get_dispatch_urgency(self, assessment: SLARiskAssessment) -> str:
        """전체 상황을 한 마디로 요약."""
        if assessment.critical_count > 0:
            return "IMMEDIATE"
        if assessment.high_count > 0:
            return "URGENT"
        if assessment.medium_count > 0:
            return "SOON"
        return "NORMAL"


def _classify_risk(time_to_due: float) -> Tuple[str, str]:
    """time_to_due(시간)로 위험 등급과 권고 액션 반환."""
    if time_to_due < RISK_THRESHOLDS["CRITICAL"]:
        return "CRITICAL", "DISPATCH_NOW"
    elif time_to_due < RISK_THRESHOLDS["HIGH"]:
        return "HIGH", "DISPATCH_SOON"
    elif time_to_due < RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM", "MONITOR"
    else:
        return "LOW", "OK"
