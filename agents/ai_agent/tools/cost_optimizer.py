"""
cost_optimizer.py
=================
DISPATCH vs WAIT 비용 비교 최적화 모듈.

기존 simulator_v1/cost.py의 CostEngine 공식을 기반으로
현재 출하 비용과 1~2 tick 대기 후 예상 비용을 비교해
최적 출하 시점과 예상 consolidation 효율을 계산.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict


# ---------------------------------------------------------------------------
# 비용 파라미터 (cost.py 기본값과 동기화)
# ---------------------------------------------------------------------------

DEFAULT_FIXED_COST_PER_MBL      = 100.0   # MBL 1개당 고정 운임
DEFAULT_VARIABLE_COST_PER_CBM   = 10.0    # CBM당 변동 운임
DEFAULT_HOLDING_COST_PER_HOUR   = 2.0     # 화물 1개 1시간당 보관비
DEFAULT_LATE_PENALTY            = 50.0    # SLA 위반 1건당 패널티


# ---------------------------------------------------------------------------
# 결과 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class CostAnalysis:
    dispatch_now_cost: float = 0.0
    wait_1tick_expected_cost: float = 0.0
    wait_2tick_expected_cost: float = 0.0
    optimal_dispatch_time: float = 0.0      # 시뮬레이션 절대 시간 (hour)
    cost_breakdown_now: Dict[str, float] = field(default_factory=dict)
    consolidation_efficiency: float = 0.0  # 평균 fill rate (0~1)
    cost_per_shipment: float = 0.0
    number_of_mbls_now: int = 0
    recommendation: str = "WAIT"           # "DISPATCH" | "WAIT"
    confidence: float = 0.0                # 권고 신뢰도 (0~1)
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "dispatch_now_cost": round(self.dispatch_now_cost, 2),
            "wait_1tick_expected_cost": round(self.wait_1tick_expected_cost, 2),
            "wait_2tick_expected_cost": round(self.wait_2tick_expected_cost, 2),
            "optimal_dispatch_time": round(self.optimal_dispatch_time, 1),
            "cost_breakdown_now": {k: round(v, 2) for k, v in self.cost_breakdown_now.items()},
            "consolidation_efficiency": round(self.consolidation_efficiency, 3),
            "cost_per_shipment": round(self.cost_per_shipment, 2),
            "number_of_mbls_now": self.number_of_mbls_now,
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# 비용 최적화기
# ---------------------------------------------------------------------------

class CostOptimizer:
    """
    현재 시점 출하 vs 대기의 비용 차이를 분석해 최적 출하 시점을 권고.
    """

    def __init__(
        self,
        fixed_cost_per_mbl: float = DEFAULT_FIXED_COST_PER_MBL,
        variable_cost_per_cbm: float = DEFAULT_VARIABLE_COST_PER_CBM,
        holding_cost_per_hour: float = DEFAULT_HOLDING_COST_PER_HOUR,
        late_penalty: float = DEFAULT_LATE_PENALTY,
    ):
        self.fixed = fixed_cost_per_mbl
        self.variable = variable_cost_per_cbm
        self.holding = holding_cost_per_hour
        self.late_penalty = late_penalty

    def analyze(
        self,
        observation: dict,
        proposed_mbl_groups: List[List[str]],
        expected_arrivals_next_tick: float = 0.0,
        expected_cbm_next_tick: float = 0.0,
    ) -> CostAnalysis:
        """
        Parameters
        ----------
        observation : dict
            시뮬레이터 observation dict
        proposed_mbl_groups : List[List[str]]
            현재 제안된 MBL 그룹 (bin packing 결과 or compatible_bin_pack 결과)
        expected_arrivals_next_tick : float
            다음 tick에 예상되는 신규 화물 수 (VolumeForecaster 결과)
        expected_cbm_next_tick : float
            다음 tick에 예상되는 신규 CBM (VolumeForecaster 결과)

        Returns
        -------
        CostAnalysis
        """
        buf = observation.get("buffer", {})
        shipments = buf.get("shipments", [])
        current_time = observation.get("current_time", 0.0)
        time_to_cutoff = observation.get("time_to_cutoff", 24.0)
        max_cbm = observation.get("config", {}).get("max_cbm_per_mbl", 10.0)

        if not shipments:
            return CostAnalysis(
                recommendation="WAIT",
                confidence=1.0,
                reasoning="버퍼 비어 있음 — 대기",
            )

        # --- 현재 출하 비용 계산 ---
        cost_now = self._compute_dispatch_cost(shipments, proposed_mbl_groups, max_cbm)
        breakdown_now = cost_now

        # consolidation 효율: 각 MBL의 fill rate 평균
        eff = self._consolidation_efficiency(shipments, proposed_mbl_groups, max_cbm)
        n_mbls = len(proposed_mbl_groups) if proposed_mbl_groups else 1
        cost_per_ship = cost_now["total"] / len(shipments) if shipments else 0.0

        # --- 1 tick 대기 비용 (예상) ---
        # 보관비 증가 + SLA 위반 확률 반영 + 예상 신규 화물로 fill rate 개선 가능성
        extra_holding = len(shipments) * self.holding  # 1 tick 추가 보관
        sla_risk_penalty = self._expected_sla_penalty(shipments, tick=1.0)
        # 신규 화물로 fill rate 개선 효과: 더 채워지면 MBL 수 감소 가능
        improved_fill_benefit = self._fill_improvement_benefit(
            shipments, expected_cbm_next_tick, max_cbm
        )
        cost_wait_1 = cost_now["total"] + extra_holding + sla_risk_penalty - improved_fill_benefit

        # --- 2 tick 대기 비용 (예상) ---
        extra_holding_2 = len(shipments) * self.holding * 2
        sla_risk_penalty_2 = self._expected_sla_penalty(shipments, tick=2.0)
        cost_wait_2 = cost_now["total"] + extra_holding_2 + sla_risk_penalty_2 - improved_fill_benefit * 1.5

        # --- 권고 결정 ---
        recommendation, confidence, reasoning = self._decide(
            cost_now=cost_now["total"],
            cost_wait_1=cost_wait_1,
            efficiency=eff,
            time_to_cutoff=time_to_cutoff,
            sla_risk=sla_risk_penalty,
            shipments=shipments,
        )

        return CostAnalysis(
            dispatch_now_cost=cost_now["total"],
            wait_1tick_expected_cost=max(0.0, cost_wait_1),
            wait_2tick_expected_cost=max(0.0, cost_wait_2),
            optimal_dispatch_time=current_time if recommendation == "DISPATCH" else current_time + 1.0,
            cost_breakdown_now={
                "mbl_cost": cost_now["mbl"],
                "holding_cost": cost_now["holding"],
                "late_cost": cost_now["late"],
                "total": cost_now["total"],
            },
            consolidation_efficiency=eff,
            cost_per_shipment=cost_per_ship,
            number_of_mbls_now=n_mbls,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # 내부 계산 메서드
    # ------------------------------------------------------------------

    def _compute_dispatch_cost(
        self,
        shipments: List[dict],
        mbl_groups: List[List[str]],
        max_cbm: float,
    ) -> Dict[str, float]:
        """현재 출하 시 비용 계산."""
        id_to_ship = {s["shipment_id"]: s for s in shipments}

        if not mbl_groups:
            n_mbls = 1
            total_cbm_all = sum(s.get("effective_cbm", s.get("cbm", 0)) for s in shipments)
        else:
            n_mbls = len(mbl_groups)
            total_cbm_all = 0.0
            for group in mbl_groups:
                group_cbm = sum(
                    id_to_ship[sid].get("effective_cbm", id_to_ship[sid].get("cbm", 0))
                    for sid in group if sid in id_to_ship
                )
                total_cbm_all += group_cbm

        mbl_cost = n_mbls * self.fixed + self.variable * total_cbm_all
        holding_cost = sum(
            self.holding * s.get("waiting_time", 0.0) for s in shipments
        )
        late_cost = sum(
            self.late_penalty for s in shipments
            if s.get("time_to_due", 999.0) < 0  # 이미 늦은 화물
        )
        total = mbl_cost + holding_cost + late_cost
        return {"mbl": mbl_cost, "holding": holding_cost, "late": late_cost, "total": total}

    def _consolidation_efficiency(
        self,
        shipments: List[dict],
        mbl_groups: List[List[str]],
        max_cbm: float,
    ) -> float:
        """각 MBL의 fill rate (effective_cbm / max_cbm) 평균."""
        if not mbl_groups or max_cbm <= 0:
            total_cbm = sum(s.get("effective_cbm", s.get("cbm", 0)) for s in shipments)
            return min(1.0, total_cbm / max_cbm) if max_cbm > 0 else 0.0

        id_to_ship = {s["shipment_id"]: s for s in shipments}
        fill_rates = []
        for group in mbl_groups:
            group_cbm = sum(
                id_to_ship[sid].get("effective_cbm", id_to_ship[sid].get("cbm", 0))
                for sid in group if sid in id_to_ship
            )
            fill_rates.append(min(1.0, group_cbm / max_cbm))

        return sum(fill_rates) / len(fill_rates) if fill_rates else 0.0

    def _expected_sla_penalty(self, shipments: List[dict], tick: float) -> float:
        """tick 시간 대기 시 추가 SLA 위반 예상 페널티."""
        penalty = 0.0
        for s in shipments:
            ttd = s.get("time_to_due", 999.0)
            if 0 <= ttd <= tick:
                penalty += self.late_penalty  # 확실한 위반
            elif tick < ttd <= tick * 2:
                penalty += self.late_penalty * 0.5  # 높은 위험
        return penalty

    def _fill_improvement_benefit(
        self,
        current_shipments: List[dict],
        expected_cbm_next: float,
        max_cbm: float,
    ) -> float:
        """
        신규 화물 도착으로 MBL 수 감소 시 절약 가능한 비용.
        현재 반쯤 찬 MBL에 신규 화물이 채워지면 MBL 1개 절약 가능.
        """
        if expected_cbm_next <= 0 or max_cbm <= 0:
            return 0.0

        total_cbm = sum(s.get("effective_cbm", s.get("cbm", 0)) for s in current_shipments)
        current_n_mbls = max(1, int(total_cbm / max_cbm) + (1 if total_cbm % max_cbm > 0 else 0))
        combined_cbm = total_cbm + expected_cbm_next
        future_n_mbls = max(1, int(combined_cbm / max_cbm) + (1 if combined_cbm % max_cbm > 0 else 0))

        # MBL 수가 줄어들면 고정 비용 절약
        if future_n_mbls < current_n_mbls:
            return (current_n_mbls - future_n_mbls) * self.fixed
        return 0.0

    def _decide(
        self,
        cost_now: float,
        cost_wait_1: float,
        efficiency: float,
        time_to_cutoff: float,
        sla_risk: float,
        shipments: List[dict],
    ):
        """최종 DISPATCH / WAIT 권고 및 신뢰도."""
        reasons = []

        # SLA CRITICAL 화물 있으면 즉시 출하
        critical = [s for s in shipments if s.get("time_to_due", 999) < 6.0]
        if critical:
            return "DISPATCH", 0.95, f"CRITICAL SLA 위험 {len(critical)}건"

        # cutoff 임박
        if time_to_cutoff <= 2.0:
            return "DISPATCH", 0.98, f"cutoff {time_to_cutoff:.1f}h 이내"

        # fill rate 기준: 70% 이상이면 출하 효율적
        if efficiency >= 0.70:
            reasons.append(f"fill_rate {efficiency*100:.0f}%≥70%")
            if cost_now <= cost_wait_1:
                return "DISPATCH", 0.85, " + ".join(reasons) + " — 지금이 비용 최적"
            else:
                return "DISPATCH", 0.75, " + ".join(reasons) + " — fill_rate 충분"

        # 대기하면 비용 절감
        if cost_wait_1 < cost_now * 0.95:
            saving = cost_now - cost_wait_1
            return "WAIT", 0.80, f"대기 시 ${saving:.1f} 절감 예상 (fill_rate {efficiency*100:.0f}%)"

        # 비용 거의 비슷하면 fill rate로 결정
        if efficiency >= 0.50:
            return "DISPATCH", 0.60, f"fill_rate {efficiency*100:.0f}% — 충분한 통합"

        return "WAIT", 0.65, f"fill_rate {efficiency*100:.0f}% 미달 — 추가 화물 대기"
