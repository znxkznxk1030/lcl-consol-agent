"""
reward.py
=========
Step-level reward 계산.

보상 구성:
  r_hold   = -Δ holding cost (매 tick 보관비 누적)
  r_late   = -Δ SLA 위반 패널티 (위반 발생 시)
  r_fill   = +fill rate bonus (dispatch 시 적재 효율 보상)
  r_compat = -compatibility 위반 패널티

모든 항목은 배송 화물 수로 정규화 (scale 안정성).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ── 보상 가중치 ────────────────────────────────────────────────
W_HOLD   = 1.0    # 보관비 가중치
W_LATE   = 5.0    # SLA 위반 패널티 (강하게)
W_FILL   = 2.0    # 충전율 보너스
W_COMPAT = 3.0    # 호환성 위반 패널티

FILL_TARGET_BONUS_THRESHOLD = 0.65  # 이 이상 충전율이면 보너스


@dataclass
class StepInfo:
    """환경 한 tick 전후의 상태 스냅샷."""
    time: float
    buffer_count: int
    total_effective_cbm: float
    mbls_created: int        # 이번 tick에 생성된 MBL 수
    late_shipments: int      # 이번 tick에 발생한 SLA 위반 수
    fill_rates: list         # 이번 tick에 생성된 MBL들의 fill rate
    compat_violations: int   # 이번 tick 호환성 위반 수
    holding_cost_per_hour: float = 2.0
    late_penalty: float = 50.0
    dispatched_count: int = 0


def compute_reward(before: StepInfo, after: StepInfo) -> float:
    """
    두 tick 사이의 step reward 계산.

    Parameters
    ----------
    before : StepInfo  — action 실행 전 상태
    after  : StepInfo  — action 실행 후 (다음 tick) 상태

    Returns
    -------
    float : 정규화된 step reward
    """
    n = max(after.buffer_count + after.dispatched_count, 1)

    # 보관비: 버퍼에 머문 화물들의 시간당 비용
    r_hold = -W_HOLD * after.buffer_count * after.holding_cost_per_hour / n

    # SLA 위반 패널티
    r_late = -W_LATE * after.late_shipments * after.late_penalty / n

    # 충전율 보너스: dispatch 시 fill rate가 threshold 이상이면 보너스
    if after.fill_rates:
        avg_fill = sum(after.fill_rates) / len(after.fill_rates)
        r_fill = W_FILL * max(0.0, avg_fill - FILL_TARGET_BONUS_THRESHOLD)
    else:
        r_fill = 0.0

    # 호환성 위반 패널티 (불필요한 MBL 추가 생성)
    r_compat = -W_COMPAT * after.compat_violations / n

    return float(r_hold + r_late + r_fill + r_compat)


def compute_episode_return(metrics: dict) -> float:
    """
    에피소드 종료 후 최종 평가 지표 → scalar reward.
    학습 평가 및 논문 비교 지표로 사용.
    """
    fill_rate   = metrics.get("avg_fill_rate", 0.0)
    sla_viol    = metrics.get("sla_violation_rate", 0.0)
    total_cost  = metrics.get("total_cost", 1e6)
    n_ships     = max(metrics.get("total_shipments", 1), 1)

    return (
        2.0 * fill_rate
        - 5.0 * sla_viol
        - total_cost / (n_ships * 100.0)   # 정규화
    )
