"""
volume_forecast.py
==================
물동량 예측 (Cargo Volume Forecasting) 모듈.

기존 simulator_v1/distributions.py의 HOURLY_MULTIPLIER 패턴을 기반으로,
최근 도착 이벤트를 지수 평활화하여 향후 24/48h 물동량을 예측.

접근 방식:
- 기본 도착률: 시뮬레이터 설정값 또는 Olist 캘리브레이션 값 사용
- 시간대 패턴: HOURLY_MULTIPLIER (24개 요소)
- 최근 관측치로 기본 도착률 보정 (지수 평활화, α=0.3)
- 95% 신뢰 구간: Poisson 분포 특성 (평균 ± 1.96 * sqrt(평균))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# 기본 도착률 (아이템 유형별 시간당 평균 화물 수)
# distributions.py 기반 + Olist 캘리브레이션 기본값
# ---------------------------------------------------------------------------

DEFAULT_ARRIVAL_RATES: Dict[str, float] = {
    "ELECTRONICS":    1.40,
    "CLOTHING":       2.82,
    "COSMETICS":      1.06,
    "FOOD_PRODUCTS":  0.09,
    "AUTO_PARTS":     0.49,
    "CHEMICALS":      0.04,
    "FURNITURE":      0.27,
    "MACHINERY":      0.17,
}

# distributions.py HOURLY_MULTIPLIER 동기화
HOURLY_MULTIPLIER: List[float] = [
    0.30, 0.20, 0.10, 0.10, 0.20, 0.50,   # 0-5시  (심야)
    0.80, 1.20, 1.50, 1.80, 1.90, 2.00,   # 6-11시 (오전 피크)
    1.80, 1.70, 1.60, 1.50, 1.40, 1.20,   # 12-17시 (오후)
    1.00, 0.80, 0.60, 0.50, 0.40, 0.30,   # 18-23시 (저녁)
]

# 아이템 유형별 중앙값 CBM (distributions.py _CBM_PARAMS에서 exp(mu) 계산)
_MEDIAN_CBM: Dict[str, float] = {
    "ELECTRONICS":    math.exp(-1.60),   # ~0.20
    "CLOTHING":       math.exp(-1.20),   # ~0.30
    "COSMETICS":      math.exp(-1.80),   # ~0.17
    "FOOD_PRODUCTS":  math.exp(-0.50),   # ~0.61
    "AUTO_PARTS":     math.exp( 0.10),   # ~1.11
    "CHEMICALS":      math.exp(-0.20),   # ~0.82
    "FURNITURE":      math.exp( 1.20),   # ~3.32
    "MACHINERY":      math.exp( 1.50),   # ~4.48
}

# 지수 평활화 파라미터
_ALPHA = 0.3  # 최근 관측치 가중치 (0~1; 높을수록 최근 데이터 반영 강도↑)


# ---------------------------------------------------------------------------
# 결과 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class ForecastSeries:
    item_type: str
    hourly_expected_count: List[float]   # 각 시간대 예상 화물 수
    hourly_expected_cbm: List[float]     # 각 시간대 예상 CBM
    cumulative_expected_cbm: float
    trend_direction: str   # "increasing" | "stable" | "decreasing"

    def to_dict(self) -> dict:
        return {
            "item_type": self.item_type,
            "hourly_expected_count": [round(v, 3) for v in self.hourly_expected_count],
            "hourly_expected_cbm": [round(v, 3) for v in self.hourly_expected_cbm],
            "cumulative_expected_cbm": round(self.cumulative_expected_cbm, 3),
            "trend_direction": self.trend_direction,
        }


@dataclass
class VolumeForecast:
    horizon_hours: int
    current_sim_time: float
    forecast_by_item_type: Dict[str, ForecastSeries] = field(default_factory=dict)
    total_expected_shipments: float = 0.0
    total_expected_cbm: float = 0.0
    peak_hour: int = 10                       # 가장 많은 도착이 예상되는 절대 시간 (0~23)
    congestion_windows: List[Tuple[int, int, str]] = field(default_factory=list)  # (start, end, severity)
    confidence_interval_90: Tuple[float, float] = (0.0, 0.0)  # (lo, hi) 총 화물 수
    smoothed_base_rate: float = 0.0           # 보정된 전체 기본 도착률

    def to_dict(self) -> dict:
        return {
            "horizon_hours": self.horizon_hours,
            "current_sim_time": round(self.current_sim_time, 1),
            "forecast_by_item_type": {
                k: v.to_dict() for k, v in self.forecast_by_item_type.items()
            },
            "total_expected_shipments": round(self.total_expected_shipments, 1),
            "total_expected_cbm": round(self.total_expected_cbm, 3),
            "peak_hour": self.peak_hour,
            "congestion_windows": [
                {"start": s, "end": e, "severity": sev}
                for s, e, sev in self.congestion_windows
            ],
            "confidence_interval_90": {
                "low":  round(self.confidence_interval_90[0], 1),
                "high": round(self.confidence_interval_90[1], 1),
            },
            "smoothed_base_rate": round(self.smoothed_base_rate, 3),
        }


# ---------------------------------------------------------------------------
# 예측기
# ---------------------------------------------------------------------------

class VolumeForecaster:
    """
    향후 N 시간의 물동량(화물 수, CBM)을 예측.

    관측 이력(events)을 받아 지수 평활화로 기본 도착률을 보정한 후
    HOURLY_MULTIPLIER 패턴을 적용해 시간대별 예측값을 산출.
    """

    def __init__(self, arrival_rates: Optional[Dict[str, float]] = None):
        self.base_rates = arrival_rates or DEFAULT_ARRIVAL_RATES.copy()
        self._smoothed_rates: Dict[str, float] = self.base_rates.copy()

    def forecast(
        self,
        observation: dict,
        horizon_hours: int = 24,
        recent_events: Optional[List[dict]] = None,
    ) -> VolumeForecast:
        """
        Parameters
        ----------
        observation : dict
            시뮬레이터 observation dict (current_time 사용)
        horizon_hours : int
            예측 기간 (기본 24시간)
        recent_events : Optional[List[dict]]
            최근 SHIPMENT_ARRIVAL 이벤트 목록.
            각 dict에 "item_type", "time" 키 포함.
            None이면 기본 도착률 사용.

        Returns
        -------
        VolumeForecast
        """
        current_time = observation.get("current_time", 0.0)

        # 최근 관측으로 도착률 보정
        if recent_events:
            self._update_rates_from_events(recent_events, current_time)

        total_base = sum(self._smoothed_rates.values())

        # 아이템 유형별 예측 시리즈 생성
        forecast_by_type: Dict[str, ForecastSeries] = {}
        total_shipments = 0.0
        total_cbm = 0.0

        hourly_totals: List[float] = []  # 전체 합계 (혼잡도 계산용)

        for item_type, base_rate in self._smoothed_rates.items():
            hourly_counts = []
            hourly_cbms = []
            for h in range(horizon_hours):
                abs_hour = current_time + h
                multiplier = HOURLY_MULTIPLIER[int(abs_hour) % 24]
                expected_count = base_rate * multiplier
                expected_cbm = expected_count * _MEDIAN_CBM.get(item_type, 0.5)
                hourly_counts.append(expected_count)
                hourly_cbms.append(expected_cbm)

            cum_cbm = sum(hourly_cbms)
            trend = _compute_trend(hourly_counts)

            forecast_by_type[item_type] = ForecastSeries(
                item_type=item_type,
                hourly_expected_count=hourly_counts,
                hourly_expected_cbm=hourly_cbms,
                cumulative_expected_cbm=cum_cbm,
                trend_direction=trend,
            )
            total_shipments += sum(hourly_counts)
            total_cbm += cum_cbm

            # 시간별 합계 누적
            for h, cnt in enumerate(hourly_counts):
                if h < len(hourly_totals):
                    hourly_totals[h] += cnt
                else:
                    hourly_totals.append(cnt)

        # 피크 시간 (예측 기간 내 가장 높은 시간대, 절대 시각 → 0~23시 변환)
        if hourly_totals:
            peak_offset = hourly_totals.index(max(hourly_totals))
            peak_hour = int(current_time + peak_offset) % 24
        else:
            peak_hour = 10

        # 혼잡 구간 탐지 (평균의 1.5배 초과)
        avg_hourly = total_shipments / horizon_hours if horizon_hours > 0 else 0
        congestion = _find_congestion_windows(hourly_totals, avg_hourly, current_time)

        # 90% 신뢰 구간 (Poisson: 평균 ± 1.645 * sqrt(평균))
        lo = max(0.0, total_shipments - 1.645 * math.sqrt(max(0, total_shipments)))
        hi = total_shipments + 1.645 * math.sqrt(max(0, total_shipments))

        return VolumeForecast(
            horizon_hours=horizon_hours,
            current_sim_time=current_time,
            forecast_by_item_type=forecast_by_type,
            total_expected_shipments=total_shipments,
            total_expected_cbm=total_cbm,
            peak_hour=peak_hour,
            congestion_windows=congestion,
            confidence_interval_90=(lo, hi),
            smoothed_base_rate=total_base,
        )

    def _update_rates_from_events(
        self,
        events: List[dict],
        current_time: float,
        lookback_hours: float = 12.0,
    ) -> None:
        """최근 N 시간의 도착 이벤트로 기본 도착률 지수 평활화 업데이트."""
        cutoff = current_time - lookback_hours
        recent = [e for e in events if e.get("time", 0) >= cutoff and e.get("type") == "SHIPMENT_ARRIVAL"]

        if not recent:
            return

        # 아이템 유형별 관측 도착률 계산
        counts: Dict[str, int] = {}
        for e in recent:
            item_type = e.get("item_type", "ELECTRONICS")
            counts[item_type] = counts.get(item_type, 0) + 1

        observed_rates: Dict[str, float] = {
            k: v / lookback_hours for k, v in counts.items()
        }

        # 지수 평활화: α × observed + (1-α) × prior
        for item_type in self._smoothed_rates:
            if item_type in observed_rates:
                prior = self._smoothed_rates[item_type]
                observed = observed_rates[item_type]
                self._smoothed_rates[item_type] = _ALPHA * observed + (1 - _ALPHA) * prior


# ---------------------------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------------------------

def _compute_trend(hourly: List[float]) -> str:
    """후반부 vs 전반부 평균 비교로 추세 결정."""
    if len(hourly) < 2:
        return "stable"
    half = len(hourly) // 2
    first_half_avg = sum(hourly[:half]) / half if half > 0 else 0
    second_half_avg = sum(hourly[half:]) / (len(hourly) - half)
    ratio = second_half_avg / first_half_avg if first_half_avg > 0 else 1.0
    if ratio > 1.10:
        return "increasing"
    elif ratio < 0.90:
        return "decreasing"
    return "stable"


def _find_congestion_windows(
    hourly_totals: List[float],
    avg: float,
    base_time: float,
    threshold_ratio: float = 1.5,
) -> List[Tuple[int, int, str]]:
    """평균의 threshold_ratio 배 초과하는 연속 구간 탐지."""
    windows = []
    threshold_high = avg * threshold_ratio
    threshold_medium = avg * 1.2

    in_window = False
    start = 0
    for h, count in enumerate(hourly_totals):
        abs_h = int(base_time + h)
        if count >= threshold_high and not in_window:
            in_window = True
            start = abs_h
            severity = "HIGH"
        elif count >= threshold_medium and not in_window:
            in_window = True
            start = abs_h
            severity = "MEDIUM"
        elif count < threshold_medium and in_window:
            windows.append((start, abs_h - 1, severity))
            in_window = False

    if in_window:
        windows.append((start, int(base_time + len(hourly_totals) - 1), "MEDIUM"))

    return windows
