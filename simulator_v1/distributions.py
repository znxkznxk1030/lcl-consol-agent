"""
distributions.py
================
현실 LCL 통계 기반 샘플링 함수 모음.

아이템 유형:
  ELECTRONICS  전자제품    소형·파손주의·고빈도
  CLOTHING     의류/섬유   소형·일반·고빈도
  COSMETICS    화장품      소형·일반/위험물·중빈도
  FOOD_PRODUCTS 식품/음료  중형·식품카테고리·중빈도
  AUTO_PARTS   자동차부품  중형·일반·중빈도
  CHEMICALS    화학물질    중형·위험물·저빈도
  FURNITURE    가구/인테리어 대형·파손주의·저빈도
  MACHINERY    기계/장비   대형·특대형·저빈도

참고:
- CBM: log-normal (Hummels & Schaur 2013, LCL freight size distribution)
- 무게: stowage factor (IATA ULD Regulations, Freightos 업종 평균)
- 패키지 수: Negative Binomial (Gamma-Poisson mixture)
- 카테고리 비율: Freightos/IATA 실측 LCL 화물 통계
- 도착률 시간 패턴: 해상 LCL 화물 예약 시간대 분포
"""

from __future__ import annotations

import math
import random
from typing import Dict, Tuple

from .compatibility import CargoCategory


# ---------------------------------------------------------------------------
# 1. CBM — Log-normal 파라미터
# ---------------------------------------------------------------------------
# (mu_log, sigma_log, clamp_lo, clamp_hi)
_CBM_PARAMS: Dict[str, Tuple[float, float, float, float]] = {
    "ELECTRONICS":    (-1.60, 0.40, 0.03, 0.60),   # 중앙값 ~0.20 CBM
    "CLOTHING":       (-1.20, 0.45, 0.05, 0.90),   # 중앙값 ~0.30 CBM
    "COSMETICS":      (-1.80, 0.35, 0.02, 0.40),   # 중앙값 ~0.17 CBM
    "FOOD_PRODUCTS":  (-0.50, 0.45, 0.15, 2.00),   # 중앙값 ~0.61 CBM
    "AUTO_PARTS":     ( 0.10, 0.50, 0.30, 4.00),   # 중앙값 ~1.11 CBM
    "CHEMICALS":      (-0.20, 0.40, 0.20, 3.00),   # 중앙값 ~0.82 CBM
    "FURNITURE":      ( 1.20, 0.50, 0.80, 8.00),   # 중앙값 ~3.32 CBM
    "MACHINERY":      ( 1.50, 0.50, 1.50, 12.00),  # 중앙값 ~4.48 CBM
}


def sample_cbm(rng: random.Random, item_type_value: str) -> float:
    mu, sigma, lo, hi = _CBM_PARAMS[item_type_value]
    v = rng.lognormvariate(mu, sigma)
    return round(max(lo, min(hi, v)), 3)


# ---------------------------------------------------------------------------
# 2. 무게 — Stowage factor 기반 (CBM × density)
# ---------------------------------------------------------------------------
# (mean_kg_per_cbm, std_kg_per_cbm)
_STOWAGE_PARAMS: Dict[CargoCategory, Tuple[float, float]] = {
    CargoCategory.GENERAL:   (250.0, 60.0),
    CargoCategory.HAZMAT:    (400.0, 80.0),
    CargoCategory.FOOD:      (300.0, 70.0),
    CargoCategory.FRAGILE:   (150.0, 40.0),
    CargoCategory.OVERSIZED: (200.0, 50.0),
}

# ItemType별 무게 clamp (kg)
_WEIGHT_CLAMP: Dict[str, Tuple[float, float]] = {
    "ELECTRONICS":   (0.5,   200.0),
    "CLOTHING":      (1.0,   150.0),
    "COSMETICS":     (0.5,    80.0),
    "FOOD_PRODUCTS": (5.0,   500.0),
    "AUTO_PARTS":    (5.0,  1500.0),
    "CHEMICALS":     (5.0,  1000.0),
    "FURNITURE":     (5.0,   800.0),
    "MACHINERY":     (20.0, 5000.0),
}


def sample_weight(
    rng: random.Random,
    cbm: float,
    cargo_category: CargoCategory,
    item_type_value: str,
) -> float:
    mean, std = _STOWAGE_PARAMS[cargo_category]
    density = max(50.0, rng.gauss(mean, std))
    raw = cbm * density
    lo, hi = _WEIGHT_CLAMP[item_type_value]
    return round(max(lo, min(hi, raw)), 2)


# ---------------------------------------------------------------------------
# 3. 패키지 수 — Negative Binomial (Gamma-Poisson mixture)
# ---------------------------------------------------------------------------
# (r, p, clamp_lo, clamp_hi)  — NB(r, p), 평균 = r*(1-p)/p
_PKG_PARAMS: Dict[str, Tuple[float, float, int, int]] = {
    "ELECTRONICS":   (2.0, 0.50, 1,  10),  # 평균 ~2.0
    "CLOTHING":      (3.0, 0.45, 1,  20),  # 평균 ~3.7 (묶음 배송)
    "COSMETICS":     (2.0, 0.55, 1,  12),  # 평균 ~1.6
    "FOOD_PRODUCTS": (2.0, 0.40, 1,  15),  # 평균 ~3.0
    "AUTO_PARTS":    (2.0, 0.40, 1,  12),  # 평균 ~3.0
    "CHEMICALS":     (1.0, 0.50, 1,   6),  # 평균 ~1.0
    "FURNITURE":     (1.0, 0.60, 1,   8),  # 평균 ~0.7 → 1
    "MACHINERY":     (1.0, 0.65, 1,   5),  # 평균 ~0.5 → 1
}


def _poisson_from_lambda(rng: random.Random, lam: float) -> int:
    """Knuth 알고리즘으로 Poisson(lam) 샘플링."""
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def sample_packages(rng: random.Random, item_type_value: str) -> int:
    r, p, lo, hi = _PKG_PARAMS[item_type_value]
    scale = (1.0 - p) / p
    lam = rng.gammavariate(r, scale)
    k = _poisson_from_lambda(rng, lam)
    return max(lo, min(hi, k))


# ---------------------------------------------------------------------------
# 4. CargoCategory 비율 — 아이템 유형별 현실 분포
# ---------------------------------------------------------------------------
# [(category, cumulative_prob)]
_CATEGORY_PROBS: Dict[str, list] = {
    "ELECTRONICS": [
        (CargoCategory.FRAGILE, 0.75),
        (CargoCategory.GENERAL, 1.00),   # +0.25 (내구재 일부)
    ],
    "CLOTHING": [
        (CargoCategory.GENERAL, 0.90),
        (CargoCategory.FRAGILE, 1.00),   # +0.10 (고가 의류)
    ],
    "COSMETICS": [
        (CargoCategory.GENERAL, 0.55),
        (CargoCategory.HAZMAT,  1.00),   # +0.45 (인화성 성분)
    ],
    "FOOD_PRODUCTS": [
        (CargoCategory.FOOD,    0.95),
        (CargoCategory.GENERAL, 1.00),   # +0.05
    ],
    "AUTO_PARTS": [
        (CargoCategory.GENERAL, 0.70),
        (CargoCategory.HAZMAT,  0.90),   # +0.20 (오일류 포함)
        (CargoCategory.FRAGILE, 1.00),   # +0.10 (유리 부품)
    ],
    "CHEMICALS": [
        (CargoCategory.HAZMAT,  0.90),
        (CargoCategory.GENERAL, 1.00),   # +0.10 (비위험 화학품)
    ],
    "FURNITURE": [
        (CargoCategory.FRAGILE,  0.50),
        (CargoCategory.OVERSIZED,0.85),  # +0.35
        (CargoCategory.GENERAL,  1.00),  # +0.15
    ],
    "MACHINERY": [
        (CargoCategory.OVERSIZED,0.80),
        (CargoCategory.GENERAL,  1.00),  # +0.20 (소형 장비)
    ],
}


def sample_category(rng: random.Random, item_type_value: str) -> CargoCategory:
    r = rng.random()
    for category, cum_prob in _CATEGORY_PROBS[item_type_value]:
        if r <= cum_prob:
            return category
    return CargoCategory.GENERAL


# ---------------------------------------------------------------------------
# 5. 시간대별 도착률 multiplier (비균질 포아송용)
# ---------------------------------------------------------------------------
HOURLY_MULTIPLIER: list = [
    0.30, 0.20, 0.10, 0.10, 0.20, 0.50,   # 0-5시  (심야)
    0.80, 1.20, 1.50, 1.80, 1.90, 2.00,   # 6-11시 (오전 피크)
    1.80, 1.70, 1.60, 1.50, 1.40, 1.20,   # 12-17시 (오후)
    1.00, 0.80, 0.60, 0.50, 0.40, 0.30,   # 18-23시 (저녁)
]

_MAX_MULTIPLIER: float = max(HOURLY_MULTIPLIER)  # 2.0


def hourly_rate_multiplier(sim_hour: float) -> float:
    idx = int(sim_hour) % 24
    return HOURLY_MULTIPLIER[idx]


def thinning_arrivals(
    rng: random.Random,
    base_rate: float,
    hour_start: float,
) -> list:
    """
    Lewis-Shedler thinning 알고리즘으로 비균질 포아송 도착 시각 오프셋 목록 반환.
    [hour_start, hour_start+1) 구간.
    """
    rate_max = base_rate * _MAX_MULTIPLIER
    if rate_max <= 0:
        return []

    offsets = []
    t = 0.0
    while True:
        t += rng.expovariate(rate_max)
        if t >= 1.0:
            break
        actual_rate = base_rate * hourly_rate_multiplier(hour_start + t)
        if rng.random() < actual_rate / rate_max:
            offsets.append(t)
    return offsets
