"""
volume_model.py
===============
공용 부피 모델.

Olist 기반 개별 상품/소포 치수로 계산한 raw CBM은 실제 LCL 점유 공간보다 작게
나오는 경향이 있으므로, 패키징/취급 여유 공간과 카테고리별 dead space를 반영한
effective CBM과 usable container CBM을 계산한다.
"""

from __future__ import annotations

from typing import Mapping, Any


# 개별 상품 부피 -> 실제 혼적 점유 부피 보정
BASE_PACKAGING_MULTIPLIER = 1.12
CATEGORY_SPACE_MULTIPLIERS = {
    "GENERAL": 1.00,
    "FOOD": 1.04,
    "HAZMAT": 1.08,
    "FRAGILE": 1.18,
    "OVERSIZED": 1.15,
}

# 컨테이너 내부 총용적 대비 실사용 가능 비율
USABLE_CONTAINER_RATIO = 0.90


def shipment_cbm_from_dict(shipment: Mapping[str, Any]) -> float:
    """
    화물 raw CBM.

    우선순위:
    1. 명시된 cbm
    2. 치수(cm)에서 역산
    """
    cbm = shipment.get("cbm")
    if cbm is not None:
        return round(float(cbm), 6)

    length_cm = shipment.get("length_cm")
    width_cm = shipment.get("width_cm")
    height_cm = shipment.get("height_cm")
    if length_cm and width_cm and height_cm:
        return round(float(length_cm) * float(width_cm) * float(height_cm) / 1_000_000, 6)
    return 0.0


def effective_cbm_from_raw(cbm: float, cargo_category: str | None = None) -> float:
    base = max(float(cbm), 0.0)
    cat_multiplier = CATEGORY_SPACE_MULTIPLIERS.get(cargo_category or "GENERAL", 1.0)
    return round(base * BASE_PACKAGING_MULTIPLIER * cat_multiplier, 6)


def effective_cbm_from_dict(shipment: Mapping[str, Any]) -> float:
    return effective_cbm_from_raw(
        shipment_cbm_from_dict(shipment),
        shipment.get("cargo_category"),
    )


def usable_container_cbm(max_cbm_per_mbl: float) -> float:
    return round(float(max_cbm_per_mbl) * USABLE_CONTAINER_RATIO, 3)
