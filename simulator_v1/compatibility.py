"""
compatibility.py
================
화물 카테고리 간 혼적(混積) 호환성 규칙

핵심 규칙:
- HAZMAT(위험물) + FOOD / FRAGILE  → 혼적 불가
- OVERSIZED(특대형)                → 단독 MBL만 허용 (다른 모든 카테고리와 불가)
- FRAGILE(파손주의)                → 유효 CBM = 실제 CBM × 1.3 (완충 공간 필요)

Agent가 혼적 불가 화물을 한 MBL에 담으면:
  → Simulator가 자동으로 호환 가능한 서브 그룹으로 분리
  → 분리된 그룹마다 별도 MBL 생성 (고정비 증가)
  → COMPATIBILITY_VIOLATION 이벤트 로깅
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, List, Set

if TYPE_CHECKING:
    from .entities import Shipment


# ---------------------------------------------------------------------------
# CargoCategory
# ---------------------------------------------------------------------------

class CargoCategory(Enum):
    GENERAL   = "GENERAL"    # 일반 화물
    HAZMAT    = "HAZMAT"     # 위험물 (Hazardous Material)
    FOOD      = "FOOD"       # 식품 (온도 민감, 오염 주의)
    FRAGILE   = "FRAGILE"    # 파손주의 (완충 공간 필요)
    OVERSIZED = "OVERSIZED"  # 특대형 (단독 MBL 필수)


# ---------------------------------------------------------------------------
# 혼적 불가 쌍 (순서 무관)
# ---------------------------------------------------------------------------

INCOMPATIBLE_PAIRS: Set[frozenset] = {
    frozenset({CargoCategory.HAZMAT,    CargoCategory.FOOD}),
    frozenset({CargoCategory.HAZMAT,    CargoCategory.FRAGILE}),
    frozenset({CargoCategory.OVERSIZED, CargoCategory.GENERAL}),
    frozenset({CargoCategory.OVERSIZED, CargoCategory.HAZMAT}),
    frozenset({CargoCategory.OVERSIZED, CargoCategory.FOOD}),
    frozenset({CargoCategory.OVERSIZED, CargoCategory.FRAGILE}),
    frozenset({CargoCategory.OVERSIZED, CargoCategory.OVERSIZED}),  # OVERSIZED끼리도 불가
}

# FRAGILE 화물의 유효 CBM 배수 (패딩·완충재 공간 포함)
FRAGILE_CBM_MULTIPLIER: float = 1.3


# ---------------------------------------------------------------------------
# 호환성 조회 함수
# ---------------------------------------------------------------------------

def is_compatible_pair(a: CargoCategory, b: CargoCategory) -> bool:
    """두 카테고리가 같은 MBL에 실릴 수 있으면 True."""
    return frozenset({a, b}) not in INCOMPATIBLE_PAIRS


def is_compatible_group(categories: List[CargoCategory]) -> bool:
    """카테고리 목록 전체가 서로 호환 가능하면 True."""
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            if not is_compatible_pair(categories[i], categories[j]):
                return False
    return True


def count_violation_pairs(shipments: List["Shipment"]) -> int:
    """혼적 불가 쌍의 수를 반환 (0이면 호환 가능)."""
    cats = [s.cargo_category for s in shipments]
    count = 0
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            if not is_compatible_pair(cats[i], cats[j]):
                count += 1
    return count


# ---------------------------------------------------------------------------
# 자동 분리
# ---------------------------------------------------------------------------

def split_into_compatible_groups(shipments: List["Shipment"]) -> List[List["Shipment"]]:
    """
    혼적 불가 화물이 섞인 리스트를 호환 가능한 서브 그룹 리스트로 분리.

    알고리즘:
    1. OVERSIZED는 각각 단독 그룹으로 분리.
    2. 나머지는 Greedy 방식으로 그룹핑:
       - 현재 그룹의 첫 화물 카테고리 집합과 모두 호환되면 합류
       - 호환 안 되면 다음 라운드(새 그룹)로 이월
    """
    groups: List[List["Shipment"]] = []

    # Step 1: OVERSIZED 분리
    oversized = [s for s in shipments if s.cargo_category == CargoCategory.OVERSIZED]
    remaining = [s for s in shipments if s.cargo_category != CargoCategory.OVERSIZED]
    for s in oversized:
        groups.append([s])

    # Step 2: 나머지 greedy 그룹핑
    while remaining:
        group: List["Shipment"] = [remaining[0]]
        group_cats: Set[CargoCategory] = {remaining[0].cargo_category}
        leftover: List["Shipment"] = []

        for s in remaining[1:]:
            if all(is_compatible_pair(s.cargo_category, c) for c in group_cats):
                group.append(s)
                group_cats.add(s.cargo_category)
            else:
                leftover.append(s)

        groups.append(group)
        remaining = leftover

    return groups
