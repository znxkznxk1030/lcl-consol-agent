"""
hapag_spec.py
=============
Hapag-Lloyd Container Specification 검증 모듈.

공식 스펙 출처: Hapag-Lloyd Container Specification Guide (2023)
https://www.hapag-lloyd.com/en/products-services/cargo/dry-cargo/container-types.html

지원 컨테이너 타입:
  20GP  — 20ft General Purpose
  40GP  — 40ft General Purpose
  40HC  — 40ft High Cube
  45HC  — 45ft High Cube
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Hapag-Lloyd 공식 컨테이너 스펙
# ---------------------------------------------------------------------------

HAPAG_SPECS: Dict[str, dict] = {
    "20GP": {
        "internal_length_mm": 5898,
        "internal_width_mm":  2352,
        "internal_height_mm": 2393,
        "door_width_mm":      2340,
        "door_height_mm":     2280,
        "max_payload_kg":     28_230,
        "tare_weight_kg":      2_230,
        "max_gross_kg":       30_480,
        "max_stack_weight_kg":96_000,
        "internal_volume_cbm": 33.2,
        "floor_load_kg_per_m2": 3_050,
    },
    "40GP": {
        "internal_length_mm": 12_032,
        "internal_width_mm":   2_352,
        "internal_height_mm":  2_393,
        "door_width_mm":       2_340,
        "door_height_mm":      2_280,
        "max_payload_kg":     26_750,
        "tare_weight_kg":      3_750,
        "max_gross_kg":       30_480,
        "max_stack_weight_kg":96_000,
        "internal_volume_cbm": 67.7,
        "floor_load_kg_per_m2": 3_050,
    },
    "40HC": {
        "internal_length_mm": 12_032,
        "internal_width_mm":   2_352,
        "internal_height_mm":  2_698,
        "door_width_mm":       2_340,
        "door_height_mm":      2_585,
        "max_payload_kg":     26_460,
        "tare_weight_kg":      3_940,
        "max_gross_kg":       30_480,
        "max_stack_weight_kg":96_000,
        "internal_volume_cbm": 76.4,
        "floor_load_kg_per_m2": 3_050,
    },
    "45HC": {
        "internal_length_mm": 13_556,
        "internal_width_mm":   2_352,
        "internal_height_mm":  2_698,
        "door_width_mm":       2_340,
        "door_height_mm":      2_585,
        "max_payload_kg":     27_700,
        "tare_weight_kg":      4_800,
        "max_gross_kg":       32_500,
        "max_stack_weight_kg":96_000,
        "internal_volume_cbm": 86.1,
        "floor_load_kg_per_m2": 3_050,
    },
}

# 컨테이너 크기 순서 (최소→최대): 최소 적합 컨테이너 추천에 사용
_CONTAINER_ORDER = ["20GP", "40GP", "40HC", "45HC"]

# 5% dunnage/lashing allowance
_DUNNAGE_FACTOR = 0.95


# ---------------------------------------------------------------------------
# 결과 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class SpecCheckResult:
    container_type: str
    compliant: bool
    violations: List[str] = field(default_factory=list)
    weight_utilization_pct: float = 0.0
    volume_utilization_pct: float = 0.0
    recommended_container: str = ""   # 수용 가능한 최소 컨테이너
    warnings: List[str] = field(default_factory=list)   # ≥90% 경고
    total_weight_kg: float = 0.0
    total_cbm: float = 0.0

    def to_dict(self) -> dict:
        return {
            "container_type": self.container_type,
            "compliant": self.compliant,
            "violations": self.violations,
            "weight_utilization_pct": round(self.weight_utilization_pct, 1),
            "volume_utilization_pct": round(self.volume_utilization_pct, 1),
            "recommended_container": self.recommended_container,
            "warnings": self.warnings,
            "total_weight_kg": round(self.total_weight_kg, 1),
            "total_cbm": round(self.total_cbm, 3),
        }


# ---------------------------------------------------------------------------
# 스펙 검증기
# ---------------------------------------------------------------------------

class HapagSpecChecker:
    """
    제안된 MBL 그룹(화물 목록)이 Hapag-Lloyd 컨테이너 스펙을 만족하는지 검증.
    """

    def check(
        self,
        shipments: List[dict],
        container_type: str = "40GP",
    ) -> SpecCheckResult:
        """
        Parameters
        ----------
        shipments : List[dict]
            observation["buffer"]["shipments"] 에서 필터링된 MBL 후보 화물 목록.
            각 dict는 최소한 "cbm", "weight", "cargo_category", "shipment_id" 키를 가져야 함.
            선택적으로 "length_cm", "width_cm", "height_cm" 포함 가능.
        container_type : str
            검증할 컨테이너 타입 (20GP / 40GP / 40HC / 45HC)

        Returns
        -------
        SpecCheckResult
        """
        if container_type not in HAPAG_SPECS:
            container_type = "40GP"

        spec = HAPAG_SPECS[container_type]
        violations: List[str] = []
        warnings: List[str] = []

        # 합계 계산
        total_cbm = sum(s.get("effective_cbm", s.get("cbm", 0.0)) for s in shipments)
        total_weight = sum(s.get("weight", 0.0) for s in shipments)

        # 1) 중량 검사
        max_payload = spec["max_payload_kg"]
        w_util = (total_weight / max_payload * 100) if max_payload > 0 else 0.0
        if total_weight > max_payload:
            violations.append(
                f"payload_exceeded: {total_weight:.0f}kg > {max_payload}kg "
                f"(초과 {total_weight - max_payload:.0f}kg)"
            )
        elif w_util >= 90.0:
            warnings.append(f"payload_near_limit: {w_util:.1f}% ({total_weight:.0f}/{max_payload}kg)")

        # 2) 용적 검사 (dunnage 고려)
        usable_cbm = spec["internal_volume_cbm"] * _DUNNAGE_FACTOR
        v_util = (total_cbm / usable_cbm * 100) if usable_cbm > 0 else 0.0
        if total_cbm > usable_cbm:
            violations.append(
                f"volume_exceeded: {total_cbm:.3f}CBM > {usable_cbm:.2f}CBM "
                f"(usable {_DUNNAGE_FACTOR*100:.0f}% of {spec['internal_volume_cbm']}CBM)"
            )
        elif v_util >= 90.0:
            warnings.append(f"volume_near_limit: {v_util:.1f}% ({total_cbm:.3f}/{usable_cbm:.2f}CBM)")

        # 3) OVERSIZED 도어 개구부 검사
        door_w = spec["door_width_mm"]
        door_h = spec["door_height_mm"]
        for s in shipments:
            if s.get("cargo_category") == "OVERSIZED":
                w_cm = s.get("width_cm")
                h_cm = s.get("height_cm")
                if w_cm and w_cm * 10 > door_w:
                    violations.append(
                        f"{s['shipment_id']}: width {w_cm}cm > door {door_w/10:.0f}cm"
                    )
                if h_cm and h_cm * 10 > door_h:
                    violations.append(
                        f"{s['shipment_id']}: height {h_cm}cm > door {door_h/10:.0f}cm"
                    )

        # 4) 바닥 하중 추정 (총 중량 / 바닥 면적)
        floor_area_m2 = (spec["internal_length_mm"] / 1000) * (spec["internal_width_mm"] / 1000)
        floor_load = total_weight / floor_area_m2 if floor_area_m2 > 0 else 0.0
        max_floor = spec["floor_load_kg_per_m2"]
        if floor_load > max_floor:
            violations.append(
                f"floor_load_exceeded: {floor_load:.0f}kg/m² > {max_floor}kg/m²"
            )
        elif floor_load >= max_floor * 0.90:
            warnings.append(f"floor_load_near_limit: {floor_load:.0f}/{max_floor}kg/m²")

        # 5) 최적 컨테이너 추천 (위반 없이 수용 가능한 최소 타입)
        recommended = _find_minimum_container(total_cbm, total_weight)

        return SpecCheckResult(
            container_type=container_type,
            compliant=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            weight_utilization_pct=w_util,
            volume_utilization_pct=v_util,
            recommended_container=recommended,
            total_weight_kg=total_weight,
            total_cbm=total_cbm,
        )

    def check_multiple(
        self,
        mbl_groups: List[List[dict]],
        container_type: str = "40GP",
    ) -> List[SpecCheckResult]:
        """여러 MBL 그룹 일괄 검증."""
        return [self.check(group, container_type) for group in mbl_groups]


def _find_minimum_container(total_cbm: float, total_weight: float) -> str:
    """CBM과 중량을 모두 수용 가능한 최소 컨테이너 타입 반환."""
    for ct in _CONTAINER_ORDER:
        spec = HAPAG_SPECS[ct]
        usable = spec["internal_volume_cbm"] * _DUNNAGE_FACTOR
        if total_cbm <= usable and total_weight <= spec["max_payload_kg"]:
            return ct
    return "45HC"  # 최대 컨테이너도 초과하면 경고


def get_spec(container_type: str) -> Optional[dict]:
    return HAPAG_SPECS.get(container_type)
