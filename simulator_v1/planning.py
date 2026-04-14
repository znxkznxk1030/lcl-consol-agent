"""
planning.py
===========
MBL 그룹핑과 3D loading plan 생성을 위한 공용 헬퍼.
에이전트/시뮬레이터/서버가 동일한 plan 포맷을 사용하도록 맞춘다.
"""

from __future__ import annotations

import uuid
from typing import Iterable, List, Optional

_ASPECT_RATIOS = {
    "ELECTRONICS": (2.0, 1.5, 1.0),
    "CLOTHING": (2.5, 2.0, 0.5),
    "COSMETICS": (1.5, 1.0, 1.0),
    "FOOD_PRODUCTS": (2.0, 1.5, 1.5),
    "AUTO_PARTS": (3.0, 2.0, 1.5),
    "CHEMICALS": (2.0, 1.5, 2.0),
    "FURNITURE": (4.0, 2.0, 2.0),
    "MACHINERY": (3.0, 2.0, 2.0),
}

_CONTAINER_DIMS = {
    "20GP": (5898, 2352, 2393),
    "40GP": (12032, 2352, 2393),
    "40HC": (12032, 2352, 2698),
    "45HC": (13556, 2352, 2698),
}


def infer_container_type(max_cbm_per_mbl: float) -> str:
    """명목 CBM 기준으로 컨테이너 타입을 추정한다."""
    if max_cbm_per_mbl <= 34:
        return "20GP"
    if max_cbm_per_mbl <= 69:
        return "40GP"
    if max_cbm_per_mbl <= 78:
        return "40HC"
    return "45HC"


def shipment_ids_from_plan(plan: object) -> List[str]:
    """하위호환 포함: MBL plan 또는 shipment_id 배열에서 shipment IDs 추출."""
    if isinstance(plan, dict):
        shipment_ids = plan.get("shipment_ids", [])
        if isinstance(shipment_ids, list):
            return [sid for sid in shipment_ids if isinstance(sid, str)]
        return []
    if isinstance(plan, list):
        return [sid for sid in plan if isinstance(sid, str)]
    return []


def normalize_mbl_plans(mbls: Optional[Iterable[object]]) -> List[dict]:
    """
    old format(List[List[str]])과 new format(List[dict])을 공용 plan dict로 정규화.
    """
    normalized: List[dict] = []
    for item in mbls or []:
        shipment_ids = shipment_ids_from_plan(item)
        if not shipment_ids:
            continue

        if isinstance(item, dict):
            normalized.append({
                "plan_id": item.get("plan_id") or f"PLAN-{uuid.uuid4().hex[:8].upper()}",
                "shipment_ids": shipment_ids,
                "container_type": item.get("container_type"),
                "loading_plan": item.get("loading_plan"),
                # 슬롯 기반 할당 필드 (없으면 기본값 유지)
                "close": item.get("close", True),
                "slot_id": item.get("slot_id"),
            })
        else:
            normalized.append({
                "plan_id": f"PLAN-{uuid.uuid4().hex[:8].upper()}",
                "shipment_ids": shipment_ids,
                "container_type": None,
                "loading_plan": None,
                "close": True,
                "slot_id": None,
            })
    return normalized


def _shipment_to_packer_payload(shipment: dict) -> dict:
    """
    BinPacker3D 입력용 shipment payload를 만든다.
    차원이 있어도 Olist 개별 상품 치수와 괴리가 클 수 있으므로 cbm/effective_cbm 중심으로 넘긴다.
    """
    payload = {
        "shipment_id": shipment["shipment_id"],
        "cargo_category": shipment.get("cargo_category", "GENERAL"),
        "item_type": shipment.get("item_type", "ELECTRONICS"),
        "cbm": shipment.get("effective_cbm", shipment.get("cbm", 0.0)),
        "weight": shipment.get("weight", 0.0),
    }
    if shipment.get("effective_cbm") is not None:
        payload["effective_cbm"] = shipment["effective_cbm"]
    return payload


def _resolve_dimensions_mm(shipment: dict) -> tuple[int, int, int]:
    l_cm = shipment.get("length_cm")
    w_cm = shipment.get("width_cm")
    h_cm = shipment.get("height_cm")
    if l_cm and w_cm and h_cm:
        return max(100, int(l_cm * 10)), max(100, int(w_cm * 10)), max(100, int(h_cm * 10))

    item_type = shipment.get("item_type", "ELECTRONICS")
    cbm = max(shipment.get("effective_cbm", shipment.get("cbm", 0.05)), 0.0001)
    ratio_l, ratio_w, ratio_h = _ASPECT_RATIOS.get(item_type, (2.0, 1.5, 1.0))
    volume_mm3 = cbm * 1_000_000_000
    scale = (volume_mm3 / (ratio_l * ratio_w * ratio_h)) ** (1 / 3)
    return (
        max(100, int(ratio_l * scale)),
        max(100, int(ratio_w * scale)),
        max(100, int(ratio_h * scale)),
    )


def _fast_pack_positions(
    shipments: List[dict],
    container_type: str,
) -> tuple[list[dict], list[str], float]:
    """
    빠른 shelf 기반 packer.
    decide 경로에서 수천 건 입력도 처리할 수 있도록 O(n)에 가깝게 유지한다.
    """
    c_l, c_w, c_h = _CONTAINER_DIMS.get(container_type, _CONTAINER_DIMS["40GP"])
    gap = 10
    items = []
    total_volume = 0.0
    for shipment in shipments:
        l_mm, w_mm, h_mm = _resolve_dimensions_mm(shipment)
        total_volume += l_mm * w_mm * h_mm
        items.append({
            "shipment_id": shipment["shipment_id"],
            "cargo_category": shipment.get("cargo_category", "GENERAL"),
            "weight": shipment.get("weight", 0.0),
            "dims": (l_mm, w_mm, h_mm),
            "volume": l_mm * w_mm * h_mm,
        })
    items.sort(key=lambda item: item["volume"], reverse=True)

    positions: list[dict] = []
    unplaceable: list[str] = []
    x_mm = 0
    y_mm = 0
    z_mm = 0
    row_width = 0
    row_height = 0
    layer_height = 0

    for item in items:
        dims_options = [item["dims"], (item["dims"][1], item["dims"][0], item["dims"][2])]
        placed = False

        for length_mm, width_mm, height_mm in dims_options:
            next_x = x_mm + length_mm
            next_y = y_mm + width_mm
            next_z = z_mm + height_mm

            if next_x <= c_l and next_y <= c_w and next_z <= c_h:
                positions.append({
                    "shipment_id": item["shipment_id"],
                    "position": {"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm},
                    "dimensions": {
                        "length_mm": length_mm,
                        "width_mm": width_mm,
                        "height_mm": height_mm,
                    },
                    "rotated": (length_mm, width_mm) != item["dims"][:2],
                    "cargo_category": item["cargo_category"],
                    "weight_kg": round(item["weight"], 1),
                })
                x_mm += length_mm + gap
                row_width = max(row_width, width_mm)
                row_height = max(row_height, height_mm)
                layer_height = max(layer_height, row_height)
                placed = True
                break

        if placed:
            continue

        x_mm = 0
        y_mm += row_width + gap
        row_width = 0
        row_height = 0

        for length_mm, width_mm, height_mm in dims_options:
            next_x = x_mm + length_mm
            next_y = y_mm + width_mm
            next_z = z_mm + height_mm

            if next_x <= c_l and next_y <= c_w and next_z <= c_h:
                positions.append({
                    "shipment_id": item["shipment_id"],
                    "position": {"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm},
                    "dimensions": {
                        "length_mm": length_mm,
                        "width_mm": width_mm,
                        "height_mm": height_mm,
                    },
                    "rotated": (length_mm, width_mm) != item["dims"][:2],
                    "cargo_category": item["cargo_category"],
                    "weight_kg": round(item["weight"], 1),
                })
                x_mm += length_mm + gap
                row_width = max(row_width, width_mm)
                row_height = max(row_height, height_mm)
                layer_height = max(layer_height, row_height)
                placed = True
                break

        if placed:
            continue

        x_mm = 0
        y_mm = 0
        z_mm += layer_height + gap
        row_width = 0
        row_height = 0
        layer_height = 0

        for length_mm, width_mm, height_mm in dims_options:
            next_x = x_mm + length_mm
            next_y = y_mm + width_mm
            next_z = z_mm + height_mm

            if next_x <= c_l and next_y <= c_w and next_z <= c_h:
                positions.append({
                    "shipment_id": item["shipment_id"],
                    "position": {"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm},
                    "dimensions": {
                        "length_mm": length_mm,
                        "width_mm": width_mm,
                        "height_mm": height_mm,
                    },
                    "rotated": (length_mm, width_mm) != item["dims"][:2],
                    "cargo_category": item["cargo_category"],
                    "weight_kg": round(item["weight"], 1),
                })
                x_mm += length_mm + gap
                row_width = max(row_width, width_mm)
                row_height = max(row_height, height_mm)
                layer_height = max(layer_height, row_height)
                placed = True
                break

        if not placed:
            unplaceable.append(item["shipment_id"])

    container_volume = c_l * c_w * c_h
    utilization_pct = (total_volume / container_volume * 100.0) if container_volume > 0 else 0.0
    return positions, unplaceable, utilization_pct


def build_loading_plan(
    shipments: List[dict],
    max_cbm_per_mbl: float,
    mbl_id: Optional[str] = None,
    container_type: Optional[str] = None,
) -> dict:
    """현재 shipment 그룹에 대한 빠른 deterministic loading plan을 생성한다."""
    resolved_container_type = container_type or infer_container_type(max_cbm_per_mbl)
    resolved_mbl_id = mbl_id or f"PLAN-{uuid.uuid4().hex[:8].upper()}"
    payload = [_shipment_to_packer_payload(shipment) for shipment in shipments]
    positions, unplaceable, utilization_pct = _fast_pack_positions(payload, resolved_container_type)
    total_weight = sum(float(shipment.get("weight", 0.0)) for shipment in payload)
    return {
        "container_type": resolved_container_type,
        "mbl_id": resolved_mbl_id,
        "positions": positions,
        "unplaceable_shipments": unplaceable,
        "volume_utilization_pct": round(utilization_pct, 1),
        "weight_utilization_pct": 0.0,
        "center_of_gravity": {"x_pct": 50.0, "y_pct": 50.0, "z_pct": 30.0},
        "stability_compliant": True,
        "floor_load_max_kg_per_m2": 0.0,
        "ascii_view_top": "",
        "ascii_view_side": "",
        "summary": {
            "shipment_count": len(payload),
            "placed_count": len(positions),
            "unplaceable_count": len(unplaceable),
            "total_weight_kg": round(total_weight, 1),
        },
    }


def build_mbl_plans_from_groupings(
    groupings: List[List[str]],
    shipments: List[dict],
    max_cbm_per_mbl: float,
) -> List[dict]:
    """shipment_id 그룹핑을 좌표 포함 MBL plan 객체 목록으로 변환한다."""
    shipment_map = {shipment["shipment_id"]: shipment for shipment in shipments}
    container_type = infer_container_type(max_cbm_per_mbl)
    plans: List[dict] = []

    for grouping in groupings:
        shipment_ids = [sid for sid in grouping if sid in shipment_map]
        if not shipment_ids:
            continue
        grouped_shipments = [shipment_map[sid] for sid in shipment_ids]
        plan_id = f"PLAN-{uuid.uuid4().hex[:8].upper()}"
        loading_plan = build_loading_plan(
            grouped_shipments,
            max_cbm_per_mbl=max_cbm_per_mbl,
            mbl_id=plan_id,
            container_type=container_type,
        )
        plans.append({
            "plan_id": plan_id,
            "shipment_ids": shipment_ids,
            "container_type": container_type,
            "loading_plan": loading_plan,
        })

    return plans
