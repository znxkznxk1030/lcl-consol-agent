"""
bin_packer_3d.py
================
3D 컨테이너 적재 최적화 모듈 (Extreme Points 알고리즘).

참고: Crainic, T.G. et al. (2008) "Extreme Point-Based Heuristics for
Three-Dimensional Bin Packing." INFORMS Journal on Computing.

제약 조건:
  - FRAGILE  : 수직 방향 유지, 하단에 비취약 화물 배치 금지 (상단만 허용)
  - HAZMAT   : IMDG Code — 도어 인접 전방 20% 구역 (비상시 접근 용이)
  - FOOD     : HAZMAT 구역과 분리 (후방 80% 구역)
  - 무게 중심 : 컨테이너 길이 40~60% 구간 (IMO 트림 안정성)
  - 바닥 하중 : 3,000 kg/m² 초과 금지
  - 스태킹   : 하중은 박스 면적에 분산 (단순화: 화물 자체 중량 / 바닥 면적)

차원 미기재 화물: ItemType별 평균 종횡비로 CBM에서 역산.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# ItemType별 기본 종횡비 (length : width : height)
# Olist 데이터 기반 평균값 (정규화된 비율)
# ---------------------------------------------------------------------------

_ASPECT_RATIOS: Dict[str, Tuple[float, float, float]] = {
    "ELECTRONICS":    (2.0, 1.5, 1.0),
    "CLOTHING":       (2.5, 2.0, 0.5),
    "COSMETICS":      (1.5, 1.0, 1.0),
    "FOOD_PRODUCTS":  (2.0, 1.5, 1.5),
    "AUTO_PARTS":     (3.0, 2.0, 1.5),
    "CHEMICALS":      (2.0, 1.5, 2.0),
    "FURNITURE":      (4.0, 2.0, 2.0),
    "MACHINERY":      (3.0, 2.0, 2.0),
}

# 컨테이너 앞쪽(도어쪽) HAZMAT 전용 구역 비율
_HAZMAT_ZONE_RATIO = 0.20   # 컨테이너 길이의 앞 20%
# 바닥 하중 한계
_MAX_FLOOR_LOAD_KG_M2 = 3_000.0
# FRAGILE 스태킹 제한 (위에 올릴 수 있는 최대 중량, kg)
_FRAGILE_STACK_LIMIT_KG = 50.0


# ---------------------------------------------------------------------------
# 결과 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class LoadingPosition:
    shipment_id: str
    x_mm: int    # 컨테이너 후방 좌측 하단 원점 기준
    y_mm: int    # 너비 방향
    z_mm: int    # 높이 방향
    length_mm: int
    width_mm: int
    height_mm: int
    rotated: bool = False   # 수평 90° 회전 여부
    cargo_category: str = "GENERAL"
    weight_kg: float = 0.0

    def to_dict(self) -> dict:
        return {
            "shipment_id": self.shipment_id,
            "position": {"x_mm": self.x_mm, "y_mm": self.y_mm, "z_mm": self.z_mm},
            "dimensions": {
                "length_mm": self.length_mm,
                "width_mm": self.width_mm,
                "height_mm": self.height_mm,
            },
            "rotated": self.rotated,
            "cargo_category": self.cargo_category,
            "weight_kg": round(self.weight_kg, 1),
        }


@dataclass
class LoadingPlan:
    container_type: str
    mbl_id: str
    positions: List[LoadingPosition] = field(default_factory=list)
    unplaceable_shipments: List[str] = field(default_factory=list)
    volume_utilization_pct: float = 0.0
    weight_utilization_pct: float = 0.0
    cog_x_pct: float = 50.0    # 무게 중심 X 위치 (컨테이너 길이 %)
    cog_y_pct: float = 50.0
    cog_z_pct: float = 30.0
    stability_compliant: bool = True
    floor_load_max_kg_per_m2: float = 0.0
    ascii_view_top: str = ""
    ascii_view_side: str = ""

    def to_dict(self) -> dict:
        return {
            "container_type": self.container_type,
            "mbl_id": self.mbl_id,
            "positions": [p.to_dict() for p in self.positions],
            "unplaceable_shipments": self.unplaceable_shipments,
            "volume_utilization_pct": round(self.volume_utilization_pct, 1),
            "weight_utilization_pct": round(self.weight_utilization_pct, 1),
            "center_of_gravity": {
                "x_pct": round(self.cog_x_pct, 1),
                "y_pct": round(self.cog_y_pct, 1),
                "z_pct": round(self.cog_z_pct, 1),
            },
            "stability_compliant": self.stability_compliant,
            "floor_load_max_kg_per_m2": round(self.floor_load_max_kg_per_m2, 1),
            "ascii_view_top": self.ascii_view_top,
            "ascii_view_side": self.ascii_view_side,
        }


# ---------------------------------------------------------------------------
# Hapag-Lloyd 컨테이너 내부 치수 (mm) — hapag_spec.py와 동기화
# ---------------------------------------------------------------------------

_CONTAINER_DIMS: Dict[str, Tuple[int, int, int]] = {
    "20GP": (5898, 2352, 2393),
    "40GP": (12032, 2352, 2393),
    "40HC": (12032, 2352, 2698),
    "45HC": (13556, 2352, 2698),
}


# ---------------------------------------------------------------------------
# 3D 적재 최적화기
# ---------------------------------------------------------------------------

class BinPacker3D:
    """
    Extreme Points 기반 3D 컨테이너 적재 최적화.
    화물 목록과 컨테이너 타입을 받아 LoadingPlan을 반환.
    """

    def pack(
        self,
        shipments: List[dict],
        container_type: str = "40GP",
        mbl_id: str = "MBL-000",
    ) -> LoadingPlan:
        """
        Parameters
        ----------
        shipments : List[dict]
            화물 목록 (observation buffer shipments 형식)
        container_type : str
            컨테이너 타입
        mbl_id : str
            MBL 식별자

        Returns
        -------
        LoadingPlan
        """
        if container_type not in _CONTAINER_DIMS:
            container_type = "40GP"

        C_L, C_W, C_H = _CONTAINER_DIMS[container_type]
        max_payload_kg = _get_max_payload(container_type)

        if not shipments:
            return LoadingPlan(container_type=container_type, mbl_id=mbl_id)

        # 화물에 치수 할당 (없으면 CBM에서 역산)
        items = [_resolve_dimensions(s) for s in shipments]

        # 배치 순서: HAZMAT 우선 → OVERSIZED → GENERAL/FRAGILE → FOOD
        # (HAZMAT을 앞쪽에 배치해야 하므로 먼저 처리)
        def sort_key(item):
            cat = item["cargo_category"]
            order = {"HAZMAT": 0, "OVERSIZED": 1, "GENERAL": 2, "FRAGILE": 3, "FOOD": 4}
            return (order.get(cat, 2), -item["volume_mm3"])

        items.sort(key=sort_key)

        # Extreme Points 적재
        placed: List[LoadingPosition] = []
        unplaceable: List[str] = []
        extreme_points: List[Tuple[int, int, int]] = [(0, 0, 0)]  # (x, y, z)

        for item in items:
            placed_flag = False
            best_pos = None
            best_score = float("inf")

            # 컨테이너 구역 제약
            x_min, x_max = _get_zone_limits(item["cargo_category"], C_L)

            # 크기 (회전 포함: 수평 90°만 허용)
            dims_options = _get_rotation_options(item)

            for ep in extreme_points:
                px, py, pz = ep
                if px < x_min or px >= x_max:
                    continue

                for (L, W, H) in dims_options:
                    # FRAGILE: 회전으로 높이가 바뀌면 안 됨 (수직 유지)
                    if item["cargo_category"] == "FRAGILE" and H != item["h_mm"]:
                        continue

                    # 컨테이너 경계 확인
                    if px + L > C_L or py + W > C_W or pz + H > C_H:
                        continue
                    # 구역 경계 확인
                    if px + L > x_max:
                        continue

                    # 충돌 확인
                    if _has_collision(placed, px, py, pz, L, W, H):
                        continue

                    # 지지면 확인 (바닥이거나 아래 화물 위)
                    if not _is_supported(placed, px, py, pz, L, W):
                        continue

                    # FRAGILE 위에 무거운 화물 배치 금지
                    if _on_top_of_fragile(placed, px, py, pz, L, W):
                        if item["weight_kg"] > _FRAGILE_STACK_LIMIT_KG:
                            continue

                    # 점수: 낮은 z → 컨테이너 전방 → 작은 y 순 선호 (안정성)
                    score = pz * 10000 + px + py
                    if score < best_score:
                        best_score = score
                        best_pos = (px, py, pz, L, W, H)

            if best_pos:
                px, py, pz, L, W, H = best_pos
                rotated = (L != item["l_mm"] or W != item["w_mm"])
                lp = LoadingPosition(
                    shipment_id=item["shipment_id"],
                    x_mm=px, y_mm=py, z_mm=pz,
                    length_mm=L, width_mm=W, height_mm=H,
                    rotated=rotated,
                    cargo_category=item["cargo_category"],
                    weight_kg=item["weight_kg"],
                )
                placed.append(lp)
                placed_flag = True

                # 새 Extreme Points 추가 (배치된 박스의 3 꼭짓점)
                extreme_points.append((px + L, py, pz))      # 오른쪽
                extreme_points.append((px, py + W, pz))      # 옆
                extreme_points.append((px, py, pz + H))      # 위
                # 중복 제거 및 컨테이너 범위 내만 유지
                extreme_points = list({
                    ep for ep in extreme_points
                    if ep[0] < C_L and ep[1] < C_W and ep[2] < C_H
                })

            if not placed_flag:
                unplaceable.append(item["shipment_id"])

        # --- 지표 계산 ---
        total_item_vol = sum(
            p.length_mm * p.width_mm * p.height_mm for p in placed
        )
        container_vol = C_L * C_W * C_H
        v_util = total_item_vol / container_vol * 100 if container_vol > 0 else 0.0

        total_weight = sum(p.weight_kg for p in placed)
        w_util = total_weight / max_payload_kg * 100 if max_payload_kg > 0 else 0.0

        # 무게 중심
        cog_x, cog_y, cog_z = _compute_cog(placed, C_L, C_W, C_H)

        # 바닥 하중 (x-y 그리드로 추정, 단순화)
        floor_load = _estimate_max_floor_load(placed)

        # 안정성: COG X가 40~60% 구간이면 준수
        stability = 40.0 <= cog_x <= 60.0

        # ASCII 뷰
        ascii_top = _render_ascii_top(placed, C_L, C_W, width=40, height=12)
        ascii_side = _render_ascii_side(placed, C_L, C_H, width=40, height=10)

        return LoadingPlan(
            container_type=container_type,
            mbl_id=mbl_id,
            positions=placed,
            unplaceable_shipments=unplaceable,
            volume_utilization_pct=v_util,
            weight_utilization_pct=w_util,
            cog_x_pct=cog_x,
            cog_y_pct=cog_y,
            cog_z_pct=cog_z,
            stability_compliant=stability,
            floor_load_max_kg_per_m2=floor_load,
            ascii_view_top=ascii_top,
            ascii_view_side=ascii_side,
        )


# ---------------------------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------------------------

def _get_max_payload(container_type: str) -> float:
    payloads = {"20GP": 28230, "40GP": 26750, "40HC": 26460, "45HC": 27700}
    return payloads.get(container_type, 26750)


def _resolve_dimensions(s: dict) -> dict:
    """화물 치수 해결: 기재된 치수 사용, 없으면 CBM에서 역산."""
    item_type = s.get("item_type", "ELECTRONICS")
    cbm = s.get("cbm", 0.1)
    weight_kg = s.get("weight", 10.0)

    l_cm = s.get("length_cm")
    w_cm = s.get("width_cm")
    h_cm = s.get("height_cm")

    if l_cm and w_cm and h_cm:
        l_mm = int(l_cm * 10)
        w_mm = int(w_cm * 10)
        h_mm = int(h_cm * 10)
    else:
        # CBM에서 종횡비로 역산
        ar = _ASPECT_RATIOS.get(item_type, (2.0, 1.5, 1.0))
        vol_mm3 = cbm * 1_000_000_000  # CBM → mm³
        scale = (vol_mm3 / (ar[0] * ar[1] * ar[2])) ** (1/3)
        l_mm = max(100, int(ar[0] * scale))
        w_mm = max(100, int(ar[1] * scale))
        h_mm = max(100, int(ar[2] * scale))

    return {
        "shipment_id": s["shipment_id"],
        "cargo_category": s.get("cargo_category", "GENERAL"),
        "item_type": item_type,
        "l_mm": l_mm,
        "w_mm": w_mm,
        "h_mm": h_mm,
        "weight_kg": weight_kg,
        "volume_mm3": l_mm * w_mm * h_mm,
    }


def _get_zone_limits(cargo_category: str, container_length: int) -> Tuple[int, int]:
    """카테고리별 X축 배치 허용 구간 (mm)."""
    if cargo_category == "HAZMAT":
        # 도어 쪽(x=0이 도어) 앞 20%
        return (0, int(container_length * _HAZMAT_ZONE_RATIO))
    elif cargo_category == "FOOD":
        # HAZMAT 구역 외 후방 80%
        return (int(container_length * _HAZMAT_ZONE_RATIO), container_length)
    else:
        return (0, container_length)


def _get_rotation_options(item: dict) -> List[Tuple[int, int, int]]:
    """허용 회전 옵션 (L, W, H). FRAGILE은 H 고정."""
    l, w, h = item["l_mm"], item["w_mm"], item["h_mm"]
    if item["cargo_category"] == "FRAGILE":
        return [(l, w, h), (w, l, h)]  # 수평 회전만
    else:
        return [(l, w, h), (w, l, h)]  # 단순화: 수평 90° 회전만


def _has_collision(placed: List[LoadingPosition], px, py, pz, L, W, H) -> bool:
    """새 박스가 기존 배치된 박스와 겹치는지 확인."""
    for p in placed:
        if (px < p.x_mm + p.length_mm and px + L > p.x_mm and
                py < p.y_mm + p.width_mm and py + W > p.y_mm and
                pz < p.z_mm + p.height_mm and pz + H > p.z_mm):
            return True
    return False


def _is_supported(placed: List[LoadingPosition], px, py, pz, L, W) -> bool:
    """박스가 바닥(z=0)이거나 아래 박스 위에 놓이는지 확인."""
    if pz == 0:
        return True
    # 아래에 지지 면적이 있는지 확인
    for p in placed:
        if p.z_mm + p.height_mm == pz:
            overlap_x = max(0, min(px + L, p.x_mm + p.length_mm) - max(px, p.x_mm))
            overlap_y = max(0, min(py + W, p.y_mm + p.width_mm) - max(py, p.y_mm))
            if overlap_x * overlap_y > 0:
                return True
    return False


def _on_top_of_fragile(placed: List[LoadingPosition], px, py, pz, L, W) -> bool:
    """배치 위치 아래에 FRAGILE 화물이 있는지 확인."""
    for p in placed:
        if p.cargo_category == "FRAGILE" and p.z_mm + p.height_mm == pz:
            overlap_x = max(0, min(px + L, p.x_mm + p.length_mm) - max(px, p.x_mm))
            overlap_y = max(0, min(py + W, p.y_mm + p.width_mm) - max(py, p.y_mm))
            if overlap_x * overlap_y > 0:
                return True
    return False


def _compute_cog(
    placed: List[LoadingPosition],
    C_L: int, C_W: int, C_H: int,
) -> Tuple[float, float, float]:
    """무게 중심 (컨테이너 치수 대비 %)."""
    if not placed:
        return 50.0, 50.0, 30.0
    total_w = sum(p.weight_kg for p in placed)
    if total_w <= 0:
        return 50.0, 50.0, 30.0
    cx = sum(p.weight_kg * (p.x_mm + p.length_mm / 2) for p in placed) / total_w
    cy = sum(p.weight_kg * (p.y_mm + p.width_mm / 2) for p in placed) / total_w
    cz = sum(p.weight_kg * (p.z_mm + p.height_mm / 2) for p in placed) / total_w
    return (cx / C_L * 100, cy / C_W * 100, cz / C_H * 100)


def _estimate_max_floor_load(placed: List[LoadingPosition]) -> float:
    """바닥에 닿은 화물의 단위 면적당 최대 하중 추정 (kg/m²)."""
    floor_items = [p for p in placed if p.z_mm == 0]
    if not floor_items:
        return 0.0
    max_load = 0.0
    for p in floor_items:
        area_m2 = (p.length_mm / 1000) * (p.width_mm / 1000)
        if area_m2 > 0:
            load = p.weight_kg / area_m2
            max_load = max(max_load, load)
    return max_load


def _render_ascii_top(
    placed: List[LoadingPosition],
    C_L: int, C_W: int,
    width: int = 40, height: int = 12,
) -> str:
    """상단 뷰 ASCII 렌더링 (X=길이, Y=너비)."""
    grid = [["·"] * width for _ in range(height)]
    symbols = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    for i, p in enumerate(placed[:len(symbols)]):
        sym = symbols[i]
        x1 = int(p.x_mm / C_L * width)
        x2 = max(x1 + 1, int((p.x_mm + p.length_mm) / C_L * width))
        y1 = int(p.y_mm / C_W * height)
        y2 = max(y1 + 1, int((p.y_mm + p.width_mm) / C_W * height))
        for y in range(min(y1, height - 1), min(y2, height)):
            for x in range(min(x1, width - 1), min(x2, width)):
                grid[y][x] = sym

    border = "+" + "-" * width + "+"
    rows = [border]
    for row in grid:
        rows.append("|" + "".join(row) + "|")
    rows.append(border)
    rows.append(f"  [TOP VIEW] Container Length →  ({C_L}mm)")
    return "\n".join(rows)


def _render_ascii_side(
    placed: List[LoadingPosition],
    C_L: int, C_H: int,
    width: int = 40, height: int = 10,
) -> str:
    """측면 뷰 ASCII 렌더링 (X=길이, Z=높이)."""
    grid = [["·"] * width for _ in range(height)]
    symbols = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    for i, p in enumerate(placed[:len(symbols)]):
        sym = symbols[i]
        x1 = int(p.x_mm / C_L * width)
        x2 = max(x1 + 1, int((p.x_mm + p.length_mm) / C_L * width))
        # Z축: 아래가 0 → 그리드 뒤집기
        z1_grid = height - 1 - int((p.z_mm + p.height_mm) / C_H * height)
        z2_grid = height - 1 - int(p.z_mm / C_H * height)
        z1_grid = max(0, z1_grid)
        z2_grid = min(height - 1, z2_grid)
        for z in range(z1_grid, z2_grid + 1):
            for x in range(min(x1, width - 1), min(x2, width)):
                grid[z][x] = sym

    border = "+" + "-" * width + "+"
    rows = [border]
    for row in grid:
        rows.append("|" + "".join(row) + "|")
    rows.append(border)
    rows.append(f"  [SIDE VIEW] ← Door  Container Length →  ({C_L}mm)")
    return "\n".join(rows)
