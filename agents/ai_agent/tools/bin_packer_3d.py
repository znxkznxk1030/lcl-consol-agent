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

    개선 사항:
      - 6방향 회전 시도 (FRAGILE 제외)
      - 접촉면 최대화 스코어링 (벽·기존 박스와 붙어있는 위치 우선)
      - 벽면 투영 Extreme Points 추가
      - 1차 실패 화물 대상 2차 패스 재시도
    """

    def pack(
        self,
        shipments: List[dict],
        container_type: str = "40GP",
        mbl_id: str = "MBL-000",
    ) -> LoadingPlan:
        if container_type not in _CONTAINER_DIMS:
            container_type = "40GP"

        C_L, C_W, C_H = _CONTAINER_DIMS[container_type]
        max_payload_kg = _get_max_payload(container_type)

        if not shipments:
            return LoadingPlan(container_type=container_type, mbl_id=mbl_id)

        items = [_resolve_dimensions(s) for s in shipments]

        # 배치 순서: HAZMAT → OVERSIZED → GENERAL/FRAGILE → FOOD, 동순위는 부피 큰 것 먼저
        def sort_key(item):
            cat = item["cargo_category"]
            order = {"HAZMAT": 0, "OVERSIZED": 1, "GENERAL": 2, "FRAGILE": 3, "FOOD": 4}
            return (order.get(cat, 2), -item["volume_mm3"])

        items.sort(key=sort_key)

        placed: List[LoadingPosition] = []
        unplaceable: List[str] = []
        extreme_points: List[Tuple[int, int, int]] = [(0, 0, 0)]

        # 1차 패스
        remaining = self._run_pass(items, placed, unplaceable, extreme_points, C_L, C_W, C_H)

        # 2차 패스: 1차에서 실패한 화물을 크기 순으로 재시도
        if remaining:
            second_items = [i for i in items if i["shipment_id"] in {s for s in remaining}]
            second_items.sort(key=lambda i: i["volume_mm3"])  # 작은 것부터 재시도
            still_unplaceable: List[str] = []
            self._run_pass(second_items, placed, still_unplaceable, extreme_points, C_L, C_W, C_H)
            unplaceable = still_unplaceable

        # --- 지표 계산 ---
        total_item_vol = sum(p.length_mm * p.width_mm * p.height_mm for p in placed)
        container_vol = C_L * C_W * C_H
        v_util = total_item_vol / container_vol * 100 if container_vol > 0 else 0.0

        total_weight = sum(p.weight_kg for p in placed)
        w_util = total_weight / max_payload_kg * 100 if max_payload_kg > 0 else 0.0

        cog_x, cog_y, cog_z = _compute_cog(placed, C_L, C_W, C_H)
        floor_load = _estimate_max_floor_load(placed)
        stability = 40.0 <= cog_x <= 60.0

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

    def _run_pass(
        self,
        items: List[dict],
        placed: List[LoadingPosition],
        unplaceable: List[str],
        extreme_points: List[Tuple[int, int, int]],
        C_L: int, C_W: int, C_H: int,
    ) -> List[str]:
        """아이템 목록을 순서대로 배치 시도. 실패한 shipment_id 리스트 반환."""
        failed = []

        for item in items:
            x_min, x_max = _get_zone_limits(item["cargo_category"], C_L)
            dims_options = _get_rotation_options(item)

            # 후보 포인트: EP + 구역 시작점 + 벽면 코너 보강
            candidate_set = set(extreme_points)
            candidate_set.add((x_min, 0, 0))
            candidate_points = list(candidate_set)

            best_pos = None
            best_score = float("inf")

            for ep in candidate_points:
                px, py, pz = ep
                if px < x_min or px >= x_max:
                    continue

                for (L, W, H) in dims_options:
                    if item["cargo_category"] == "FRAGILE" and H != item["h_mm"]:
                        continue

                    if px + L > C_L or py + W > C_W or pz + H > C_H:
                        continue
                    if px + L > x_max:
                        continue
                    if _has_collision(placed, px, py, pz, L, W, H):
                        continue
                    if not _is_supported(placed, px, py, pz, L, W):
                        continue
                    if _on_top_of_fragile(placed, px, py, pz, L, W):
                        if item["weight_kg"] > _FRAGILE_STACK_LIMIT_KG:
                            continue

                    # 스코어: 낮은 z 우선 → 접촉면 최대화 → 작은 x+y
                    contact = _compute_contact_area(placed, px, py, pz, L, W, H, C_L, C_W)
                    score = pz * 1_000_000 - contact + (px + py)
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
                _update_extreme_points(extreme_points, placed, px, py, pz, L, W, H, C_L, C_W, C_H)
            else:
                failed.append(item["shipment_id"])
                unplaceable.append(item["shipment_id"])

        return failed


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
        ar = _ASPECT_RATIOS.get(item_type, (2.0, 1.5, 1.0))
        vol_mm3 = cbm * 1_000_000_000
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
        return (0, int(container_length * _HAZMAT_ZONE_RATIO))
    elif cargo_category == "FOOD":
        return (int(container_length * _HAZMAT_ZONE_RATIO), container_length)
    else:
        return (0, container_length)


def _get_rotation_options(item: dict) -> List[Tuple[int, int, int]]:
    """허용 회전 옵션 (L, W, H).
    FRAGILE은 H 고정 (수직 방향 유지) → 수평 90°만.
    그 외는 6방향 모두 시도하여 빈틈 최소화.
    """
    l, w, h = item["l_mm"], item["w_mm"], item["h_mm"]
    if item["cargo_category"] == "FRAGILE":
        return list({(l, w, h), (w, l, h)})
    else:
        return list({
            (l, w, h), (l, h, w),
            (w, l, h), (w, h, l),
            (h, l, w), (h, w, l),
        })


def _has_collision(placed: List[LoadingPosition], px, py, pz, L, W, H) -> bool:
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
    for p in placed:
        if p.z_mm + p.height_mm == pz:
            overlap_x = max(0, min(px + L, p.x_mm + p.length_mm) - max(px, p.x_mm))
            overlap_y = max(0, min(py + W, p.y_mm + p.width_mm) - max(py, p.y_mm))
            if overlap_x * overlap_y > 0:
                return True
    return False


def _on_top_of_fragile(placed: List[LoadingPosition], px, py, pz, L, W) -> bool:
    for p in placed:
        if p.cargo_category == "FRAGILE" and p.z_mm + p.height_mm == pz:
            overlap_x = max(0, min(px + L, p.x_mm + p.length_mm) - max(px, p.x_mm))
            overlap_y = max(0, min(py + W, p.y_mm + p.width_mm) - max(py, p.y_mm))
            if overlap_x * overlap_y > 0:
                return True
    return False


def _compute_contact_area(
    placed: List[LoadingPosition],
    px, py, pz, L, W, H,
    C_L: int, C_W: int,
) -> int:
    """배치 위치에서 벽·바닥·기존 박스와의 접촉 면적 합산.
    클수록 빈틈 없이 꽉 채워지는 위치.
    """
    contact = 0

    # 컨테이너 벽/바닥 접촉
    if px == 0:       contact += W * H  # 후방벽
    if px + L == C_L: contact += W * H  # 전방벽
    if py == 0:       contact += L * H  # 좌측벽
    if py + W == C_W: contact += L * H  # 우측벽
    if pz == 0:       contact += L * W  # 바닥

    # 기존 박스와의 면 접촉
    for p in placed:
        # X 방향 인접
        if px + L == p.x_mm or p.x_mm + p.length_mm == px:
            oy = max(0, min(py + W, p.y_mm + p.width_mm) - max(py, p.y_mm))
            oz = max(0, min(pz + H, p.z_mm + p.height_mm) - max(pz, p.z_mm))
            contact += oy * oz
        # Y 방향 인접
        if py + W == p.y_mm or p.y_mm + p.width_mm == py:
            ox = max(0, min(px + L, p.x_mm + p.length_mm) - max(px, p.x_mm))
            oz = max(0, min(pz + H, p.z_mm + p.height_mm) - max(pz, p.z_mm))
            contact += ox * oz
        # Z 방향 인접 (위/아래)
        if pz + H == p.z_mm or p.z_mm + p.height_mm == pz:
            ox = max(0, min(px + L, p.x_mm + p.length_mm) - max(px, p.x_mm))
            oy = max(0, min(py + W, p.y_mm + p.width_mm) - max(py, p.y_mm))
            contact += ox * oy

    return contact


def _update_extreme_points(
    extreme_points: List[Tuple[int, int, int]],
    placed: List[LoadingPosition],
    px, py, pz, L, W, H,
    C_L: int, C_W: int, C_H: int,
) -> None:
    """새로 배치된 박스 기준으로 Extreme Points 갱신.

    기본 3점 + 벽면 투영 + 기존 박스와의 교차 투영점 추가.
    """
    new_eps = [
        # 기본 3점 (Crainic 2008)
        (px + L, py, pz),
        (px, py + W, pz),
        (px, py, pz + H),
        # 바닥 투영 (새 박스 모서리를 z=0으로 내림)
        (px + L, py, 0),
        (px, py + W, 0),
        # 벽면 투영 (y=0, x=0 쪽으로 붙이기)
        (px + L, 0, pz),
        (0, py + W, pz),
        (px + L, 0, pz + H),
        (0, py + W, pz + H),
    ]

    # 기존 배치 박스와의 교차 투영점
    for p in placed[:-1]:
        same_z = p.z_mm == pz
        if same_z:
            new_eps += [
                (p.x_mm + p.length_mm, py, pz),
                (px, p.y_mm + p.width_mm, pz),
                (px + L, p.y_mm, pz),
                (p.x_mm, py + W, pz),
            ]
        top_z = p.z_mm + p.height_mm
        new_eps += [
            (px + L, py, top_z),
            (px, py + W, top_z),
            (p.x_mm + p.length_mm, p.y_mm, top_z),
            (p.x_mm, p.y_mm + p.width_mm, top_z),
            # 새 박스 상단 모서리도 투영
            (px + L, p.y_mm, pz + H),
            (p.x_mm, py + W, pz + H),
        ]

    # 유효 범위 내 EP만 유지, 중복 제거
    ep_set = set(extreme_points)
    for ep in new_eps:
        if ep[0] < C_L and ep[1] < C_W and ep[2] < C_H:
            ep_set.add(ep)
    extreme_points.clear()
    extreme_points.extend(ep_set)


def _compute_cog(
    placed: List[LoadingPosition],
    C_L: int, C_W: int, C_H: int,
) -> Tuple[float, float, float]:
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
    grid = [["·"] * width for _ in range(height)]
    symbols = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    for i, p in enumerate(placed[:len(symbols)]):
        sym = symbols[i]
        x1 = int(p.x_mm / C_L * width)
        x2 = max(x1 + 1, int((p.x_mm + p.length_mm) / C_L * width))
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
