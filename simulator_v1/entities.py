"""
entities.py
===========
핵심 도메인 객체: Shipment, HBL, MBL
"""

from __future__ import annotations

import uuid
import random
from dataclasses import dataclass, field
from typing import List, Optional

from enum import Enum

from .compatibility import CargoCategory
from .volume_model import effective_cbm_from_raw
from .distributions import sample_cbm, sample_weight, sample_packages, sample_category  # noqa: F401


class ItemType(Enum):
    ELECTRONICS   = "ELECTRONICS"    # 전자제품   — 소형·파손주의·고빈도
    CLOTHING      = "CLOTHING"       # 의류/섬유  — 소형·일반·고빈도
    COSMETICS     = "COSMETICS"      # 화장품     — 소형·일반/위험물·중빈도
    FOOD_PRODUCTS = "FOOD_PRODUCTS"  # 식품/음료  — 중형·식품·중빈도
    AUTO_PARTS    = "AUTO_PARTS"     # 자동차부품 — 중형·일반·중빈도
    CHEMICALS     = "CHEMICALS"      # 화학물질   — 중형·위험물·저빈도
    FURNITURE     = "FURNITURE"      # 가구/인테리어 — 대형·파손주의·저빈도
    MACHINERY     = "MACHINERY"      # 기계/장비  — 대형·특대형·저빈도


@dataclass
class Shipment:
    shipment_id: str
    item_type: ItemType
    cargo_category: CargoCategory
    arrival_time: float
    destination: str
    weight: float
    cbm: float
    packages: int
    due_time: float
    length_cm: Optional[float] = None
    height_cm: Optional[float] = None
    width_cm: Optional[float] = None

    dispatched: bool = False
    dispatch_time: Optional[float] = None

    @property
    def effective_cbm(self) -> float:
        """패킹/혼적 시 실제 점유 CBM 추정치."""
        return round(effective_cbm_from_raw(self.cbm, self.cargo_category.value), 3)

    @property
    def waiting_time(self) -> float:
        if self.dispatch_time is not None:
            return self.dispatch_time - self.arrival_time
        return -1.0

    def is_late(self) -> bool:
        return self.dispatch_time is not None and self.dispatch_time > self.due_time


@dataclass
class HBL:
    hbl_id: str
    shipment_id: str
    linked_mbl_id: str
    weight: float
    cbm: float
    packages: int
    cargo_category: str
    length_cm: Optional[float] = None
    height_cm: Optional[float] = None
    width_cm: Optional[float] = None


@dataclass
class MBL:
    mbl_id: str
    shipment_ids: List[str]
    total_weight: float
    total_cbm: float
    total_effective_cbm: float
    total_packages: int
    dispatch_time: float
    loading_plan: Optional[dict] = None
    hbls: List[HBL] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def generate_shipment(
    rng: random.Random,
    current_time: float,
    item_type: ItemType,
    destination: str = "PORT_A",
    sla_hours: float = 48.0,
    dimensions_cm: Optional[tuple[float, float, float]] = None,
) -> Shipment:
    itv = item_type.value
    cargo_category = sample_category(rng, itv)
    length_cm = height_cm = width_cm = None
    if dimensions_cm is not None:
        length_cm, height_cm, width_cm = dimensions_cm
        cbm = round(length_cm * height_cm * width_cm / 1_000_000, 6)
    else:
        cbm = sample_cbm(rng, itv)
    return Shipment(
        shipment_id=f"SHP-{uuid.uuid4().hex[:8].upper()}",
        item_type=item_type,
        cargo_category=cargo_category,
        arrival_time=current_time,
        destination=destination,
        weight=sample_weight(rng, cbm, cargo_category, itv),
        cbm=cbm,
        packages=sample_packages(rng, itv),
        due_time=current_time + sla_hours,
        length_cm=length_cm,
        height_cm=height_cm,
        width_cm=width_cm,
    )


def create_mbl(shipments: List[Shipment], dispatch_time: float, loading_plan: Optional[dict] = None) -> MBL:
    mbl_id = f"MBL-{uuid.uuid4().hex[:8].upper()}"
    if loading_plan:
        loading_plan = dict(loading_plan)
        loading_plan["mbl_id"] = mbl_id
    mbl = MBL(
        mbl_id=mbl_id,
        shipment_ids=[s.shipment_id for s in shipments],
        total_weight=round(sum(s.weight for s in shipments), 2),
        total_cbm=round(sum(s.cbm for s in shipments), 3),
        total_effective_cbm=round(sum(s.effective_cbm for s in shipments), 3),
        total_packages=sum(s.packages for s in shipments),
        dispatch_time=dispatch_time,
        loading_plan=loading_plan,
    )
    for s in shipments:
        mbl.hbls.append(HBL(
            hbl_id=f"HBL-{uuid.uuid4().hex[:8].upper()}",
            shipment_id=s.shipment_id,
            linked_mbl_id=mbl_id,
            weight=s.weight,
            cbm=s.cbm,
            packages=s.packages,
            cargo_category=s.cargo_category.value,
            length_cm=s.length_cm,
            height_cm=s.height_cm,
            width_cm=s.width_cm,
        ))
    return mbl
