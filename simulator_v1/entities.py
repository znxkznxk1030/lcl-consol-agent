"""
entities.py
===========
핵심 도메인 객체: Shipment, HBL, MBL
"""

from __future__ import annotations

import uuid
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from enum import Enum

from .compatibility import CargoCategory, FRAGILE_CBM_MULTIPLIER


class ItemType(Enum):
    A = "A"   # frequent, small
    B = "B"   # medium
    C = "C"   # rare, large


ITEM_PROFILES: Dict[ItemType, Dict[str, tuple]] = {
    ItemType.A: {"cbm": (0.1, 0.5),  "weight": (5,   50),  "packages": (1, 3)},
    ItemType.B: {"cbm": (0.5, 2.0),  "weight": (50,  200), "packages": (2, 8)},
    ItemType.C: {"cbm": (2.0, 5.0),  "weight": (200, 800), "packages": (5, 20)},
}

# ItemType별 CargoCategory 확률 분포 [(category, cumulative_prob)]
CATEGORY_PROBS: Dict[ItemType, List[Tuple[CargoCategory, float]]] = {
    ItemType.A: [
        (CargoCategory.GENERAL,  0.60),
        (CargoCategory.FOOD,     0.85),   # +0.25
        (CargoCategory.FRAGILE,  1.00),   # +0.15
    ],
    ItemType.B: [
        (CargoCategory.GENERAL,  0.50),
        (CargoCategory.HAZMAT,   0.70),   # +0.20
        (CargoCategory.FOOD,     0.85),   # +0.15
        (CargoCategory.FRAGILE,  1.00),   # +0.15
    ],
    ItemType.C: [
        (CargoCategory.GENERAL,   0.40),
        (CargoCategory.HAZMAT,    0.65),  # +0.25
        (CargoCategory.OVERSIZED, 0.90),  # +0.25
        (CargoCategory.FRAGILE,   1.00),  # +0.10
    ],
}


def _sample_category(rng: random.Random, item_type: ItemType) -> CargoCategory:
    r = rng.random()
    for category, cum_prob in CATEGORY_PROBS[item_type]:
        if r <= cum_prob:
            return category
    return CargoCategory.GENERAL


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

    dispatched: bool = False
    dispatch_time: Optional[float] = None

    @property
    def effective_cbm(self) -> float:
        """패킹 시 실제 점유 CBM. FRAGILE은 완충 공간 포함."""
        if self.cargo_category == CargoCategory.FRAGILE:
            return round(self.cbm * FRAGILE_CBM_MULTIPLIER, 3)
        return self.cbm

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


@dataclass
class MBL:
    mbl_id: str
    shipment_ids: List[str]
    total_weight: float
    total_cbm: float
    total_effective_cbm: float
    total_packages: int
    dispatch_time: float
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
) -> Shipment:
    profile = ITEM_PROFILES[item_type]
    cargo_category = _sample_category(rng, item_type)
    return Shipment(
        shipment_id=f"SHP-{uuid.uuid4().hex[:8].upper()}",
        item_type=item_type,
        cargo_category=cargo_category,
        arrival_time=current_time,
        destination=destination,
        weight=round(rng.uniform(*profile["weight"]), 2),
        cbm=round(rng.uniform(*profile["cbm"]), 3),
        packages=rng.randint(*profile["packages"]),
        due_time=current_time + sla_hours,
    )


def create_mbl(shipments: List[Shipment], dispatch_time: float) -> MBL:
    mbl_id = f"MBL-{uuid.uuid4().hex[:8].upper()}"
    mbl = MBL(
        mbl_id=mbl_id,
        shipment_ids=[s.shipment_id for s in shipments],
        total_weight=round(sum(s.weight for s in shipments), 2),
        total_cbm=round(sum(s.cbm for s in shipments), 3),
        total_effective_cbm=round(sum(s.effective_cbm for s in shipments), 3),
        total_packages=sum(s.packages for s in shipments),
        dispatch_time=dispatch_time,
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
        ))
    return mbl
