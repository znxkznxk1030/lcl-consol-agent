"""
entities.py
===========
핵심 도메인 객체: Shipment, HBL, MBL
"""

from __future__ import annotations

import uuid
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class ItemType(Enum):
    A = "A"   # frequent, small
    B = "B"   # medium
    C = "C"   # rare, large


ITEM_PROFILES: Dict[ItemType, Dict[str, tuple]] = {
    ItemType.A: {"cbm": (0.1, 0.5),  "weight": (5,   50),  "packages": (1, 3)},
    ItemType.B: {"cbm": (0.5, 2.0),  "weight": (50,  200), "packages": (2, 8)},
    ItemType.C: {"cbm": (2.0, 5.0),  "weight": (200, 800), "packages": (5, 20)},
}


@dataclass
class Shipment:
    shipment_id: str
    item_type: ItemType
    arrival_time: float
    destination: str
    weight: float
    cbm: float
    packages: int
    due_time: float

    dispatched: bool = False
    dispatch_time: Optional[float] = None

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


@dataclass
class MBL:
    mbl_id: str
    shipment_ids: List[str]
    total_weight: float
    total_cbm: float
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
    return Shipment(
        shipment_id=f"SHP-{uuid.uuid4().hex[:8].upper()}",
        item_type=item_type,
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
        ))
    return mbl
