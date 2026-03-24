"""
buffer.py
=========
창고 버퍼: 대기 중인 shipment 관리
"""

from __future__ import annotations
from typing import List, Dict
from .entities import Shipment


class WarehouseBuffer:
    def __init__(self) -> None:
        self._shipments: List[Shipment] = []

    def add(self, shipment: Shipment) -> None:
        self._shipments.append(shipment)

    def remove(self, shipment_ids: List[str]) -> List[Shipment]:
        id_set = set(shipment_ids)
        removed, remaining = [], []
        for s in self._shipments:
            (removed if s.shipment_id in id_set else remaining).append(s)
        self._shipments = remaining
        return removed

    def all(self) -> List[Shipment]:
        return list(self._shipments)

    def ids(self) -> List[str]:
        return [s.shipment_id for s in self._shipments]

    @property
    def count(self) -> int:
        return len(self._shipments)

    @property
    def total_cbm(self) -> float:
        return round(sum(s.cbm for s in self._shipments), 3)

    @property
    def total_weight(self) -> float:
        return round(sum(s.weight for s in self._shipments), 2)
