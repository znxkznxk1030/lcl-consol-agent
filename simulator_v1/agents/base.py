"""
agents/base.py
==============
AgentBase: 모든 agent가 구현해야 하는 인터페이스
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict


class AgentBase(ABC):
    agent_id: str = "base"

    @abstractmethod
    def act(self, observation: dict) -> dict:
        """
        Parameters
        ----------
        observation : dict
            Observation 스키마 (schemas.Observation.to_dict())

        Returns
        -------
        dict
            Action 스키마
            {
                "schema": "action/v1",
                "agent_id": str,
                "action": "WAIT" | "DISPATCH",
                "mbls": List[List[str]],   # 각 inner list = 하나의 MBL
                "reason": str  (optional)
            }
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 유틸: First-Fit Decreasing bin packing
    # ------------------------------------------------------------------

    def _bin_pack(
        self,
        shipments: List[dict],
        max_cbm: float,
    ) -> List[List[str]]:
        """
        FFD(First-Fit Decreasing) 알고리즘으로 shipments를 여러 MBL에 배분.
        effective_cbm 내림차순 정렬 후, 먼저 들어갈 수 있는 빈에 배치.
        반환값: [[id1, id2], [id3], ...] (MBL별 shipment ID 목록)
        """
        sorted_ships = sorted(
            shipments,
            key=lambda s: s.get("effective_cbm", s["cbm"]),
            reverse=True,
        )
        bins: List[Dict] = []  # {"ids": [...], "cbm": float}

        for s in sorted_ships:
            ecbm = s.get("effective_cbm", s["cbm"])
            placed = False
            for b in bins:
                if b["cbm"] + ecbm <= max_cbm:
                    b["ids"].append(s["shipment_id"])
                    b["cbm"] += ecbm
                    placed = True
                    break
            if not placed:
                bins.append({"ids": [s["shipment_id"]], "cbm": ecbm})

        return [b["ids"] for b in bins] if bins else []

    # ------------------------------------------------------------------
    # 유틸: 호환성 인식 bin packing
    # ------------------------------------------------------------------

    def _compatible_bin_pack(
        self,
        shipments: List[dict],
        max_cbm: float,
    ) -> List[List[str]]:
        """
        카테고리별로 혼적 가능한 그룹을 먼저 나누고,
        각 그룹 내에서 FFD bin packing 수행.
        결과를 모두 합쳐 반환.
        """
        by_category: Dict[str, List[dict]] = defaultdict(list)
        for s in sorted(shipments, key=lambda x: x["arrival_time"]):
            by_category[s.get("cargo_category", "GENERAL")].append(s)

        all_mbl_ids: List[List[str]] = []

        # HAZMAT + GENERAL
        hazmat_group = (
            by_category.get("HAZMAT", [])
            + by_category.get("GENERAL", [])
            + by_category.get("OVERSIZED", [])
        )
        if hazmat_group:
            all_mbl_ids.extend(self._bin_pack(hazmat_group, max_cbm))

        # FOOD + FRAGILE + GENERAL + OVERSIZED (HAZMAT 없는 경우)
        food_group = by_category.get("FOOD", []) + by_category.get("FRAGILE", [])
        if food_group:
            # GENERAL은 HAZMAT 그룹과 중복되므로 제외
            if not by_category.get("HAZMAT"):
                food_group += by_category.get("GENERAL", []) + by_category.get("OVERSIZED", [])
            all_mbl_ids.extend(self._bin_pack(food_group, max_cbm))

        # 중복 제거 (GENERAL이 여러 그룹에 들어간 경우)
        seen: set = set()
        result: List[List[str]] = []
        for mbl in all_mbl_ids:
            filtered = [sid for sid in mbl if sid not in seen]
            seen.update(filtered)
            if filtered:
                result.append(filtered)

        return result

    def _make_action(
        self,
        action: str,
        mbls: List[List[str]],
        reason: str = "",
    ) -> dict:
        return {
            "schema": "action/v1",
            "agent_id": self.agent_id,
            "action": action,
            "mbls": mbls,
            "reason": reason,
        }
