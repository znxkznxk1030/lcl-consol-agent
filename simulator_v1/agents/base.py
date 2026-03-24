"""
agents/base.py
==============
AgentBase: 모든 agent가 구현해야 하는 인터페이스
- act(observation: dict) -> dict 하나만 구현하면 됨
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
                "selected_ids": List[str],
                "reason": str  (optional)
            }
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 유틸: 단순 greedy 선택 (effective_cbm 기준, 호환성 무시)
    # ------------------------------------------------------------------

    def _greedy_select(self, shipments: List[dict], max_cbm: float) -> List[str]:
        """도착 순 정렬 후 effective_cbm 기준 CBM 한도 내 greedy 선택.
        호환성은 고려하지 않으며 env가 위반 시 자동 분리한다."""
        selected, total = [], 0.0
        for s in sorted(shipments, key=lambda x: x["arrival_time"]):
            ecbm = s.get("effective_cbm", s["cbm"])
            if total + ecbm <= max_cbm:
                selected.append(s["shipment_id"])
                total += ecbm
        return selected

    # ------------------------------------------------------------------
    # 유틸: 호환성 인식 greedy 선택
    # ------------------------------------------------------------------

    def _compatible_greedy_select(
        self, shipments: List[dict], max_cbm: float
    ) -> List[str]:
        """
        1단계: 호환 가능한 그룹으로 분리 (HAZMAT끼리, FOOD+FRAGILE끼리, GENERAL 등)
        2단계: 각 그룹 내 effective_cbm 기준 greedy 선택
        3단계: 가장 많은 CBM을 채울 수 있는 그룹의 결과를 반환

        이 helper를 쓰면 호환성 위반 없이 단일 MBL을 최대한 채울 수 있다.
        """
        # 카테고리별로 묶기
        by_category: Dict[str, List[dict]] = defaultdict(list)
        for s in sorted(shipments, key=lambda x: x["arrival_time"]):
            by_category[s.get("cargo_category", "GENERAL")].append(s)

        # 호환 가능한 카테고리 조합 후보 생성
        # 규칙 요약:
        #   OVERSIZED → 단독
        #   HAZMAT    → GENERAL과만 혼적 가능
        #   FOOD      → GENERAL, FRAGILE과 혼적 가능
        #   FRAGILE   → GENERAL, FOOD과 혼적 가능
        #   GENERAL   → HAZMAT, FOOD, FRAGILE과 혼적 가능 (단, OVERSIZED 제외)
        candidate_groups: List[List[dict]] = []

        # OVERSIZED 단독 후보
        for s in by_category.get("OVERSIZED", []):
            candidate_groups.append([s])

        # HAZMAT + GENERAL
        hazmat_group = by_category.get("HAZMAT", []) + by_category.get("GENERAL", [])
        if hazmat_group:
            candidate_groups.append(hazmat_group)

        # FOOD + FRAGILE + GENERAL
        food_group = (
            by_category.get("FOOD", [])
            + by_category.get("FRAGILE", [])
            + by_category.get("GENERAL", [])
        )
        if food_group:
            candidate_groups.append(food_group)

        # GENERAL만 (위 두 후보에 포함되지만 HAZMAT/FOOD 없는 경우)
        general_only = by_category.get("GENERAL", [])
        if general_only and not by_category.get("HAZMAT") and not by_category.get("FOOD"):
            candidate_groups.append(general_only)

        # 각 후보에서 greedy 선택 후 가장 CBM이 높은 결과 반환
        best_ids: List[str] = []
        best_cbm: float = -1.0

        for group in candidate_groups:
            selected, total = [], 0.0
            for s in group:
                ecbm = s.get("effective_cbm", s["cbm"])
                if total + ecbm <= max_cbm:
                    selected.append(s["shipment_id"])
                    total += ecbm
            if total > best_cbm:
                best_cbm = total
                best_ids = selected

        return best_ids

    def _make_action(self, action: str, selected_ids: List[str], reason: str = "") -> dict:
        return {
            "schema": "action/v1",
            "agent_id": self.agent_id,
            "action": action,
            "selected_ids": selected_ids,
            "reason": reason,
        }
