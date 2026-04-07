"""
report_builder.py
=================
ConsolidationReport 데이터 클래스 및 Markdown 보고서 렌더러.

보고서 구성:
  Section 1: 물동량 예측 (Volume Forecast)
  Section 2: SLA 위험도 분석
  Section 3: Hapag-Lloyd 스펙 준수 검증
  Section 4: 컨테이너 적재 계획 (3D Loading Plan)
  Section 5: 비용 분석
  Section 6: AI 의사결정 결과
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


@dataclass
class ConsolidationReport:
    # 식별자
    report_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str = "default"
    generated_at: float = 0.0       # 시뮬레이션 시간 (hour)
    generated_at_wall: str = ""     # 실제 시간 (ISO 8601)
    agent_version: str = "ai_agent/v1"

    # Section 1: 물동량 예측
    volume_forecast: Dict[str, Any] = field(default_factory=dict)

    # Section 2: SLA 위험도
    sla_assessment: Dict[str, Any] = field(default_factory=dict)

    # Section 3: 컨테이너 스펙 검증
    hapag_spec_checks: List[Dict[str, Any]] = field(default_factory=list)

    # Section 4: 적재 계획
    loading_plans: List[Dict[str, Any]] = field(default_factory=list)

    # Section 5: 비용 분석
    cost_analysis: Dict[str, Any] = field(default_factory=dict)

    # Section 6: 결정
    action: str = "WAIT"
    mbl_groupings: List[List[str]] = field(default_factory=list)
    selected_container_type: str = "40GP"
    claude_reasoning: str = ""
    claude_confidence: float = 0.0
    fallback_used: bool = False

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "session_id": self.session_id,
            "generated_at_sim_hour": round(self.generated_at, 1),
            "generated_at_wall": self.generated_at_wall,
            "agent_version": self.agent_version,
            "volume_forecast": self.volume_forecast,
            "sla_assessment": self.sla_assessment,
            "hapag_spec_checks": self.hapag_spec_checks,
            "loading_plans": self.loading_plans,
            "cost_analysis": self.cost_analysis,
            "decision": {
                "action": self.action,
                "mbl_groupings": self.mbl_groupings,
                "selected_container_type": self.selected_container_type,
                "claude_reasoning": self.claude_reasoning,
                "claude_confidence": round(self.claude_confidence, 3),
                "fallback_used": self.fallback_used,
            },
        }

    def to_markdown(self) -> str:
        """보고서를 Markdown 형식으로 렌더링."""
        lines = []
        ts = self.generated_at_wall or f"Sim T+{self.generated_at:.1f}h"

        lines.append(f"# LCL AI Agent 통합 보고서")
        lines.append(f"**보고서 ID**: `{self.report_id}` | **생성 시각**: {ts} | **시뮬 시각**: T+{self.generated_at:.1f}h")
        lines.append(f"**에이전트**: {self.agent_version} | **Fallback**: {'사용됨 ⚠️' if self.fallback_used else '정상'}")
        lines.append("")

        # --- Section 1: 물동량 예측 ---
        lines.append("---")
        lines.append("## 1. 물동량 예측 (24시간)")
        vf = self.volume_forecast
        if vf:
            total_s = vf.get("total_expected_shipments", 0)
            total_c = vf.get("total_expected_cbm", 0)
            peak = vf.get("peak_hour", "N/A")
            ci = vf.get("confidence_interval_90", {})
            ci_lo = ci.get("low", 0) if isinstance(ci, dict) else ci[0]
            ci_hi = ci.get("high", 0) if isinstance(ci, dict) else ci[1]
            lines.append(f"| 지표 | 값 |")
            lines.append(f"|------|-----|")
            lines.append(f"| 예상 화물 수 | **{total_s:.1f}건** (90% CI: {ci_lo:.0f}~{ci_hi:.0f}) |")
            lines.append(f"| 예상 총 CBM | **{total_c:.2f} CBM** |")
            lines.append(f"| 피크 시간대 | {peak}시 |")

            cw = vf.get("congestion_windows", [])
            if cw:
                lines.append(f"\n**혼잡 구간**:")
                for w in cw[:3]:
                    if isinstance(w, dict):
                        lines.append(f"- T+{w.get('start')}h ~ T+{w.get('end')}h [{w.get('severity')}]")
                    else:
                        lines.append(f"- T+{w[0]}h ~ T+{w[1]}h [{w[2]}]")
        else:
            lines.append("_(예측 데이터 없음)_")
        lines.append("")

        # --- Section 2: SLA 위험도 ---
        lines.append("---")
        lines.append("## 2. SLA 위험도 분석")
        sa = self.sla_assessment
        if sa:
            rs = sa.get("risk_summary", {})
            lines.append(f"| 등급 | 건수 |")
            lines.append(f"|------|------|")
            lines.append(f"| 🔴 CRITICAL | **{rs.get('critical', 0)}건** |")
            lines.append(f"| 🟠 HIGH | **{rs.get('high', 0)}건** |")
            lines.append(f"| 🟡 MEDIUM | **{rs.get('medium', 0)}건** |")
            lines.append(f"| 🟢 LOW | {rs.get('low', 0)}건 |")
            lines.append(f"| **합계** | **{rs.get('total', 0)}건** |")
            lines.append("")

            vp = sa.get("sla_violation_probability", 0)
            ep = sa.get("expected_penalty_if_wait", 0)
            lines.append(f"- 1 tick 대기 시 SLA 위반 예상 확률: **{vp*100:.1f}%**")
            lines.append(f"- 대기 시 추가 예상 패널티: **${ep:.0f}**")

            priority = sa.get("recommended_priority_dispatch", [])
            if priority:
                lines.append(f"\n**즉시 출하 권고 화물** ({len(priority)}건):")
                for sid in priority[:10]:
                    lines.append(f"  - `{sid}`")
        else:
            lines.append("_(SLA 분석 데이터 없음)_")
        lines.append("")

        # --- Section 3: Hapag-Lloyd 스펙 ---
        lines.append("---")
        lines.append("## 3. Hapag-Lloyd 컨테이너 스펙 검증")
        if self.hapag_spec_checks:
            for i, sc in enumerate(self.hapag_spec_checks):
                status = "✅ 준수" if sc.get("compliant") else "❌ 위반"
                lines.append(f"\n### MBL {i+1} — {sc.get('container_type', 'N/A')} {status}")
                lines.append(f"| 항목 | 값 |")
                lines.append(f"|------|-----|")
                lines.append(f"| 용적 활용률 | {sc.get('volume_utilization_pct', 0):.1f}% |")
                lines.append(f"| 중량 활용률 | {sc.get('weight_utilization_pct', 0):.1f}% |")
                lines.append(f"| 총 중량 | {sc.get('total_weight_kg', 0):.0f} kg |")
                lines.append(f"| 총 CBM | {sc.get('total_cbm', 0):.3f} CBM |")
                lines.append(f"| 권고 컨테이너 | {sc.get('recommended_container', 'N/A')} |")

                if sc.get("violations"):
                    lines.append("\n**위반 사항:**")
                    for v in sc["violations"]:
                        lines.append(f"  - ⚠️ {v}")
                if sc.get("warnings"):
                    lines.append("\n**경고:**")
                    for w in sc["warnings"]:
                        lines.append(f"  - ⚡ {w}")
        else:
            lines.append("_(스펙 검증 데이터 없음)_")
        lines.append("")

        # --- Section 4: 적재 계획 ---
        lines.append("---")
        lines.append("## 4. 컨테이너 적재 계획 (3D Loading Plan)")
        if self.loading_plans:
            for i, lp in enumerate(self.loading_plans):
                lines.append(f"\n### MBL {i+1} — {lp.get('container_type', 'N/A')} (`{lp.get('mbl_id', 'N/A')}`)")
                stability = "✅ 안정" if lp.get("stability_compliant") else "⚠️ 불안정"
                lines.append(f"| 항목 | 값 |")
                lines.append(f"|------|-----|")
                lines.append(f"| 용적 활용률 | **{lp.get('volume_utilization_pct', 0):.1f}%** |")
                lines.append(f"| 중량 활용률 | {lp.get('weight_utilization_pct', 0):.1f}% |")
                cog = lp.get("center_of_gravity", {})
                lines.append(f"| 무게 중심 (X) | {cog.get('x_pct', 50):.1f}% {stability} |")
                lines.append(f"| 바닥 최대 하중 | {lp.get('floor_load_max_kg_per_m2', 0):.0f} kg/m² |")
                unplace = lp.get("unplaceable_shipments", [])
                if unplace:
                    lines.append(f"| 미배치 화물 | {len(unplace)}건 ⚠️ |")

                ascii_top = lp.get("ascii_view_top", "")
                if ascii_top:
                    lines.append("\n**적재도 (상단 뷰):**")
                    lines.append("```")
                    lines.append(ascii_top)
                    lines.append("```")

                ascii_side = lp.get("ascii_view_side", "")
                if ascii_side:
                    lines.append("\n**적재도 (측면 뷰):**")
                    lines.append("```")
                    lines.append(ascii_side)
                    lines.append("```")
        else:
            lines.append("_(적재 계획 데이터 없음)_")
        lines.append("")

        # --- Section 5: 비용 분석 ---
        lines.append("---")
        lines.append("## 5. 비용 분석")
        ca = self.cost_analysis
        if ca:
            now = ca.get("dispatch_now_cost", 0)
            wait1 = ca.get("wait_1tick_expected_cost", 0)
            eff = ca.get("consolidation_efficiency", 0) * 100
            cps = ca.get("cost_per_shipment", 0)
            rec = ca.get("recommendation", "N/A")
            bd = ca.get("cost_breakdown_now", {})

            lines.append(f"| 시나리오 | 비용 |")
            lines.append(f"|----------|------|")
            lines.append(f"| **지금 출하** | **${now:.2f}** |")
            lines.append(f"| 1 tick 대기 후 | ${wait1:.2f} |")
            lines.append("")
            lines.append(f"**비용 분해 (지금 출하):**")
            lines.append(f"- MBL 운임: ${bd.get('mbl_cost', 0):.2f}")
            lines.append(f"- 보관비: ${bd.get('holding_cost', 0):.2f}")
            lines.append(f"- SLA 패널티: ${bd.get('late_cost', 0):.2f}")
            lines.append("")
            lines.append(f"- Consolidation 효율: **{eff:.1f}%**")
            lines.append(f"- 화물당 비용: **${cps:.2f}**")
            lines.append(f"- 비용 모델 권고: **{rec}**")
            if ca.get("reasoning"):
                lines.append(f"- 근거: _{ca['reasoning']}_")
        else:
            lines.append("_(비용 분석 데이터 없음)_")
        lines.append("")

        # --- Section 6: AI 결정 ---
        lines.append("---")
        lines.append("## 6. AI 의사결정 결과")
        action_emoji = "🚢 DISPATCH" if self.action == "DISPATCH" else "⏳ WAIT"
        lines.append(f"### 결정: **{action_emoji}**")
        lines.append(f"- 선택 컨테이너: **{self.selected_container_type}**")
        lines.append(f"- 신뢰도: **{self.claude_confidence*100:.0f}%**")
        if self.fallback_used:
            lines.append("- ⚠️ **Fallback 사용** (Claude API 오류 또는 미설정)")
        if self.claude_reasoning:
            lines.append(f"\n**판단 근거:**\n> {self.claude_reasoning}")

        if self.action == "DISPATCH" and self.mbl_groupings:
            lines.append(f"\n**MBL 배정 계획** ({len(self.mbl_groupings)}개 MBL):")
            for i, grp in enumerate(self.mbl_groupings):
                lines.append(f"- **MBL {i+1}** ({len(grp)}건): {', '.join(f'`{sid}`' for sid in grp[:5])}")
                if len(grp) > 5:
                    lines.append(f"  + 외 {len(grp)-5}건")

        lines.append("")
        lines.append("---")
        lines.append(f"_Generated by {self.agent_version} | Report `{self.report_id}`_")

        return "\n".join(lines)


def build_report(
    session_id: str,
    sim_time: float,
    volume_forecast: dict,
    sla_assessment: dict,
    hapag_spec_checks: List[dict],
    loading_plans: List[dict],
    cost_analysis: dict,
    action: str,
    mbl_groupings: List[List[str]],
    selected_container_type: str,
    claude_reasoning: str,
    claude_confidence: float,
    fallback_used: bool,
) -> ConsolidationReport:
    """ConsolidationReport 팩토리 함수."""
    now_wall = datetime.now(timezone.utc).isoformat()
    return ConsolidationReport(
        session_id=session_id,
        generated_at=sim_time,
        generated_at_wall=now_wall,
        volume_forecast=volume_forecast,
        sla_assessment=sla_assessment,
        hapag_spec_checks=hapag_spec_checks,
        loading_plans=loading_plans,
        cost_analysis=cost_analysis,
        action=action,
        mbl_groupings=mbl_groupings,
        selected_container_type=selected_container_type,
        claude_reasoning=claude_reasoning,
        claude_confidence=claude_confidence,
        fallback_used=fallback_used,
    )
