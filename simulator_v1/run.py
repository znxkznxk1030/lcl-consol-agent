"""
run.py
======
시뮬레이션 실행 및 결과 비교
"""

from __future__ import annotations

import json
from pathlib import Path

from .env import ConsolidationEnv, EnvConfig
from .agents.rule import TimeBasedAgent, ThresholdAgent, HybridAgent


def run_all(config: EnvConfig = EnvConfig(), save_dir: str = None) -> list[dict]:
    agents = [
        TimeBasedAgent(),
        ThresholdAgent(cbm_threshold=8.0),
        HybridAgent(cbm_threshold=7.0, max_wait_hours=36.0, cutoff_buffer_hours=1.0),
    ]

    results = []
    print("=" * 65)
    print("  Dynamic Consolidation Simulator v1")
    print("=" * 65)

    for agent in agents:
        env = ConsolidationEnv(config)
        result = env.run(agent)
        results.append(result)

        m = result.metrics
        print(f"\n[Agent: {result.agent_id.upper()}]")
        print(f"  {'total_shipments':<30} {m.total_shipments}")
        print(f"  {'number_of_mbls':<30} {m.number_of_mbls}")
        print(f"  {'avg_waiting_time_hrs':<30} {m.avg_waiting_time_hrs}")
        print(f"  {'sla_violation_rate':<30} {m.sla_violation_rate}")
        print(f"  {'avg_fill_rate':<30} {m.avg_fill_rate}")
        print(f"  {'total_cost':<30} {m.total_cost}")
        print(f"    mbl_cost:     {m.mbl_cost}")
        print(f"    holding_cost: {m.holding_cost}")
        print(f"    late_cost:    {m.late_cost}")

        if save_dir:
            out = Path(save_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"result_{agent.agent_id}.json"
            path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            print(f"  → saved: {path}")

    print()
    return results


if __name__ == "__main__":
    cfg = EnvConfig(seed=42, sim_duration_hours=72)
    run_all(cfg, save_dir="outputs")
