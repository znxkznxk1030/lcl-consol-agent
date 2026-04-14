"""
evaluate.py
===========
AAMAS 논문 실험용 에이전트 비교 평가 모듈.

실행
----
    python -m rl.evaluate --checkpoint checkpoints/mappo_best.pt --episodes 20

비교 대상 (baseline)
--------------------
  time_based   : 24h 주기 고정 출고
  threshold    : effective CBM >= 8.0 시 출고
  hybrid       : SLA + CBM + max_wait 혼합
  mappo        : 학습된 MAPPO 에이전트
"""

from __future__ import annotations

import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass, asdict
from typing import Dict, List
import numpy as np

from simulator_v1.env import ConsolidationEnv, EnvConfig
from simulator_v1.agents.rule import TimeBasedAgent, ThresholdAgent, HybridAgent

from .rl_agent import RLConsolidationAgent
from .reward import compute_episode_return


@dataclass
class AgentResult:
    agent_id:           str
    avg_fill_rate:      float
    sla_violation_rate: float
    total_cost:         float
    avg_waiting_hrs:    float
    n_mbls:             float
    compat_violations:  float
    episode_return:     float   # composite metric (논문 주 지표)
    std_return:         float


def evaluate_agent(agent, n_episodes: int = 20, base_seed: int = 100) -> AgentResult:
    """단일 에이전트를 n_episodes 에피소드로 평가."""
    returns      = []
    fill_rates   = []
    sla_viols    = []
    costs        = []
    waiting_hrs  = []
    n_mbls_list  = []
    compat_list  = []

    for i in range(n_episodes):
        cfg = EnvConfig(seed=base_seed + i)
        env = ConsolidationEnv(cfg)
        result = env.run(agent)
        m = result.metrics

        returns.append(compute_episode_return(asdict(m)))
        fill_rates.append(m.avg_fill_rate)
        sla_viols.append(m.sla_violation_rate)
        costs.append(m.total_cost)
        waiting_hrs.append(m.avg_waiting_time_hrs)
        n_mbls_list.append(m.number_of_mbls)
        compat_list.append(m.compatibility_violations)

    return AgentResult(
        agent_id=           getattr(agent, "agent_id", "unknown"),
        avg_fill_rate=      float(np.mean(fill_rates)),
        sla_violation_rate= float(np.mean(sla_viols)),
        total_cost=         float(np.mean(costs)),
        avg_waiting_hrs=    float(np.mean(waiting_hrs)),
        n_mbls=             float(np.mean(n_mbls_list)),
        compat_violations=  float(np.mean(compat_list)),
        episode_return=     float(np.mean(returns)),
        std_return=         float(np.std(returns)),
    )


def run_comparison(
    checkpoint_path: Optional[str] = None,
    n_episodes: int = 20,
    output_path: Optional[str] = None,
) -> Dict[str, AgentResult]:
    """
    모든 baseline + MAPPO 에이전트 비교 평가 후 결과 반환.

    Parameters
    ----------
    checkpoint_path : MAPPO 체크포인트 경로 (None 이면 MAPPO 제외)
    n_episodes      : 평가 에피소드 수
    output_path     : 결과 JSON 저장 경로 (None 이면 저장 안 함)
    """
    agents = {
        "time_based": TimeBasedAgent(),
        "threshold":  ThresholdAgent(cbm_threshold=8.0),
        "hybrid":     HybridAgent(cbm_threshold=7.0, max_wait_hours=36.0),
    }
    if checkpoint_path:
        agents["mappo"] = RLConsolidationAgent(checkpoint_path, deterministic=True)

    results = {}
    for name, agent in agents.items():
        print(f"[Eval] Evaluating {name} ({n_episodes} episodes)...")
        r = evaluate_agent(agent, n_episodes=n_episodes)
        results[name] = r
        print(
            f"  fill={r.avg_fill_rate:.2%}  "
            f"sla_viol={r.sla_violation_rate:.2%}  "
            f"cost={r.total_cost:.0f}  "
            f"return={r.episode_return:.4f}±{r.std_return:.4f}"
        )

    # 표 출력
    _print_table(results)

    if output_path:
        with open(output_path, "w") as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)
        print(f"[Eval] Results saved → {output_path}")

    return results


def _print_table(results: Dict[str, AgentResult]) -> None:
    """논문 Table 형식 출력."""
    header = (
        f"{'Agent':<18} {'Fill Rate':>10} {'SLA Viol':>10} "
        f"{'Cost':>10} {'Wait(h)':>9} {'MBLs':>7} {'Return':>10}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<18} {r.avg_fill_rate:>9.2%}  {r.sla_violation_rate:>9.2%}  "
            f"{r.total_cost:>9.0f}  {r.avg_waiting_hrs:>8.1f}  "
            f"{r.n_mbls:>6.1f}  {r.episode_return:>+10.4f}"
        )
    print("=" * len(header) + "\n")


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAPPO vs Baseline Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="MAPPO checkpoint path (.pt)")
    parser.add_argument("--episodes",   type=int, default=20)
    parser.add_argument("--output",     type=str, default="eval_results.json")
    args = parser.parse_args()

    run_comparison(
        checkpoint_path=args.checkpoint,
        n_episodes=args.episodes,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
