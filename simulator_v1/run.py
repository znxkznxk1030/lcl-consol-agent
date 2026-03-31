"""
run.py
======
시뮬레이션 실행 및 결과 비교
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

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
        print(f"  {'compatibility_violations':<30} {m.compatibility_violations}")
        print(f"  {'compatibility_extra_mbls':<30} {m.compatibility_extra_mbls}")
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


def run_olist(
    archive_dir: str | Path,
    total_rate: float = 6.0,
    use_olist_cbm: bool = False,
    save_dir: Optional[str] = None,
    seed: int = 42,
    sim_duration_hours: int = 72,
) -> list[dict]:
    """
    Olist CSV 분포 기반으로 시뮬레이션을 실행한다.

    Parameters
    ----------
    archive_dir : str | Path
        archive 폴더 경로 (olist_*.csv 가 있어야 함).
    total_rate : float
        시간당 총 도착 화물 수. 카테고리 비율은 Olist 데이터에서 추출.
        기본값 6.0 (기존 기본 설정과 유사).
    use_olist_cbm : bool
        True 이면 Olist 실측 치수 → LCL 환산 CBM 파라미터 사용.
        False(기본) 이면 기존 학술 기반 CBM 파라미터 유지.
    save_dir : str, optional
        결과 JSON 저장 폴더.
    seed : int
        난수 시드.
    sim_duration_hours : int
        시뮬레이션 기간 (시간).

    Returns
    -------
    list[SimulationResult]
    """
    from .olist_calibration import make_olist_config

    cfg = make_olist_config(
        archive_dir=archive_dir,
        total_rate=total_rate,
        use_olist_cbm=use_olist_cbm,
        seed=seed,
        sim_duration_hours=sim_duration_hours,
    )

    print("=" * 65)
    print("  Dynamic Consolidation Simulator v1  [Olist 기반 물동량]")
    print("=" * 65)
    print("\n  arrival_rates (Olist 분포 → total_rate 스케일):")
    for k, v in sorted(cfg.arrival_rates.items(), key=lambda x: -x[1]):
        print(f"    {k:<15}: {v:.4f} /hr")
    print(f"    {'합계':<15}: {sum(cfg.arrival_rates.values()):.4f} /hr")

    return run_all(cfg, save_dir=save_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LCL 시뮬레이터")
    parser.add_argument("--olist", metavar="ARCHIVE_DIR",
                        help="Olist CSV 폴더 경로. 지정 시 Olist 기반 물동량으로 실행.")
    parser.add_argument("--total-rate", type=float, default=6.0,
                        help="시간당 총 도착 화물 수 (기본 6.0)")
    parser.add_argument("--olist-cbm", action="store_true",
                        help="Olist 실측 치수 기반 CBM 파라미터 사용")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--save", metavar="DIR", default="outputs",
                        help="결과 저장 폴더 (기본: outputs)")
    args = parser.parse_args()

    if args.olist:
        run_olist(
            archive_dir=args.olist,
            total_rate=args.total_rate,
            use_olist_cbm=args.olist_cbm,
            save_dir=args.save,
            seed=args.seed,
            sim_duration_hours=args.hours,
        )
    else:
        cfg = EnvConfig(seed=args.seed, sim_duration_hours=args.hours)
        run_all(cfg, save_dir=args.save)
