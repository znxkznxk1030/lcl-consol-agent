#!/usr/bin/env python3
"""
run.py — LCL Simulator CLI
===========================
사용법:
  python run.py sim        # 시뮬레이션 단독 실행 (규칙 기반 에이전트)
  python run.py server     # 시뮬레이션 서버 실행 (:8000)
  python run.py agent      # LLM AI Agent 서버 실행 (:8001)
  python run.py greedy     # Greedy Agent 서버 실행 (:8002, LLM 없음)
  python run.py rl-train   # MAPPO RL 에이전트 학습
  python run.py rl-eval    # MAPPO vs Baseline 비교 평가
  python run.py rl-server  # MAPPO RL 에이전트 서버 실행 (:8003)
  python run.py all        # 시뮬레이션 서버 + 에이전트 서버 동시 실행
  python run.py all-greedy # 시뮬레이션 서버 + greedy 서버 동시 실행
  python run.py all-rl     # 시뮬레이션 서버 + RL 에이전트 서버 동시 실행

LLM Agent 환경변수:
  ANTHROPIC_API_KEY       # Claude API 키 (없으면 fallback 모드)
  DEFAULT_CONTAINER_TYPE  # 기본 컨테이너 타입 (기본: 40GP)
"""

import sys
import subprocess


def run_sim():
    subprocess.run([sys.executable, "-m", "simulator_v1.run"])


def run_ai_sim():
    raise SystemExit("`ai_sim` is deprecated. Run `python run.py server` and `python run.py agent` instead.")


def run_server():
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "server.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ])


def run_agent():
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "agents.agent_server:app",
        "--host", "0.0.0.0",
        "--port", "8001",
        "--reload",
    ])


def run_greedy():
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "agents.greedy_agent_server:app",
        "--host", "0.0.0.0",
        "--port", "8002",
        "--reload",
    ])


def _run_procs(procs: list, labels: list[str]) -> None:
    import signal

    for label in labels:
        print(label)
    print("Ctrl+C to stop all servers.")

    def _shutdown(sig, frame):
        for p in procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    for p in procs:
        p.wait()


def run_all():
    procs = [
        subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "server.main:app",
            "--host", "0.0.0.0", "--port", "8000", "--reload",
        ]),
        subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "agents.agent_server:app",
            "--host", "0.0.0.0", "--port", "8001", "--reload",
        ]),
    ]
    _run_procs(procs, [
        "Simulation server : http://localhost:8000",
        "Agent server (LLM): http://localhost:8001",
    ])


def run_all_greedy():
    procs = [
        subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "server.main:app",
            "--host", "0.0.0.0", "--port", "8000", "--reload",
        ]),
        subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "agents.greedy_agent_server:app",
            "--host", "0.0.0.0", "--port", "8002", "--reload",
        ]),
    ]
    _run_procs(procs, [
        "Simulation server   : http://localhost:8000",
        "Greedy agent server : http://localhost:8002",
    ])


def run_rl_train():
    subprocess.run([sys.executable, "-m", "rl.train"] + sys.argv[2:])


def run_rl_eval():
    subprocess.run([sys.executable, "-m", "rl.evaluate"] + sys.argv[2:])


def run_rl_server():
    subprocess.run([sys.executable, "-m", "rl.server"] + sys.argv[2:])


def run_all_rl():
    procs = [
        subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "server.main:app",
            "--host", "0.0.0.0", "--port", "8000", "--reload",
        ]),
        subprocess.Popen([
            sys.executable, "-m", "rl.server",
            "--checkpoint", "checkpoints/mappo_best.pt",
            "--port", "8003",
        ]),
    ]
    _run_procs(procs, [
        "Simulation server  : http://localhost:8000",
        "RL agent server    : http://localhost:8003",
    ])


COMMANDS = {
    "sim":        run_sim,
    "ai_sim":     run_ai_sim,
    "server":     run_server,
    "agent":      run_agent,
    "greedy":     run_greedy,
    "rl-train":   run_rl_train,
    "rl-eval":    run_rl_eval,
    "rl-server":  run_rl_server,
    "all":        run_all,
    "all-greedy": run_all_greedy,
    "all-rl":     run_all_rl,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
