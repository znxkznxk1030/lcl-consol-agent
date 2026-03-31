#!/usr/bin/env python3
"""
run.py — LCL Simulator CLI
===========================
사용법:
  python run.py sim        # 시뮬레이션 단독 실행 (터미널 출력)
  python run.py server     # 시뮬레이션 서버 실행 (:8000)
  python run.py agent      # 에이전트 서버 실행 (:8001)
  python run.py all        # 시뮬레이션 서버 + 에이전트 서버 동시 실행
"""

import sys
import subprocess


def run_sim():
    subprocess.run([sys.executable, "-m", "simulator_v1.run"])


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


def run_all():
    import signal
    import os

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

    print("Simulation server : http://localhost:8000")
    print("Agent server      : http://localhost:8001")
    print("Ctrl+C to stop both servers.")

    def _shutdown(sig, frame):
        for p in procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    for p in procs:
        p.wait()


COMMANDS = {
    "sim": run_sim,
    "server": run_server,
    "agent": run_agent,
    "all": run_all,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
