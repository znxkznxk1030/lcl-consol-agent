# LCL Simulator

LCL (Less than Container Load) 컨테이너 통합 시뮬레이션을 위한 프로젝트입니다. 에이전트 기반 시뮬레이션과 웹 인터페이스를 제공합니다.

## 프로젝트 구조

- `agents/`: 에이전트 서버 및 클라이언트 코드
- `server/`: 시뮬레이션 서버 (FastAPI 기반, 포트 8000)
- `web/`: 웹 인터페이스 (HTML/CSS/JavaScript)

## 설치 및 실행

### 요구사항
- Python 3.8 이상
- FastAPI, Uvicorn, Pydantic 등 (의존성 패키지 설치 필요)

### 실행 방법
1. 가상환경 활성화 (선택사항):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   ```

2. 의존성 설치:
   ```bash
   pip install fastapi uvicorn pydantic
   ```
   (추가 의존성이 필요할 수 있음: simulator_v1 등)

3. 서버 실행:
   ```bash
   uvicorn server.main:app --reload
   ```

4. 웹 브라우저에서 [http://localhost:8000](http://localhost:8000) 접속하여 시뮬레이션 인터페이스 사용.

## 사용법

- 웹 인터페이스를 통해 시뮬레이션을 시작하고 모니터링할 수 있습니다.
- 에이전트 서버는 별도로 실행하여 시뮬레이션에 참여할 수 있습니다.

## 기여

이슈나 풀 리퀘스트를 통해 기여해주세요.

## 라이선스

(라이선스 정보 추가)