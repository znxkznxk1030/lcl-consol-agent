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

3. 실행:

   | 명령어 | 설명 |
   |--------|------|
   | `python run.py sim` | 시뮬레이션 단독 실행 (터미널 결과 출력) |
   | `python run.py server` | 시뮬레이션 서버 실행 (`:8000`, 환경/state/dispatch 제공) |
   | `python run.py agent` | LLM AI Agent 서버 실행 (`:8001`, `/decide` 제공) |
   | `python run.py all` | 두 서버 동시 실행 (Ctrl+C로 종료) |

4. 웹 브라우저에서 [http://localhost:8000](http://localhost:8000) 접속하여 시뮬레이션 인터페이스 사용.

## 사용법

- 시뮬레이션 서버는 화물 도착, 버퍼 상태, dispatch 적용만 담당합니다.
- 에이전트 서버는 현재 state를 입력받아 LLM 기반으로 출하 결정을 반환합니다.
- 웹 인터페이스의 `Ask Agent`는 에이전트 서버의 `/decide`를 호출합니다.

## LLM 연결 설정

에이전트 서버는 LLM SDK와 API 키가 모두 준비되어야 실제 LLM으로 동작합니다.
설정이 없으면 서버는 실행되지만 `fallback mode`로 동작합니다.

### 1. SDK 설치

사용할 provider에 맞는 패키지를 설치합니다.

```bash
pip install anthropic
pip install openai
pip install google-generativeai
```

필요한 provider만 설치해도 됩니다.

### 2. 환경변수 설정

#### Anthropic 사용 예시

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=YOUR_KEY
export LLM_MODEL=claude-sonnet-4-6
```

#### OpenAI 사용 예시

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=YOUR_KEY
export LLM_MODEL=gpt-4o
```

#### Google Gemini 사용 예시

```bash
export LLM_PROVIDER=google
export GOOGLE_API_KEY=YOUR_KEY
export LLM_MODEL=gemini-1.5-flash
```

추가로 기본 컨테이너 타입을 지정할 수 있습니다.

```bash
export DEFAULT_CONTAINER_TYPE=40GP
```

### 3. Agent 서버 실행

```bash
python run.py agent
```

또는 시뮬레이터와 함께 실행:

```bash
python run.py all
```

### 4. 연결 확인

아래 endpoint로 실제 LLM 연결 여부를 확인할 수 있습니다.

```bash
curl http://localhost:8001/health
```

정상적으로 LLM이 연결되면 아래처럼 보입니다.

```json
{
  "ok": true,
  "agent": "llm-ai",
  "ai_agent_available": true,
  "llm_enabled": true,
  "ai_fallback_mode": false
}
```

반대로 아래 상태면 에이전트 객체는 생성됐지만 실제 LLM은 연결되지 않은 상태입니다.

```json
{
  "ok": true,
  "agent": "llm-ai",
  "ai_agent_available": true,
  "llm_enabled": false,
  "ai_fallback_mode": true
}
```

이 경우에는 보통 다음 중 하나입니다.

- API 키가 설정되지 않음
- 해당 provider SDK가 설치되지 않음
- `LLM_PROVIDER`와 API 키 종류가 맞지 않음

---

## Olist 실제 데이터 기반 물동량 시뮬레이션

[Olist Brazilian E-Commerce 데이터셋](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)의 CSV 파일 분포를 활용하여
실제 카테고리 비율과 물동량 패턴으로 시뮬레이션을 실행할 수 있습니다.

### 필요 파일

`archive/` 폴더에 아래 CSV 파일이 있어야 합니다.

```
archive/
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_products_dataset.csv
└── product_category_name_translation.csv
```

### 추가 의존성 설치

```bash
pip install pandas numpy
```

### Python 코드에서 사용

```python
from simulator_v1.run import run_olist

# 기본 실행 — Olist 카테고리 비율로 시간당 총 6건 도착
results = run_olist("../archive", total_rate=6.0)

# 물동량 규모 조정 (시간당 총 도착 건수만 변경, 비율은 Olist 분포 유지)
results = run_olist("../archive", total_rate=10.0, sim_duration_hours=168)

# CBM 파라미터도 Olist 실측 치수 기반으로 교체
results = run_olist("../archive", total_rate=6.0, use_olist_cbm=True)
```

### CLI에서 사용

```bash
# lcl_simulator/ 디렉터리에서 실행
python -m simulator_v1.run --olist ../archive
python -m simulator_v1.run --olist ../archive --total-rate 10.0 --hours 168
python -m simulator_v1.run --olist ../archive --olist-cbm --save outputs/olist
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--olist ARCHIVE_DIR` | (없음) | Olist CSV 폴더 경로. 미지정 시 기본 통계 기반 실행 |
| `--total-rate FLOAT` | `6.0` | 시간당 총 도착 화물 수 |
| `--olist-cbm` | `False` | Olist 실측 치수 → LCL 환산 CBM 파라미터 사용 |
| `--seed INT` | `42` | 난수 시드 |
| `--hours INT` | `72` | 시뮬레이션 기간 (시간) |
| `--save DIR` | `outputs` | 결과 JSON 저장 폴더 |

### Olist에서 추출되는 카테고리 비율

Olist 데이터(110,197건) 기준 ItemType별 비율 (total_rate=6.0 기준):

| ItemType | 시간당 도착 | 비율 | 포함 Olist 카테고리 |
|----------|------------|------|---------------------|
| CLOTHING | 2.82 /hr | 47% | fashion_*, bed_bath_table, housewares, home_confort |
| ELECTRONICS | 1.40 /hr | 23% | electronics, computers, telephony 등 |
| COSMETICS | 1.06 /hr | 18% | health_beauty, perfumery 등 |
| AUTO_PARTS | 0.49 /hr | 8% | auto, construction_tools_*, home_construction |
| FURNITURE | 0.27 /hr | 5% | furniture_* (실제 가구류만), office_furniture |
| MACHINERY | 0.17 /hr | 3% | small_appliances, home_appliances 등 |
| FOOD_PRODUCTS | 0.09 /hr | 2% | food, drinks 등 |
| CHEMICALS | 0.04 /hr | 1% | agro_industry, industry_commerce |

> **FURNITURE 분류 기준**: `bed_bath_table`(침구·수건류)·`housewares`(소형 생활용품)은 개별 CBM이 작아 FURNITURE 버킷의 CBM 분포를 왜곡하므로 CLOTHING으로 분리했습니다. FURNITURE에는 실제 부피가 큰 가구류(`furniture_*`, `office_furniture`, `kitchen_dining_*`)만 포함합니다.

### 데이터 출처

Olist Brazilian E-Commerce Public Dataset

- **출처**: [https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **제공**: Olist (브라질 이커머스 플랫폼)
- **라이선스**: CC BY-NC-SA 4.0
- **기간**: 2016년 9월 ~ 2018년 8월 (약 17,000시간)
- **규모**: 배송 완료 주문 기준 약 110,000건의 주문 아이템

### 보정 방식 참고

| 항목 | 설명 |
|------|------|
| **도착률** | Olist 주문 타임스탬프에서 카테고리별 시간당 건수 계산 후 `total_rate`로 정규화 |
| **CBM (기본)** | 기존 학술 기반 LCL 로그 정규 파라미터 유지 |
| **CBM (`--olist-cbm`)** | Olist 실측 상품 치수에 LCL 상업 화물 스케일 팩터 적용 (예: ELECTRONICS ×72) |

## 기여

이슈나 풀 리퀘스트를 통해 기여해주세요.

## 라이선스

(라이선스 정보 추가)
