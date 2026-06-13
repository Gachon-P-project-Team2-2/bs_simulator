# BS Simulator

Dash 기반 기지국 위치 최적화 시뮬레이터입니다. 지도 위에서 분석 영역을 지정하고, 합성/OSM/GeoJSON 오브젝트와 정적 또는 동적 트래픽을 생성한 뒤, 여러 최적화 알고리즘으로 기지국 배치를 계산합니다. 결과는 지도 view와 분석 결과 모음 view에서 확인할 수 있습니다.

## 주요 기능

- 지도 기반 영역 지정 및 가상 환경 생성
- 정적 트래픽 패턴 생성: `random_clusters`, `multi_hotspot`, `center_hotspot`, `random`, `ring`, `gradient`, `stripe`, `checkerboard`
- 동적 트래픽 모드: 고정 위치 변동, 이동형 핫스팟, 위치 전환형
- 오브젝트 생성/가져오기: 합성 오브젝트, OSM 지도 데이터, GeoJSON 업로드
- SINR 기반 커버리지 평가, Shannon/MCS 스펙트럼 효율 모델
- 기지국 송신 전력/대역폭/전파 모델 설정, 전체 동일 또는 기지국별 개별 모델
- 최적화 알고리즘: K-Means, Random Walk, Simulated Annealing, Tabu Search, Genetic Algorithm, DQN Placement
- Sweep 모드 1: 선택 알고리즘의 하이퍼파라미터별 성능 비교 및 결과 적용
- Sweep 모드 2: 여러 알고리즘 비교 실행 및 선택 결과 적용
- 동적 트래픽 재생 중 평가 결과와 기지국 운영 상태 반영
- 운영 최적화: `always-on`, `threshold`, `two-threshold`, `greedy-off`, `dqn` 정책 비교
- 중앙 view 전환: `1`은 지도 view, `2`는 분석 결과 모음 view

## 빠른 시작

### 1. 의존성 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

개발/회귀 테스트까지 실행하려면:

```bash
pip install -r requirements-dev.txt
python -m playwright install chromium
```

### 2. 앱 실행

```bash
python app.py
```

기본 Dash 개발 서버가 실행됩니다. 포트를 지정하려면 `DASH_PORT`를 사용합니다.

```bash
DASH_PORT=8052 python app.py
```

### 3. 기본 사용 흐름

1. 좌측 사이드바에서 격자 크기, 트래픽 세부 설정, 오브젝트 세부 설정을 조정합니다.
2. `영역 지정`을 눌러 지도 위에 사각형 분석 영역을 그립니다.
3. 영역 확인 후 `가상 데이터 생성`으로 환경을 생성합니다.
4. 우측 `알고리즘` 탭에서 최적화 목표, 기지국 수, 전파 모델, 알고리즘, 기지국 모델을 설정합니다.
5. `계산 실행`을 눌러 기지국 위치 최적화를 수행합니다.
6. 동적 트래픽 모드인 경우 `운영 최적화 실행`으로 시간대별 ON/OFF 정책을 평가할 수 있습니다.
7. 중앙 상단의 `1`, `2` 버튼으로 지도 view와 분석 결과 모음 view를 전환합니다.

## 화면 구성

### 좌측 사이드바

- `격자 크기 (m)`: 환경 그리드 해상도입니다. 작을수록 세밀하지만 계산량이 증가합니다.
- `트래픽 세부 설정`
  - `동적 트래픽 모드`: 시간 프레임을 가진 트래픽 시나리오를 생성합니다.
  - `트래픽 패턴`: 기본값은 `random_clusters`입니다.
  - `총 면적 수요 (Mbps/km²)`: 면적 단위 트래픽 밀도입니다.
  - `핫스팟 개수`, `핫스팟 확산 반경 (m)`: 핫스팟 계열 패턴의 공간 분포를 조절합니다.
  - 동적 모드에서는 프레임 수, 시간 변화 강도, 공간 이동 범위를 추가로 설정합니다.
- `오브젝트 세부 설정`
  - `합성`: random/mixed/circle/strip/grid 패턴으로 장애물성 오브젝트를 생성합니다.
  - `OSM 지도 데이터`: 건물, 수역/물길, 도로 오브젝트를 지도 영역에서 가져옵니다.
  - `GeoJSON 업로드`: 업로드한 폴리곤을 필터링해 오브젝트로 사용합니다.

### 중앙 view

- `1`: 지도 view입니다. 트래픽, 오브젝트, 기지국, 커버리지 상태, 동적 프레임 재생 UI를 표시합니다.
- `2`: 분석 결과 모음 view입니다. 환경 요약, 현재 프레임 성능, 동적 시나리오 요약, 최적화 결과, 기지국별 분석, 운영 최적화 결과, Sweep 결과, 알고리즘 비교 결과를 한 화면에 모읍니다.

### 우측 사이드바

- `알고리즘` 탭
  - 최적화 목표: 트래픽 커버리지 또는 커버 셀 수
  - 기지국 수
  - 전파 모델: 경로 손실 지수, SINR 임계값, CoMP 조율 기지국 수
  - 알고리즘 선택 및 알고리즘별 하이퍼파라미터
  - 운영 최적화: 동적 트래픽 모드에서 별도 실행
  - 기지국 모델: 전체 동일 또는 기지국별 개별 송신 전력/대역폭 설정
- `Sweep` 탭
  - 모드 1: 현재 알고리즘의 파라미터 sweep
  - 모드 2: 여러 알고리즘의 성능 비교
  - 결과 테이블에서 행을 선택해 지도 결과로 적용할 수 있습니다.

## 트래픽 및 오브젝트 모델

트래픽은 `environment.SyntheticEnvironment`가 로컬 미터 좌표계와 지리 좌표계를 함께 관리합니다.

- Local 좌표: 분석 영역 좌상단 기준 `(x, y)` 미터 좌표
- Geo 좌표: 지도 표시용 위도/경도
- 정적 트래픽: `patterns.generate_pattern` 결과를 면적 수요 기반 Mbps 셀 수요로 변환
- 동적 트래픽: `(time_steps, rows, cols)` 형태의 시계열 트래픽 맵 생성
- 오브젝트 마스킹: 오브젝트 내부 그리드 셀의 트래픽을 제거 또는 억제하여 계산에 반영

동적 트래픽 유형은 다음 의미를 가집니다.

- `fixed_variation`: 같은 위치의 트래픽 분포 강도만 시간에 따라 변동
- `moving_hotspot`: 트래픽 분포가 프레임별로 이동
- `switching_locations`: 서로 다른 위치의 분포가 나타났다 사라짐

## 최적화 모델

알고리즘은 `optimizers/` 패키지의 공통 계약을 따릅니다.

- `ProblemInput`: 환경, 기지국 수, 전파 파라미터, 송신 전력, 대역폭, SINR 임계값을 담는 입력
- `Optimizer`: 각 알고리즘이 구현하는 공통 인터페이스
- `OptimizationResult`: 기지국 좌표, 점수, 메트릭, 수렴 이력

현재 등록된 알고리즘은 `optimizers.REGISTRY` 순서대로 UI에 표시됩니다.

- `K-Means`
- `Random Walk`
- `Simulated Annealing`
- `Tabu Search`
- `Genetic Algorithm`
- `DQN Placement`

평가 지표는 선택한 최적화 목표에 따라 달라집니다.

- 트래픽 커버리지: SINR 임계값 이상인 셀의 트래픽 합
- 커버 셀 수: SINR 임계값 이상인 셀 개수

지도와 분석 view에는 총 트래픽, 커버된 트래픽, 커버율, 처리량, 에너지 효율, 평균 SINR, 기지국별 부하 등이 표시됩니다.

## 운영 최적화

동적 트래픽 모드에서 최적화 결과가 있을 때 운영 최적화를 별도로 실행할 수 있습니다. 목적은 시간 프레임별 기지국 ON/OFF 정책을 평가해 운영 비용(OPEX)을 비교하는 것입니다.

지원 정책:

- `always-on`: 모든 기지국을 항상 활성 상태로 유지
- `threshold`: 부하가 임계값보다 낮은 기지국을 절전 후보로 판단
- `two-threshold`: Sleep/Wake 임계값을 분리해 빈번한 상태 전환을 완화
- `greedy-off`: 한 기지국씩 끄는 후보를 비교하며 비용이 낮아지는 선택을 적용
- `dqn`: 독립 DQN 에이전트로 ON/OFF 액션을 학습

운영 결과는 기준 정책과 선택 정책의 총 OPEX, 에너지, 전환 횟수, 미커버/과부하 패널티, 활성 기지국 수를 비교합니다. 지도 재생 중에는 해당 프레임의 운영 정책에 따른 기지국 ON/SLEEP 상태도 반영됩니다.

## 파일 구조

```text
.
├── app.py                         # Dash UI, 콜백, 세션 상태, 지도/분석 렌더링
├── environment.py                 # 합성 환경, 트래픽/오브젝트 생성, 좌표 변환
├── obstacle_sources.py            # OSM/GeoJSON 오브젝트 로딩 및 필터링
├── patterns.py                    # 정적/동적 트래픽 패턴 생성기
├── optimizers/
│   ├── base.py                    # ProblemInput, Optimizer, 메트릭/SINR 계산
│   ├── kmeans.py                  # K-Means 최적화
│   ├── metaheuristics/            # Random Walk, SA, Tabu, GA
│   └── drl/                       # DQN 배치 및 운영 제어 에이전트
├── assets/design.css              # Dash 전역 스타일
├── tests/                         # 단위/Playwright 회귀 테스트
├── requirements.txt               # 실행 의존성
└── requirements-dev.txt           # 테스트 의존성
```

## 테스트

문법 확인:

```bash
python -m py_compile app.py tests/test_regression.py tests/test_dynamic_traffic.py
```

동적 트래픽/운영 최적화 단위 테스트:

```bash
pytest tests/test_dynamic_traffic.py -q
```

Playwright 기반 UI 회귀 테스트:

```bash
pytest tests/test_regression.py -q
```

전체 테스트:

```bash
pytest -q
```

`tests/conftest.py`는 테스트용 Dash 서버를 임의 포트로 자동 실행합니다.

## 로그와 산출물

- 최적화 로그: `logs/optimizer.log`
- Dash 정적 자산: `assets/`
- 테스트/디버깅 중 생성된 스크린샷이나 캐시는 보통 커밋 대상이 아닙니다.

## 개발 메모

- 자세한 개발자용 구조/확장 가이드는 `docs/DEVELOPER_GUIDE.md`를 참고하세요.
- 새 알고리즘은 `optimizers.Optimizer`를 구현하고 `optimizers/__init__.py`의 `REGISTRY`에 추가하면 UI에 노출됩니다.
- 새 트래픽 패턴은 `patterns.PATTERN_CHOICES`와 `generate_pattern`에 추가하면 좌측 트래픽 패턴 드롭다운에 반영됩니다.
- 분석 결과 모음 view에 새 결과를 추가하려면 `render_analysis_view`의 섹션 구성을 확장합니다.
- 운영 정책을 추가하려면 `OPERATION_POLICY_OPTIONS`, `OPERATION_POLICY_PARAM_NAMES`, `evaluate_operation_optimization`을 함께 갱신합니다.
