# Architecture Decision Record (ADR)

> 작성일: 2026-04-15
> 본 문서는 bs_simulator의 알고리즘 아키텍처를 선정하기 위한 논의 과정과 최종 결정을 기록한다.

## 1. 배경 & 문제

- 현재 `optimizers.py`는 `BaseStationOptimizer` 단일 클래스에 4개 알고리즘(K-Means/RW/SA/Tabu)을 메소드로 나열. 알고리즘 추가마다 이 클래스와 `app.py`의 `if-elif` 분기를 모두 수정해야 함.
- `bs_opt` 리포지토리에 Genetic Algorithm이 있어 포팅 예정.
- 장래 강화학습(RL) 알고리즘 도입 가능성 존재.
- 요구: 다양한 상태 공간·목적 함수·알고리즘을 독립적으로 개발 가능해야 함.

## 2. 검토한 아키텍처들

### 옵션 A — 2-Layer (Problem + Algorithm)
- `BaseStationProblem`(상태 + evaluate를 가중치로 파라미터화) + `Solver` ABC
- **장점**: 단순, 즉시 구현 가능
- **단점**: 하나의 목적함수 가족에만 유효. 확장 시 재설계 필요

### 옵션 B — 3-Pillar (StateSpace + Objective + Algorithm)
- 세 축 분리, Context로 공유 데이터 전달
- **장점**: 목적함수 교체 가능
- **단점**: K-Means가 프레임워크 전제(Algorithm이 Objective에 agnostic)를 위반 → 특수 케이스 필요

### 옵션 C — 4-Pillar + Action 카탈로그 (VNS/ALNS 스타일)
- StateSpace가 Action 카탈로그를 제공, Algorithm은 required Actions만 선언
- K-Means의 `centroid_refit`도 하나의 Action으로 통일
- **장점**: 학계 주류 프레임워크(jMetal/DEAP/ParadisEO)와 일치. VNS/ALNS/RL까지 자연스럽게 확장
- **단점**: 추상화 계층 +1, 초기 코드 30~50% 증가. 연구 프레임워크용

### 옵션 D — MDP (State/Action/Reward) 프레이밍
- 메타휴리스틱도 정책으로 재해석하고 RL 알고리즘과 통일
- **장점**: RL 통합 자연스러움
- **단점**: K-Means는 여전히 불일치 (탐색이 아닌 적합). GA population-level과도 어색.

### 옵션 E — Optimizer as Plugin (★ 채택)
- UI가 요구하는 계약은 `(ProblemInput) → (OptimizationResult)` 하나뿐
- 각 알고리즘은 `Optimizer` ABC만 구현. **내부 구조는 완전 자유**
- 메타휴리스틱끼리는 내부 사적 모듈(`metaheuristics/_shared.py`)로 공유 유틸 재사용
- K-Means, RL은 각자 자기 방식대로 구현 — 남의 추상화를 강요받지 않음

## 3. 핵심 통찰

### 통찰 1: UI의 요구사항은 I/O 계약 하나
맵 상태(traffic_map, obstacles, spec) 입력 → 기지국 위치·점수 출력. 내부 직교성(Algorithm ⊥ Objective ⊥ StateSpace)은 **UI 관점에서 무의미한 추상화**.

### 통찰 2: 알고리즘은 실제로 Objective에 종속적
"알고리즘이 Objective에 agnostic"은 편리한 수사일 뿐. 모든 알고리즘은 Objective에 대한 **암묵적 가정**을 가진다:
- K-Means: 유클리드 WCSS 전제
- SA: 작은 이웃 이동에서 부드러움 전제
- Gradient Descent: 미분 가능성 전제
- GA: building-block 가설

완벽한 직교성은 환상. 직교성을 **강제하는** 설계는 leaky abstraction.

### 통찰 3: 내부 공유는 필요한 곳에만
메타휴리스틱 4~5종은 `_calculate_score`, `perturb` 같은 구조를 실제로 공유. 하지만 그 공유는 **메타휴리스틱 내부의 선택**이지, 프레임워크가 모든 알고리즘에 강제할 일이 아니다.

## 4. 채택 — Optimizer as Plugin

```
optimizers/
├── __init__.py              # REGISTRY
├── base.py                  # Optimizer ABC, ProblemInput, OptimizationResult, HyperParam
├── kmeans.py                # sklearn 기반, 10줄 수준
├── metaheuristics/
│   ├── __init__.py
│   ├── _shared.py           # 내부 공유 유틸 (calculate_score, perturb, clip, ...)
│   ├── random_walk.py
│   ├── simulated_annealing.py
│   ├── tabu_search.py
│   └── genetic.py           # bs_opt에서 포팅 (후속 브랜치)
└── rl/                      # 미래 확장
    ├── _env.py              # gymnasium 어댑터
    ├── _reward.py
    ├── dqn.py
    └── ppo.py
```

### 인터페이스
```python
@dataclass(frozen=True)
class ProblemInput:
    X: np.ndarray                 # (N, 2) 좌표
    weights: np.ndarray           # (N,) 트래픽
    width_m: float
    height_m: float
    radius_m: float
    capacity: float
    # Geo 변환용
    lat_min: float; lat_max: float
    lon_min: float; lon_max: float

    @classmethod
    def from_env(cls, env, radius_m, capacity): ...

@dataclass
class OptimizationResult:
    stations: np.ndarray          # (k, 2) local coords
    score: float
    metrics: dict
    history: list[dict] | None = None

@dataclass
class HyperParam:
    name: str
    kind: Literal["int", "float", "choice", "bool"]
    default: Any
    min: Any = None
    max: Any = None
    step: Any = None
    choices: list | None = None
    label: str | None = None

class Optimizer(ABC):
    name: str
    hyperparams: list[HyperParam]

    @abstractmethod
    def optimize(self, problem: ProblemInput, **hp) -> OptimizationResult: ...
```

### UI의 위치
`app.py`는 다음만 한다:
1. 사이드바 값 → `ProblemInput.from_env(env, radius_m, capacity)`
2. `REGISTRY`에서 optimizer 선택
3. `optimizer.hyperparams` 스키마 → 자동 위젯 생성
4. `optimizer.optimize(problem, **hp_values)` 호출
5. `result.stations` → Geo 변환 → 지도 표시 / `result.metrics` → 지표 / `result.history` → 수렴 그래프

**알고리즘 추가 시 `app.py` 수정 불필요**. 레지스트리에 import 한 줄만 추가.

## 5. 비-채택 근거

- **옵션 C (4-Pillar + Action)**: 학술적으로 우월하나 현재 요구사항 대비 과설계. VNS/ALNS를 실제로 구현할 계획이 서기 전까지는 옵션 E 내부에서만 실험.
- **옵션 D (MDP 프레이밍)**: RL 도입 시에도 `Optimizer as Plugin` 안에서 `rl/` 하위로 gym env를 감싸면 충분. MDP를 모든 알고리즘에 강제할 이유 없음.

## 6. 알고리즘별 처리 방식

| 알고리즘 | 구현 위치 | 내부 구조 |
|---|---|---|
| K-Means | `optimizers/kmeans.py` | sklearn.fit → centers |
| Random Walk | `optimizers/metaheuristics/random_walk.py` | `_shared`의 perturb/score 사용 |
| SA | `optimizers/metaheuristics/simulated_annealing.py` | 위와 동일 + 온도 스케줄 |
| Tabu Search | `optimizers/metaheuristics/tabu_search.py` | 위와 동일 + recency mask |
| Genetic Algorithm | `optimizers/metaheuristics/genetic.py` | population + crossover/mutate (bs_opt 포팅) |
| DQN/PPO (미래) | `optimizers/rl/*.py` | gym 환경 생성 → SB3 학습 → rollout |
| Imitation Learning (미래) | `optimizers/rl/imitation.py` | 메타휴리스틱 궤적을 교사로 |

## 7. 마이그레이션 계획 (브랜치 단위)

1. **`refactor/optimizer-plugin-architecture`**: 기존 4개 알고리즘을 새 구조로 이관. 동작은 동일. `app.py`를 레지스트리 기반으로 교체.
2. **`feat/genetic-algorithm`**: bs_opt에서 GA 포팅. Optimizer 계약 준수.
3. **`feat/rl-scaffolding`** (후속): `rl/_env.py`, `_reward.py` 뼈대만. 알고리즘 구현은 별도 브랜치.
4. **`feat/rl-ppo-dqn`** (장래): 실제 RL 알고리즘 구현.

각 브랜치는 `TODO.md` 갱신 + 회귀 테스트(streamlit UI 수동 확인, Playwright 자동화 예정) 포함.

## 8. 회귀 테스트 프로토콜

리팩터링 후 **기존 4개 알고리즘의 결과가 동일**해야 한다:
- 동일 seed, 동일 환경, 동일 하이퍼파라미터로 실행 시 `result.stations`가 기존 출력과 일치
- 다음 최소 시나리오 통과:
  1. 합성 데이터 생성 (resolution=100m, hotspots=5, obstacles=3)
  2. 각 알고리즘 실행 (n_stations=5, radius=300m, capacity=2000)
  3. 지도에 기지국 마커·커버리지 원 표시
  4. 범위 탐색(k=3..7) 수렴 그래프 정상 표시
  5. 데이터 다운로드(CSV/NPY) 정상

## 9. 열린 이슈

- K-Means의 `n_init`, `random_state` 하이퍼파라미터는 `HyperParam` 스키마로 표현 가능하지만, 다른 메타휴리스틱과 이질적 → UI에서 구분 표기 필요?
- 범위 탐색(Range) 기능은 `Optimizer.optimize`를 k마다 반복 호출하는 상위 래퍼로 처리. 현재는 `app.py` 내부, 향후 `benchmarks/runner.py`로 이관 고려.
- Geo ↔ Local 변환 유틸(`convert_to_geo`)은 `optimizers/base.py` 또는 별도 `coords.py`로 이동.
