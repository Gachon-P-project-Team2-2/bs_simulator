# bs_opt ↔ bs_simulator 통합 검토 보고서

> 작성일: 2026-04-15
> 대상 저장소: https://github.com/Gachon-P-project-Team2-2/bs_opt
> 브랜치: `docs/bs-opt-integration-review`

## 1. 결론 (먼저)
**통합 가능하며, 가치가 크다.** 특히 `bs_opt/kmj/` 하위 트리가 잘 구조화되어 있어 **알고리즘 라이브러리**로 흡수하면 bs_simulator의 4개 알고리즘을 5개(GA 추가)로 확장하고 **수렴 히스토리 기반 시각화**까지 얻을 수 있다. 다만 **좌표계·평가 방식·용량 개념**에서 비호환 지점이 있어 어댑터 레이어가 필요하다.

권장 경로: **Phase 1 — GA만 cherry-pick / Phase 2 — Solver 추상화 흡수 / Phase 3 — 완전 통합**.

---

## 2. bs_opt 구조 개요

```
bs_opt/
├── gmj/    # 개인 실험용 (Jupyter notebooks)
└── kmj/    # 프로덕션급 구조 — 통합 대상
    ├── main.py               # CLI 드라이버
    ├── core/
    │   ├── grid.py           # 8종 트래픽 패턴 생성기
    │   └── model.py          # BaseStationProblem, CostWeights
    └── algos/
        ├── base.py           # Solver, SolverResult (abstract)
        ├── random_walk.py
        ├── simulated_annealing.py
        ├── tabu_search.py
        └── genetic.py        # ✨ bs_simulator에 없음
```

### 핵심 추상화 (bs_opt/kmj)
- `BaseStationProblem`: traffic_map + n_bs + coverage_radius + cost_weights → `evaluate(state) → (cost, metrics)`
- `Solver` 인터페이스: `run(max_iter, init_state) → SolverResult(best_state, best_cost, best_metrics, history)`
- 모든 알고리즘이 동일 인터페이스 준수 → 플러그인화 가능
- **`history`** — iteration마다 best_cost/coverage 기록 → 수렴 그래프 자동 생성

---

## 3. 비교표

| 항목 | bs_simulator | bs_opt (kmj) |
|---|---|---|
| **실행 형태** | Streamlit 웹 UI | Python CLI (main.py) |
| **좌표계** | Geo(lat/lon) + Local(m) | 격자 정수 (x, y) ∈ [0, W/H) |
| **반경 단위** | 미터 | 격자 셀 |
| **트래픽 패턴** | 1종 (가우시안 핫스팟) | **8종** (random / center_hotspot / multi_hotspot / ring / gradient / stripe / checkerboard / random_clusters) |
| **장애물** | ✅ (5종 패턴, 마스킹) | ❌ |
| **평가 방향** | **score 최대화** | **cost 최소화** (부호 반대) |
| **평가 수식** | `Σmin(load, capacity) + 0.1·coverage_grids` | `(1 - coverage_ratio)·w_u + n_bs·w_bs` |
| **용량(capacity)** | ✅ 기지국별 제한, overload 구분 | ❌ 무제한 |
| **알고리즘** | K-Means / RW / SA / Tabu | RW / SA / Tabu / **GA** |
| **상태 타입** | `np.ndarray` float (n_bs, 2) | `np.ndarray` int (n_bs, 2) |
| **수렴 히스토리** | ❌ | ✅ `list[dict]` |
| **의존성** | streamlit, folium, sklearn | numpy, matplotlib only |
| **LoC (핵심)** | 803 | 1,153 |

---

## 4. 비호환 지점 상세

### 4.1 평가 방향 (score ↔ cost)
- bs_simulator: `score = 높을수록 좋음`, 4개 알고리즘 모두 `>` 비교
- bs_opt: `cost = 낮을수록 좋음`, 모두 `<` 비교
- **해결**: 어댑터에서 `score = -cost` 또는 `cost = total_traffic - score`. **통일 규약**은 bs_simulator 측(score↑) 유지 권장 — UI가 "큰 숫자일수록 좋음" 전제로 구성됨.

### 4.2 좌표계 & 반경
- bs_opt는 격자 인덱스 (정수) 기반 / coverage_radius는 격자 셀 수
- bs_simulator는 미터(float) 기반 / radius_m
- **해결**: `radius_grid = radius_m / resolution_m` 변환. state도 `int(x/resolution_m)` 매핑. 단, 연속값 최적화의 정밀도를 잃을 수 있음 → 혹은 bs_opt 알고리즘을 float으로 일반화.

### 4.3 용량(capacity) 개념 부재
- bs_opt의 `evaluate`는 "커버됐다 / 안 됐다"만 판단. bs_simulator는 거리순 할당 → 용량 초과 시 overload.
- **해결**: `BaseStationProblem`에 `capacity` 필드 추가 + `evaluate`에서 거리순 할당 로직 도입. 또는 capacity-aware cost function을 cost_weights 확장으로 구현.

### 4.4 장애물
- bs_opt에 장애물 개념이 없음. state가 장애물 내부에 생성되어도 패널티 없음.
- **해결**: bs_simulator의 `traffic_map`은 이미 마스킹된 상태이므로, 장애물 내부 셀은 traffic=0 → 자연히 평가에서 제외됨. 추가 작업 불필요.

### 4.5 K-Means 결측
- bs_opt에 K-Means 없음. 반대 방향(bs_opt에 추가)은 sklearn 의존 증가로 바람직하지 않음. bs_simulator의 K-Means를 유지하면 됨.

---

## 5. 통합 로드맵

### Phase 1 — GA만 cherry-pick (저위험 / 1~2 커밋)
- `optimizers.py`에 `run_genetic(n_stations, pop_size, n_generations, ...)` 메소드 추가
- `bs_opt/kmj/algos/genetic.py`의 로직을 bs_simulator의 `_calculate_score` 규약에 맞춰 포팅
- `app.py` selectbox에 "Genetic Algorithm" 항목 추가 + 하이퍼파라미터 expander
- **이득**: 알고리즘 5개 확보. 아키텍처는 그대로.
- **비용**: ~200 LoC 추가.

### Phase 2 — Solver/Problem 추상화 도입 (중위험 / 3~5 커밋)
- `optimizers/` 디렉토리 생성 → `base.py`, `problem.py`, `<algo>.py` 파일 분리
- `BaseStationOptimizer`를 `Solver` + `BaseStationProblem`으로 분할
- 각 알고리즘이 `Solver` 상속하도록 리팩터링
- `history` 필드 추가 → 수렴 그래프 UI 추가
- **이득**: 알고리즘 추가가 플러그인화. bs_opt 측 업데이트 흡수 쉬움.
- **비용**: `app.py`의 `if-elif algo ==` 분기를 dynamic dispatch로 전환 필요. 기존 기능 회귀 테스트 필수.

### Phase 3 — 완전 통합 (고위험 / 장기)
- bs_opt를 git submodule 또는 Python 패키지(`pip install -e ../bs_opt/kmj`)로 의존
- 트래픽 생성 로직 공유 (bs_opt의 8종 패턴을 bs_simulator UI에 노출)
- 양 프로젝트가 동일한 `Problem`/`Solver` 인터페이스로 수렴
- **이득**: 두 팀 작업의 단일 진실 원본.
- **비용**: 인터페이스 합의, CI 설정, 양쪽 팀 간 조율.

---

## 6. 바로 수확 가능한 것들 (integration과 무관하게 copy-paste 가능)

- **트래픽 패턴 확장**: `bs_opt/kmj/core/grid.py::generate_synthetic_traffic`의 패턴 분기 (~170 LoC) → bs_simulator의 `SyntheticEnvironment.generate_traffic`에 포팅 가능. UI `selectbox`에 8종 노출.
- **수렴 히스토리 시각화**: 범위 탐색(Range) 외에 단일 k에 대한 iteration vs best_score 그래프.
- **GA 알고리즘**: 단독 포팅 가능 (Phase 1 참고).

---

## 7. 위험 & 주의사항
- bs_opt는 `np.random.default_rng(42)`로 전역 시드 고정 → bs_simulator의 `random_state=42` 관행과 일관성 있음. 단, Generator API vs legacy `np.random` 혼용 주의.
- bs_opt의 integer 좌표는 부드러운 이동(step_size=0.5m 같은 미세 이동)을 불가능하게 함 → bs_simulator의 RW/SA/Tabu와 수렴 속도가 달라질 수 있음.
- 라이선스: bs_opt에 LICENSE 파일 존재 — 통합 전 호환성 확인 필요.

---

## 8. 권장 다음 단계
1. 이 보고서를 바탕으로 **Phase 1(GA 추가)** 결정 → TODO.md에 `대기` 항목 추가
2. 트래픽 패턴 8종 추가도 별도 작업으로 TODO.md에 기록
3. Phase 2/3은 bs_opt 팀과의 조율이 필요하므로 별도 논의
