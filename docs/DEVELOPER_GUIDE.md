# Developer Guide

이 문서는 BS Simulator를 수정하거나 기능을 확장하는 개발자를 위한 내부 구조 안내서입니다. 사용자용 실행/기능 요약은 루트 `README.md`를 보고, 구현을 바꿀 때는 이 문서를 기준으로 관련 모듈과 상태 흐름을 확인하세요.

## 프로젝트 지도

| 영역 | 주요 파일 | 역할 |
| --- | --- | --- |
| Dash 앱/콜백 | `app.py` | 레이아웃, 세션 상태, 지도 렌더링, 최적화/운영/Sweep 콜백 |
| 환경 모델 | `environment.py` | 분석 영역, 로컬/지리 좌표, 트래픽/오브젝트 생성, 마스킹 |
| 트래픽 패턴 | `patterns.py` | 정규화된 정적 패턴 생성기와 패턴 목록 |
| 오브젝트 소스 | `obstacle_sources.py` | OSM/GeoJSON 폴리곤 로딩, 캐시, 필터링 |
| 최적화 API | `optimizers/base.py` | `ProblemInput`, `Optimizer`, `OptimizationResult`, SINR/메트릭 계산 |
| 최적화 등록 | `optimizers/__init__.py` | UI에 노출되는 알고리즘 `REGISTRY` |
| 알고리즘 구현 | `optimizers/kmeans.py`, `optimizers/metaheuristics/`, `optimizers/drl/` | K-Means, 메타휴리스틱, DQN 알고리즘 |
| 스타일 | `assets/design.css` | Dash 전역 스타일과 UI 컴포넌트 스타일 |
| 테스트 | `tests/` | 단위 테스트, 동적 트래픽/운영 테스트, Playwright UI 회귀 테스트 |

## 실행과 테스트

런타임 의존성:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

개발 및 UI 테스트 의존성:

```bash
pip install -r requirements-dev.txt
python -m playwright install chromium
```

앱 실행:

```bash
python app.py
```

포트 지정:

```bash
DASH_PORT=8052 python app.py
```

빠른 검증:

```bash
python -m py_compile app.py tests/test_regression.py tests/test_dynamic_traffic.py
pytest tests/test_dynamic_traffic.py -q
pytest tests/test_regression.py -q
```

전체 검증:

```bash
pytest -q
```

`tests/conftest.py`는 Playwright 테스트 실행 시 Dash 서버를 임의 포트로 자동 시작합니다. 이미 실행 중인 수동 개발 서버와 별개로 동작합니다.

## 상태와 콜백 모델

이 앱은 브라우저의 `dcc.Store`에 큰 객체를 저장하지 않습니다. 실제 환경/최적화 결과는 서버 메모리의 `APP_STATE`에 있고, 브라우저 store는 세션 ID와 변경 알림용 version token 역할을 주로 합니다.

핵심 흐름:

1. `serve_layout()`가 새 `session-id`를 만들고 각종 `dcc.Store`와 `dcc.Interval`을 배치합니다.
2. `get_session_state(session_id)`가 `APP_STATE[session_id]`를 반환하고, 오래 접근하지 않은 세션을 정리합니다.
3. 환경 생성, 최적화 실행, Sweep, 알고리즘 비교, 운영 최적화 콜백은 `APP_STATE`를 갱신합니다.
4. 갱신 후 `version_token()` 형태의 작은 메타 데이터를 store에 넣어 관련 렌더링 콜백을 깨웁니다.

주요 store:

| Store | 용도 |
| --- | --- |
| `session-id` | 서버 메모리 상태를 찾는 키 |
| `env-meta` | 환경 생성/변경 알림 |
| `opt-meta` | 최적화 결과 변경 알림 |
| `operation-meta` | 운영 최적화 결과 변경 알림 |
| `range-meta` | 기지국 수 범위 탐색 결과 변경 알림 |
| `sweep-meta` | Sweep 결과 변경 알림 |
| `algo-compare-meta` | 알고리즘 비교 결과 변경 알림 |
| `station-specs-store` | 기지국별 전력/대역폭 설정 |
| `selected-station` | 지도에서 선택된 기지국 |

`APP_STATE`에서 자주 쓰는 키:

| Key | 의미 |
| --- | --- |
| `env` | `SyntheticEnvironment` 인스턴스 |
| `opt_results` | 현재 지도에 적용된 기지국 배치 결과 |
| `opt_stats` | 현재 결과의 메트릭 |
| `range_results` | 기지국 수 범위 탐색 결과 |
| `operation_results` | 시간 프레임별 운영 정책 결과 |
| `sweep_results` | Sweep 모드 1 결과 |
| `algo_compare_results` | Sweep 모드 2 알고리즘 비교 결과 |
| `opt_progress`, `sweep_progress`, `algo_compare_progress` | 백그라운드 작업 진행 상태 |

환경이나 최적화 결과를 새로 만들 때는 오래된 결과를 함께 지워야 합니다. 예를 들어 환경 재생성 시 `opt_results`, `opt_stats`, `operation_results`, `sweep_results`, `algo_compare_results`가 남아 있으면 지도와 분석 view가 서로 다른 기준의 결과를 섞어 보여줄 수 있습니다.

## 백그라운드 작업

긴 계산은 Dash 요청 스레드에서 직접 수행하지 않고 daemon thread로 실행합니다.

- 일반 최적화: `start_optimization_job()`에서 설정을 만들고 `_run_optimization_thread()`가 실행합니다.
- Sweep 모드 1: `start_sweep_job()`과 `_run_sweep_thread()`가 처리합니다.
- Sweep 모드 2: `start_algo_compare_job()`과 `_run_algo_compare_thread()`가 처리합니다.

각 작업은 `APP_STATE`의 progress 키를 갱신하고, 대응하는 `dcc.Interval` 콜백이 상태를 polling합니다. 새 백그라운드 작업을 추가할 때는 다음 규칙을 지키세요.

- 시작 콜백에서 중복 실행을 막는 `running` 체크를 둡니다.
- 스레드 내부에서 예외를 잡아 progress의 `error`에 traceback을 저장합니다.
- 완료 시 `running=False`, `done=True`를 먼저 보장하고 결과 키를 일관되게 저장합니다.
- UI에 표시되는 메타 store에는 큰 객체가 아니라 `version_token()`만 넣습니다.

## 환경과 좌표계

`SyntheticEnvironment`는 두 좌표계를 함께 관리합니다.

- Local 좌표: 분석 영역 좌상단을 `(0, 0)`으로 하는 미터 단위 좌표입니다. 최적화 알고리즘은 이 좌표계를 사용합니다.
- Geo 좌표: 지도 렌더링용 위도/경도입니다. `convert_to_geo()`와 환경의 변환 메서드가 Local/Geo 변환을 담당합니다.

중요한 데이터:

| 필드/메서드 | 의미 |
| --- | --- |
| `rows`, `cols` | 그리드 행/열 수 |
| `x_grid`, `y_grid` | Local 좌표 meshgrid |
| `lat_grid`, `lon_grid` | Geo 좌표 meshgrid |
| `traffic_map` | 현재 프레임의 마스킹 반영 트래픽 |
| `_raw_traffic_map` | 오브젝트 마스킹 전 원본 정적 트래픽 |
| `traffic_series` | 동적 트래픽 시계열 |
| `_raw_traffic_series` | 마스킹 전 동적 트래픽 시계열 |
| `dynamic_frame_index` | 현재 동적 프레임 |
| `obstacles` | Local 좌표계 오브젝트 폴리곤 |
| `station_candidate_points` | 설치 후보 지점 제약 |

정적 트래픽은 `generate_traffic_pattern_density()`에서 면적 수요 `Mbps/km2`를 셀 단위 수요로 변환합니다.

```text
cell_demand_mbps = area_demand_mbps_km2 * (resolution_m / 1000)^2
```

동적 트래픽은 `(time_steps, rows, cols)` 배열로 생성됩니다.

- `fixed_variation`: 같은 공간 분포의 강도만 시간에 따라 바뀝니다.
- `moving_hotspot`: 패턴의 중심 또는 위치 파라미터가 시간에 따라 이동합니다.
- `switching_locations`: 서로 다른 위치의 분포가 프레임별로 나타났다 사라집니다.

## 지도와 분석 렌더링

지도 view는 `dash-leaflet` 기반입니다.

- 트래픽/커버리지 셀: `build_traffic_geojson()`
- 오브젝트 레이어: `build_obstacle_polygons()`
- 기지국 마커/커버리지 원: `build_station_circles()`
- 현재 프레임 데이터프레임: `env_dataframe_for_current_frame()`
- 커버리지 상태 overlay: `compute_status_overlay()`

분석 결과 모음 view는 `render_analysis_view()`에서 생성합니다. 현재 섹션은 환경 요약, 현재 프레임 성능, 동적 트래픽 시나리오, 최적화 결과, 기지국별 분석, 운영 최적화 결과, Sweep 결과, 알고리즘 비교입니다.

새 분석 결과를 추가할 때는 다음 흐름을 따르세요.

1. `APP_STATE`에 결과를 저장할 키를 정합니다.
2. 결과 변경을 알릴 `dcc.Store` 또는 기존 meta store를 정합니다.
3. `render_analysis_view()`에 섹션 생성 로직을 추가합니다.
4. 결과가 없을 때 표시할 empty state를 함께 둡니다.
5. 필요한 경우 Playwright 회귀 테스트에 view 전환 및 텍스트 표시 검증을 추가합니다.

## 최적화 알고리즘 추가

새 알고리즘은 `optimizers.Optimizer` 계약을 구현해야 합니다.

필수 항목:

- `name`: UI에 표시되는 알고리즘 이름
- `hyperparams`: UI 자동 생성용 `HyperParam` 목록
- `optimize(problem, n_stations, **hp)`: `OptimizationResult` 반환

최소 구조:

```python
from optimizers.base import HyperParam, Optimizer, OptimizationResult, ProblemInput, compute_metrics


class MyOptimizer(Optimizer):
    name = "My Optimizer"
    hyperparams = [
        HyperParam("iterations", "int", default=100, min=10, max=1000, step=10),
    ]

    def optimize(self, problem: ProblemInput, n_stations: int, iterations: int = 100, callback=None):
        stations = ...  # shape: (n_stations, 2), Local meter coordinates
        score = ...
        return OptimizationResult(
            stations=stations,
            score=float(score),
            metrics=compute_metrics(stations, problem),
            history=None,
        )
```

등록:

1. 구현 파일을 `optimizers/` 아래에 추가합니다.
2. `optimizers/__init__.py`에서 import합니다.
3. `REGISTRY`에 클래스를 추가합니다.

주의사항:

- `stations`는 반드시 Local `(x, y)` 미터 좌표여야 합니다.
- 점수는 클수록 좋은 규약입니다.
- 커버리지/메트릭 일관성을 위해 `compute_metrics()`와 `sinr_coverage()`를 재사용하세요.
- 설치 후보 지점 제약이 있는 경우 `problem.feasible_station_points` 또는 `problem.station_candidate_points`를 존중해야 합니다.
- live progress를 지원하려면 `callback(it, total, best_stations_local, best_score)` 형태를 받아 호출하세요.
- Sweep 모드 1은 현재 `int`/`float` 하이퍼파라미터 중심으로 동작하며, 조합 수는 최대 500개로 제한됩니다.

## 트래픽 패턴 추가

정적 패턴은 `patterns.py`에서 관리합니다.

추가 절차:

1. `PATTERN_CHOICES`에 새 패턴 ID를 추가합니다.
2. `generate_pattern(rows, cols, pattern, rng, params)`에 분기를 추가합니다.
3. 반환값은 `(rows, cols)` 형태의 `numpy.ndarray`이고 기본적으로 `[0, 1]` 범위로 정규화되어야 합니다.
4. 패턴별 파라미터는 `params`에서 읽고, 누락 값은 안전한 기본값을 사용합니다.
5. UI 컨트롤이 필요하면 `sidebar_layout()`에 입력을 추가하고 환경 생성 콜백에서 params로 전달합니다.
6. 동적 모드에서 특별한 이동 규칙이 필요하면 `SyntheticEnvironment.generate_dynamic_traffic_pattern()`을 확장합니다.

테스트 권장:

- 새 패턴이 `PATTERN_CHOICES`에 노출되는지 UI 테스트 확인
- `generate_pattern()` 결과 shape, finite 값, 정규화 범위 검증
- 동적 모드에서 프레임별 변화가 의도대로 발생하는지 `tests/test_dynamic_traffic.py`에 추가

## 오브젝트 소스 추가

오브젝트는 최종적으로 Local 좌표계의 Shapely `Polygon` 목록으로 환경에 들어가야 합니다.

현재 소스:

- `합성`: `SyntheticEnvironment.generate_obstacles()`
- `OSM 지도 데이터`: `load_osm_polygons_with_cache()`
- `GeoJSON 업로드`: `geojson_to_polygons()`

새 소스를 추가하려면:

1. 좌측 UI의 `obstacle-source` 옵션을 추가합니다.
2. 필요한 파일 업로드/필터/옵션 UI를 추가합니다.
3. `load_map_obstacles()` 또는 `apply_obstacle_source()`에 소스 분기를 추가합니다.
4. Geo 좌표 폴리곤은 환경의 변환 메서드로 Local 좌표계에 맞춥니다.
5. `env.replace_obstacles()` 또는 `env.append_obstacles()` 후 `env.remask_traffic()`이 호출되는지 확인합니다.

OSM은 Overpass API를 사용하고 `.cache/osm_polygons`에 결과를 캐시합니다. 네트워크 실패와 캐시 만료를 고려해 UI 에러 메시지를 유지하세요.

## 운영 최적화 정책 추가

운영 최적화는 동적 트래픽 시나리오에서 기지국 ON/OFF 상태를 프레임별로 평가합니다. 기본 기준 정책은 `always-on`입니다.

관련 위치:

- 정책 목록: `OPERATION_POLICY_OPTIONS`
- 파라미터 정의: `OPERATION_PARAM_SPECS`
- 정책별 파라미터 노출: `OPERATION_POLICY_PARAM_NAMES`
- 파라미터 정규화: `normalize_operation_params()`
- 정책 실행: `evaluate_operation_optimization()`
- 지도 반영: `operation_active_mask_for_frame()`

새 정책 추가 절차:

1. `OPERATION_POLICY_OPTIONS`에 정책 value/label을 추가합니다.
2. 필요한 하이퍼파라미터를 `OPERATION_PARAM_SPECS`에 정의합니다.
3. 정책별 노출 파라미터를 `OPERATION_POLICY_PARAM_NAMES`에 연결합니다.
4. `normalize_operation_params()`에서 범위를 clamp합니다.
5. `evaluate_operation_optimization()`에 정책 분기를 추가합니다.
6. 각 frame 결과에 `active_mask`, `active_count`, `energy_cost`, `switching_cost`, `penalty_cost`, `step_opex`가 포함되게 합니다.
7. `operation_comparison_rows()`와 `operation_history_figure()`가 기존 필드로 동작하는지 확인합니다.

비용 계산은 `_operation_step_cost()`에서 수행합니다. 에너지는 송신 전력에서 유도한 active/sleep 전력과 부하 전력 계수를 사용하고, OPEX는 에너지 비용, 전환 비용, 미커버/과부하 패널티의 합으로 계산됩니다.

## 전파/커버리지 모델

커버리지는 단순 반경 기반이 아니라 SINR 기반입니다.

- 경로 손실: `PL(d) = path_loss_ref_db + 10 * n * log10(d / 1m)`
- SINR: 서빙 기지국 수신 전력 / 잡음과 간섭 합
- CoMP: `max_coord_stations`가 커질수록 상위 수신 전력 기지국 일부를 조율 대상으로 보고 간섭에서 제외합니다.
- 스펙트럼 효율: `shannon` 또는 `mcs`

`radius_from_tx()`는 송신 전력과 전파 파라미터로 시각화/후보 반경을 유도하지만, 실제 커버 여부와 점수는 `sinr_coverage()`가 결정합니다.

## 테스트 전략

변경 범위별 권장 테스트:

| 변경 | 권장 테스트 |
| --- | --- |
| 문법/임포트 | `python -m py_compile app.py tests/test_regression.py tests/test_dynamic_traffic.py` |
| 트래픽/운영 로직 | `pytest tests/test_dynamic_traffic.py -q` |
| UI 표시/상호작용 | `pytest tests/test_regression.py -q` |
| 알고리즘 추가 | 알고리즘 단위 테스트와 `test_algorithm_selectbox_has_all_optimizers` 확인 |
| 패턴 추가 | 패턴 단위 테스트와 `test_traffic_pattern_selectbox_has_all_patterns` 확인 |
| 전체 회귀 | `pytest -q` |

Playwright 테스트는 브라우저를 띄우므로 `python -m playwright install chromium`이 선행되어야 합니다.

## 개발 시 주의사항

- `APP_STATE`는 mutable Python 객체를 담습니다. 콜백에서 객체를 수정한 뒤에는 관련 meta store를 갱신해야 UI가 다시 렌더링됩니다.
- 환경, 기지국 모델, 전파 모델, 운영 정책 중 하나가 바뀌면 이전 결과가 같은 의미인지 검토하고 필요하면 stale state를 제거하세요.
- 내부 좌표는 Local `(x, y)` 미터이고 지도 렌더링은 `[lat, lon]`입니다. 알고리즘 입출력에서 좌표계를 섞지 마세요.
- 오브젝트 마스킹 전 트래픽과 마스킹 후 트래픽이 따로 존재합니다. 분석 목적에 따라 raw/현재 맵 중 어느 것을 써야 하는지 확인하세요.
- DQN 관련 기능은 `torch`에 의존합니다. optional path에서 실패할 때 사용자에게 이해 가능한 fallback 또는 에러가 필요합니다.
- `.cache/`, `.venv/`, `logs/`, UI 스크린샷은 일반적으로 커밋 대상이 아닙니다.
- `REGISTRY`와 `PATTERN_CHOICES`는 테스트가 직접 참조합니다. 항목을 추가하면 UI 테스트 기대값도 자동으로 확장됩니다.

## 변경 체크리스트

기능을 추가하거나 로직을 바꾼 뒤 아래를 확인하세요.

- 관련 상태 키를 초기화하거나 갱신했는가
- 지도 view와 분석 view가 같은 결과 기준을 보는가
- 동적 트래픽 프레임 변경 시 메트릭이 함께 갱신되는가
- 운영 정책이 있는 경우 `active_mask`가 지도/통계에 반영되는가
- Sweep/알고리즘 비교 결과 적용 후 기존 단일 알고리즘 결과가 사라지지 않는가
- 새 UI 컨트롤의 기본값이 서버 콜백의 fallback 값과 일치하는가
- 회귀 테스트 또는 단위 테스트가 변경 위험을 커버하는가
