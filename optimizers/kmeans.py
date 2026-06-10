"""K-Means 기반 최적화 — sklearn 사용. 메타휴리스틱과 다른 범주이므로 별도 파일."""
from __future__ import annotations

import logging
import time

from sklearn.cluster import KMeans

from .base import (
    HyperParam, Optimizer, OptimizationResult, ProblemInput, compute_metrics,
)
from .metaheuristics._shared import calculate_score
from .metaheuristics._shared import snap_stations_to_candidates

log = logging.getLogger(__name__)


class KMeansOptimizer(Optimizer):
    name = "K-Means"
    hyperparams = [
        HyperParam("n_init", "int", default=10, min=1, max=50,
                   label="n_init (초기화 횟수)"),
        HyperParam("random_state", "int", default=42, min=-1, max=99999,
                   label="random_state (시드, -1=랜덤)"),
    ]

    def optimize(self, problem: ProblemInput, n_stations: int,
                 n_init: int = 10, random_state: int = 42,
                 callback=None) -> OptimizationResult:
        t0 = time.perf_counter()
        log.info("K-Means start: n_stations=%d n_init=%d random_state=%s N=%d",
                 n_stations, n_init, random_state, len(problem.X))
        rs = None if random_state == -1 else random_state
        km = KMeans(n_clusters=n_stations, n_init=n_init, random_state=rs)
        km.fit(problem.X, sample_weight=problem.weights)
        stations = snap_stations_to_candidates(km.cluster_centers_, problem)
        score = calculate_score(stations, problem)
        metrics = compute_metrics(stations, problem)
        log.info("K-Means done: score=%.4f elapsed=%.3fs", score, time.perf_counter() - t0)
        if score == 0.0:
            log.warning("K-Means score=0: 커버리지가 전혀 없습니다. 반경/용량 설정을 확인하세요.")
        return OptimizationResult(
            stations=stations,
            score=score,
            metrics=metrics,
            history=None,
        )
