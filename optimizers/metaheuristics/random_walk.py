"""Random Walk: 랜덤 섭동 후 개선 시 수용, 아니면 버림."""
from __future__ import annotations

import logging
import time

from ..base import (
    HyperParam, Optimizer, OptimizationResult, ProblemInput, compute_metrics,
)
from ._shared import calculate_score, clip_stations, perturb, random_stations

log = logging.getLogger(__name__)


class RandomWalkOptimizer(Optimizer):
    name = "Random Walk"
    hyperparams = [
        HyperParam("iterations", "int", default=1000, min=100, max=10000, step=100,
                   label="iterations (반복 수)"),
        HyperParam("step_size", "float", default=50.0, min=1.0, max=500.0, step=1.0,
                   label="step_size (이동 크기, m)"),
    ]

    def optimize(self, problem: ProblemInput, n_stations: int,
                 iterations: int = 1000, step_size: float = 50.0,
                 callback=None) -> OptimizationResult:
        t0 = time.perf_counter()
        log.info("RandomWalk start: n_stations=%d iterations=%d step_size=%.1f N=%d",
                 n_stations, iterations, step_size, len(problem.X))

        current = random_stations(n_stations, problem)
        current_score = calculate_score(current, problem)
        best = current.copy()
        best_score = current_score
        snap_interval = max(1, iterations // 50)
        cb_interval = snap_interval * 3
        history = [{"iter": 0, "current_score": current_score, "best_score": best_score,
                    "stations": best.tolist()}]
        if callback is not None:
            callback(0, iterations, best.copy(), best_score)

        log_interval = max(1, iterations // 10)
        no_improve_streak = 0

        for it in range(1, iterations + 1):
            next_stations = clip_stations(perturb(current, step_size), problem)
            next_score = calculate_score(next_stations, problem)
            if next_score > current_score:
                current, current_score = next_stations, next_score
                no_improve_streak = 0
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
            else:
                no_improve_streak += 1

            entry: dict = {"iter": it, "current_score": current_score, "best_score": best_score}
            if it % snap_interval == 0 or it == iterations:
                entry["stations"] = best.tolist()
            if callback is not None and (it % cb_interval == 0 or it == iterations):
                callback(it, iterations, best.copy(), best_score)
            history.append(entry)

            if it % log_interval == 0 or it == iterations:
                log.debug("RandomWalk iter=%d/%d best=%.4f no_improve_streak=%d",
                          it, iterations, best_score, no_improve_streak)

        elapsed = time.perf_counter() - t0
        log.info("RandomWalk done: best_score=%.4f elapsed=%.3fs", best_score, elapsed)
        if best_score == 0.0:
            log.warning("RandomWalk score=0: 커버리지가 전혀 없습니다.")

        return OptimizationResult(
            stations=best,
            score=best_score,
            metrics=compute_metrics(best, problem),
            history=history,
        )
