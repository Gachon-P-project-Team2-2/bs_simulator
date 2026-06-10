"""Simulated Annealing: 악화 수용을 확률적으로 허용, 온도 감소."""
from __future__ import annotations

import logging
import time

import numpy as np

from ..base import (
    HyperParam, Optimizer, OptimizationResult, ProblemInput, compute_metrics,
)
from ._shared import calculate_score, clip_stations, random_stations

log = logging.getLogger(__name__)

# 탐색이 사실상 중단됐다고 판단하는 온도 임계값
_TEMP_WARN_THRESHOLD = 0.1


class SimulatedAnnealingOptimizer(Optimizer):
    name = "Simulated Annealing"
    hyperparams = [
        HyperParam("iterations", "int", default=1000, min=100, max=10000, step=100,
                   label="iterations (반복 수)"),
        HyperParam("initial_temp", "float", default=100.0, min=1.0, max=1000.0, step=1.0,
                   label="initial_temp (초기 온도)"),
        HyperParam("cooling_rate", "float", default=0.995, min=0.80, max=0.9999, step=0.001,
                   label="cooling_rate (냉각률)"),
        HyperParam("step_size", "float", default=50.0, min=1.0, max=500.0, step=1.0,
                   label="step_size (이동 크기, m)"),
    ]

    def optimize(self, problem: ProblemInput, n_stations: int,
                 iterations: int = 1000, initial_temp: float = 100.0,
                 cooling_rate: float = 0.995, step_size: float = 50.0,
                 callback=None) -> OptimizationResult:
        t0 = time.perf_counter()

        # 최종 온도 사전 계산 및 경고
        final_temp = initial_temp * (cooling_rate ** iterations)
        log.info(
            "SA start: n_stations=%d iterations=%d T0=%.1f cooling=%.4f step=%.1f "
            "→ final_temp=%.4f N=%d",
            n_stations, iterations, initial_temp, cooling_rate, step_size, final_temp, len(problem.X),
        )
        if final_temp < _TEMP_WARN_THRESHOLD:
            greedy_from = int(
                iterations - (iterations * (1 - cooling_rate)) * 100
                if cooling_rate < 1.0
                else 0
            )
            log.warning(
                "SA: 최종 온도 %.4f < %.1f — 후반부 탐색이 사실상 greedy로 수렴합니다. "
                "cooling_rate를 높이거나(예: 0.999) iterations를 줄이세요.",
                final_temp, _TEMP_WARN_THRESHOLD,
            )

        current = random_stations(n_stations, problem)
        current_score = calculate_score(current, problem)
        best = current.copy()
        best_score = current_score
        temp = initial_temp
        snap_interval = max(1, iterations // 50)
        cb_interval = snap_interval * 3
        history = [{"iter": 0, "current_score": current_score, "best_score": best_score,
                    "temp": temp, "stations": best.tolist()}]
        if callback is not None:
            callback(0, iterations, best.copy(), best_score)

        log_interval = max(1, iterations // 10)
        temp_warned = False

        for it in range(1, iterations + 1):
            next_stations = current.copy()
            idx = np.random.randint(0, n_stations)
            next_stations[idx] += np.random.normal(0, step_size, 2)
            clip_stations(next_stations, problem)
            next_score = calculate_score(next_stations, problem)
            delta = current_score - next_score   # 양수 = 악화
            if delta < 0 or np.random.rand() < np.exp(-delta / (temp + 1e-9)):
                current, current_score = next_stations, next_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
            temp *= cooling_rate

            entry: dict = {"iter": it, "current_score": current_score,
                           "best_score": best_score, "temp": temp}
            if it % snap_interval == 0 or it == iterations:
                entry["stations"] = best.tolist()
            if callback is not None and (it % cb_interval == 0 or it == iterations):
                callback(it, iterations, best.copy(), best_score)
            history.append(entry)

            if it % log_interval == 0 or it == iterations:
                log.debug("SA iter=%d/%d best=%.4f temp=%.4f", it, iterations, best_score, temp)

            if not temp_warned and temp < _TEMP_WARN_THRESHOLD:
                log.warning("SA iter=%d: 온도 %.4f < %.1f 도달 — 이후 탐색이 greedy로 전환됩니다.",
                            it, temp, _TEMP_WARN_THRESHOLD)
                temp_warned = True

        elapsed = time.perf_counter() - t0
        log.info("SA done: best_score=%.4f elapsed=%.3fs", best_score, elapsed)
        if best_score == 0.0:
            log.warning("SA score=0: 커버리지가 전혀 없습니다.")

        return OptimizationResult(
            stations=best,
            score=best_score,
            metrics=compute_metrics(best, problem),
            history=history,
        )
