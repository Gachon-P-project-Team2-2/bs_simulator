"""Tabu Search: 최근 이동한 기지국을 일정 기간 금기."""
from __future__ import annotations

import logging
import time

import numpy as np

from ..base import (
    HyperParam, Optimizer, OptimizationResult, ProblemInput, compute_metrics,
)
from ._shared import calculate_score, clip_stations, random_stations

log = logging.getLogger(__name__)


class TabuSearchOptimizer(Optimizer):
    name = "Tabu Search"
    hyperparams = [
        HyperParam("iterations", "int", default=500, min=50, max=5000, step=50,
                   label="iterations (반복 수)"),
        HyperParam("step_size", "float", default=50.0, min=1.0, max=500.0, step=1.0,
                   label="step_size (이동 크기, m)"),
        HyperParam("tabu_tenure", "int", default=10, min=1, max=100,
                   label="tabu_tenure (금기 기간)"),
    ]

    def optimize(self, problem: ProblemInput, n_stations: int,
                 iterations: int = 500, step_size: float = 50.0,
                 tabu_tenure: int = 10, callback=None) -> OptimizationResult:
        t0 = time.perf_counter()

        # tabu_tenure가 n_stations보다 크면 전체 금기 포화 가능성 경고
        if tabu_tenure >= n_stations:
            log.warning(
                "TS: tabu_tenure=%d >= n_stations=%d — 금기 리스트가 모든 기지국을 "
                "동시에 차단할 수 있습니다. Aspiration Criteria를 자동 적용합니다.",
                tabu_tenure, n_stations,
            )

        log.info("TS start: n_stations=%d iterations=%d step=%.1f tenure=%d N=%d",
                 n_stations, iterations, step_size, tabu_tenure, len(problem.X))

        current = random_stations(n_stations, problem)
        current_score = calculate_score(current, problem)
        best = current.copy()
        best_score = current_score
        tabu_list: dict[int, int] = {}   # station_idx → until_iter
        snap_interval = max(1, iterations // 50)
        cb_interval = snap_interval * 3
        history = [{"iter": 0, "current_score": current_score, "best_score": best_score,
                    "stations": best.tolist()}]
        if callback is not None:
            callback(0, iterations, best.copy(), best_score)

        log_interval = max(1, iterations // 10)
        aspiration_count = 0

        for it in range(1, iterations + 1):
            regular: list[tuple[float, np.ndarray, int]] = []
            tabu_aspirants: list[tuple[float, np.ndarray, int]] = []

            for _ in range(20):
                idx = np.random.randint(0, n_stations)
                cand = current.copy()
                cand[idx] += np.random.normal(0, step_size, 2)
                clip_stations(cand, problem)
                score = calculate_score(cand, problem)
                if idx not in tabu_list or tabu_list[idx] <= it:
                    regular.append((score, cand, idx))
                else:
                    # 금기 이동이지만 현재 최선을 능가하면 aspiration 후보로 보관
                    if score > best_score:
                        tabu_aspirants.append((score, cand, idx))

            if regular:
                regular.sort(key=lambda x: x[0], reverse=True)
                chosen_score, chosen_stations, moved_idx = regular[0]
            elif tabu_aspirants:
                # Aspiration Criteria: 금기 이동이라도 전체 최선 갱신 시 허용
                tabu_aspirants.sort(key=lambda x: x[0], reverse=True)
                chosen_score, chosen_stations, moved_idx = tabu_aspirants[0]
                aspiration_count += 1
                log.debug("TS iter=%d: aspiration 적용 (tabu idx=%d score=%.4f > best=%.4f)",
                          it, moved_idx, chosen_score, best_score)
            else:
                # 정규 후보도, aspiration 조건 충족 금기 후보도 없음 → skip
                log.debug("TS iter=%d: 모든 후보 금기 상태 — iteration skip", it)
                entry: dict = {"iter": it, "current_score": current_score,
                               "best_score": best_score}
                if it % snap_interval == 0 or it == iterations:
                    entry["stations"] = best.tolist()
                if callback is not None and (it % cb_interval == 0 or it == iterations):
                    callback(it, iterations, best.copy(), best_score)
                history.append(entry)
                continue

            current, current_score = chosen_stations, chosen_score
            tabu_list[moved_idx] = it + tabu_tenure
            if current_score > best_score:
                best_score = current_score
                best = current.copy()

            entry = {"iter": it, "current_score": current_score, "best_score": best_score}
            if it % snap_interval == 0 or it == iterations:
                entry["stations"] = best.tolist()
            if callback is not None and (it % cb_interval == 0 or it == iterations):
                callback(it, iterations, best.copy(), best_score)
            history.append(entry)

            if it % log_interval == 0 or it == iterations:
                active_tabu = sum(1 for exp in tabu_list.values() if exp > it)
                log.debug("TS iter=%d/%d best=%.4f active_tabu=%d/%d aspiration_total=%d",
                          it, iterations, best_score, active_tabu, n_stations, aspiration_count)

        elapsed = time.perf_counter() - t0
        log.info("TS done: best_score=%.4f aspiration_used=%d elapsed=%.3fs",
                 best_score, aspiration_count, elapsed)
        if best_score == 0.0:
            log.warning("TS score=0: 커버리지가 전혀 없습니다.")

        return OptimizationResult(
            stations=best,
            score=best_score,
            metrics=compute_metrics(best, problem),
            history=history,
        )
