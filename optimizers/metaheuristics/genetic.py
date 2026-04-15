"""Genetic Algorithm — bs_opt/kmj/algos/genetic.py에서 포팅.

변경점:
- 좌표: 정수 격자 → float 미터 (continuous clamp)
- 평가 방향: cost 최소화 → score 최대화 (내부에서 score 기반 비교)
- population의 각 개체 = (n_stations, 2) float 배열

bs_opt 원본의 토너먼트 선택, 단일점 교차, 포인트 변이 구조 유지.
"""
from __future__ import annotations

import numpy as np

from ..base import (
    HyperParam, Optimizer, OptimizationResult, ProblemInput, compute_metrics,
)
from ._shared import calculate_score, clip_stations, random_stations


class GeneticAlgorithmOptimizer(Optimizer):
    name = "Genetic Algorithm"
    hyperparams = [
        HyperParam("pop_size", "int", default=40, min=10, max=200, step=5,
                   label="pop_size (개체 수)"),
        HyperParam("n_generations", "int", default=200, min=10, max=2000, step=10,
                   label="n_generations (세대 수)"),
        HyperParam("crossover_rate", "float", default=0.9, min=0.0, max=1.0, step=0.01,
                   label="crossover_rate (교차 확률)"),
        HyperParam("mutation_rate", "float", default=0.2, min=0.0, max=1.0, step=0.01,
                   label="mutation_rate (변이 확률)"),
        HyperParam("tournament_size", "int", default=3, min=2, max=10,
                   label="tournament_size (토너먼트 크기)"),
        HyperParam("elitism", "int", default=2, min=0, max=20,
                   label="elitism (엘리트 개체 수)"),
        HyperParam("mutation_step", "float", default=50.0, min=1.0, max=500.0, step=1.0,
                   label="mutation_step (변이 이동 크기, m)"),
    ]

    def optimize(self, problem: ProblemInput, n_stations: int,
                 pop_size: int = 40, n_generations: int = 200,
                 crossover_rate: float = 0.9, mutation_rate: float = 0.2,
                 tournament_size: int = 3, elitism: int = 2,
                 mutation_step: float = 50.0) -> OptimizationResult:

        elitism = min(elitism, pop_size)

        # 초기 population
        population = [random_stations(n_stations, problem) for _ in range(pop_size)]
        scores = np.array([calculate_score(ind, problem) for ind in population])

        best_idx = int(np.argmax(scores))
        best = population[best_idx].copy()
        best_score = float(scores[best_idx])
        history = [{"iter": 0, "best_score": best_score,
                    "gen_best_score": float(scores.max())}]

        for gen in range(1, n_generations + 1):
            new_pop: list[np.ndarray] = []

            # Elitism: 상위 elitism개 그대로 복사
            if elitism > 0:
                elite_idxs = np.argsort(scores)[-elitism:]  # score↑이므로 상위 = 뒤쪽
                for i in elite_idxs:
                    new_pop.append(population[int(i)].copy())

            # 나머지: 토너먼트 선택 + 교차 + 변이
            while len(new_pop) < pop_size:
                p1 = _tournament_select(population, scores, tournament_size)
                p2 = _tournament_select(population, scores, tournament_size)
                c1, c2 = _crossover(p1, p2, crossover_rate, problem)
                c1 = _mutate(c1, mutation_rate, mutation_step, problem)
                c2 = _mutate(c2, mutation_rate, mutation_step, problem)
                new_pop.append(c1)
                if len(new_pop) < pop_size:
                    new_pop.append(c2)

            population = new_pop
            scores = np.array([calculate_score(ind, problem) for ind in population])

            gen_best_score = float(scores.max())
            if gen_best_score > best_score:
                best_score = gen_best_score
                best = population[int(np.argmax(scores))].copy()
            history.append({"iter": gen, "best_score": best_score,
                            "gen_best_score": gen_best_score})

        return OptimizationResult(
            stations=best,
            score=best_score,
            metrics=compute_metrics(best, problem),
            history=history,
        )


# ---------------------------------------------------------------------------
# GA 내부 헬퍼 — 이 파일에서만 사용
# ---------------------------------------------------------------------------
def _tournament_select(population: list[np.ndarray], scores: np.ndarray,
                       tournament_size: int) -> np.ndarray:
    idxs = np.random.randint(0, len(population), size=tournament_size)
    best_i = idxs[0]
    best_s = scores[best_i]
    for i in idxs[1:]:
        if scores[i] > best_s:   # score↑ 규약
            best_i = i
            best_s = scores[i]
    return population[int(best_i)].copy()


def _crossover(p1: np.ndarray, p2: np.ndarray, rate: float,
               problem: ProblemInput) -> tuple[np.ndarray, np.ndarray]:
    if np.random.rand() > rate or len(p1) <= 1:
        return p1.copy(), p2.copy()
    # (n_bs, 2) → flatten → single-point → reshape
    a = p1.reshape(-1)
    b = p2.reshape(-1)
    point = np.random.randint(1, len(a))
    c1 = np.concatenate([a[:point], b[point:]]).reshape(p1.shape)
    c2 = np.concatenate([b[:point], a[point:]]).reshape(p2.shape)
    clip_stations(c1, problem)
    clip_stations(c2, problem)
    return c1, c2


def _mutate(ind: np.ndarray, rate: float, step: float,
            problem: ProblemInput) -> np.ndarray:
    if np.random.rand() > rate:
        return ind
    out = ind.copy()
    idx = np.random.randint(0, len(out))
    out[idx] += np.random.normal(0, step, 2)
    clip_stations(out, problem)
    return out
