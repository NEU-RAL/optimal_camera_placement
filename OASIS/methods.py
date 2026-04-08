from __future__ import annotations

import time
from enum import Enum
from typing import List, Sequence, Tuple

import numpy as np

from . import FIM as infmat


class Metric(Enum):
    MIN_EIG = 1


def greedy_selection(
    problem: infmat.CalibrationProblem,
    select_k: int,
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
    metric: Metric = Metric.MIN_EIG,
):
    if metric is not Metric.MIN_EIG:
        raise ValueError("This calibration pipeline keeps only the minimum-eigenvalue objective.")
    if prior is None:
        prior = infmat.build_prior_blocks(problem)

    info_blocks = infmat.construct_candidate_inf_blocks(problem)
    available = np.ones(problem.num_candidates, dtype=bool)
    selection = np.zeros(problem.num_candidates, dtype=float)
    selected_indices: List[int] = []
    best_score = float("-inf")

    for _ in range(select_k):
        candidate_best_score = float("-inf")
        candidate_best_idx = -1
        for idx in range(problem.num_candidates):
            if not available[idx]:
                continue
            trial_selection = selection.copy()
            trial_selection[idx] = 1.0
            score = infmat.compute_min_eig_score(problem, trial_selection, info_blocks, prior=prior)
            if score > candidate_best_score:
                candidate_best_score = score
                candidate_best_idx = idx

        if candidate_best_idx < 0:
            break

        selection[candidate_best_idx] = 1.0
        available[candidate_best_idx] = False
        selected_indices.append(candidate_best_idx)
        best_score = candidate_best_score

    selected_poses = [
        {
            "rotation_wc": problem.candidate_rotations[idx],
            "translation_wc": problem.candidate_translations[idx],
            "visible_points": int(info_blocks.visible_counts[idx]),
        }
        for idx in selected_indices
    ]
    return selected_poses, selected_indices, best_score, selection


def greedy_selection_exp(
    problems: Sequence[infmat.CalibrationProblem],
    select_k: int,
    priors: Sequence[np.ndarray | infmat.CalibrationPrior] | None = None,
    metric: Metric = Metric.MIN_EIG,
):
    if metric is not Metric.MIN_EIG:
        raise ValueError("This calibration pipeline keeps only the minimum-eigenvalue objective.")
    if not problems:
        raise ValueError("At least one calibration problem is required.")

    if priors is None:
        priors = [infmat.build_prior_blocks(problem) for problem in problems]

    info_blocks_per_problem = []
    for problem in problems:
        info_blocks_per_problem.append(infmat.construct_candidate_inf_blocks(problem))

    num_candidates = problems[0].num_candidates
    available = np.ones(num_candidates, dtype=bool)
    selection = np.zeros(num_candidates, dtype=float)
    selected_indices: List[int] = []
    best_score = float("-inf")

    for _ in range(select_k):
        candidate_best_score = float("-inf")
        candidate_best_idx = -1
        for idx in range(num_candidates):
            if not available[idx]:
                continue
            trial_selection = selection.copy()
            trial_selection[idx] = 1.0
            total_score = 0.0
            for problem, prior, info_blocks in zip(problems, priors, info_blocks_per_problem):
                total_score += infmat.compute_min_eig_score(problem, trial_selection, info_blocks, prior=prior)
            if total_score > candidate_best_score:
                candidate_best_score = total_score
                candidate_best_idx = idx

        if candidate_best_idx < 0:
            break

        selection[candidate_best_idx] = 1.0
        available[candidate_best_idx] = False
        selected_indices.append(candidate_best_idx)
        best_score = candidate_best_score

    selected_poses = [
        {
            "rotation_wc": problems[0].candidate_rotations[idx],
            "translation_wc": problems[0].candidate_translations[idx],
            "visible_points": int(info_blocks_per_problem[0].visible_counts[idx]),
        }
        for idx in selected_indices
    ]
    return selected_poses, selected_indices, best_score, selection


def run_single_experiment_exp(
    problems: Sequence[infmat.CalibrationProblem],
    select_k: int,
    priors: Sequence[np.ndarray] | None = None,
):
    start = time.time()
    selected_poses, selected_indices, best_score, selection = greedy_selection_exp(
        problems,
        select_k,
        priors=priors,
        metric=Metric.MIN_EIG,
    )
    elapsed = time.time() - start
    return best_score, selected_poses, selected_indices, elapsed, selection
