from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from . import FIM as infmat


def selection_vector(num_candidates: int, indices: Sequence[int]) -> np.ndarray:
    weights = np.zeros(num_candidates, dtype=float)
    weights[np.asarray(indices, dtype=int)] = 1.0
    return weights


def evaluate_selection(
    problem: infmat.CalibrationProblem,
    indices: Sequence[int],
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
) -> Dict[str, float]:
    if prior is None:
        prior = infmat.build_prior_blocks(problem)
    info_blocks = infmat.construct_candidate_inf_blocks(problem)
    weights = selection_vector(problem.num_candidates, indices)
    h_cal = infmat.compute_calibration_schur_compact(problem, weights, info_blocks, prior=prior)
    eigvals = np.linalg.eigvalsh(h_cal)
    reg = 1e-12 * np.eye(h_cal.shape[0], dtype=float)
    cov = np.linalg.pinv(h_cal + reg)

    visible_points = 0
    for idx in indices:
        visible_points += int(info_blocks.visible_counts[int(idx)])

    return {
        "min_eig": float(eigvals[0]),
        "max_eig": float(eigvals[-1]),
        "logdet": float(np.linalg.slogdet(h_cal + reg)[1]),
        "trace_cov": float(np.trace(cov)),
        "cond": float(eigvals[-1] / max(eigvals[0], 1e-12)),
        "visible_points": float(visible_points),
    }


def candidate_min_eig_scores(
    problem: infmat.CalibrationProblem,
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
) -> np.ndarray:
    if prior is None:
        prior = infmat.build_prior_blocks(problem)
    info_blocks = infmat.construct_candidate_inf_blocks(problem)
    scores = np.zeros(problem.num_candidates, dtype=float)
    for idx in range(problem.num_candidates):
        weights = np.zeros(problem.num_candidates, dtype=float)
        weights[idx] = 1.0
        scores[idx] = infmat.compute_min_eig_score(problem, weights, info_blocks, prior=prior)
    return scores


def candidate_eigenvalue_spectra(
    problem: infmat.CalibrationProblem,
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
) -> np.ndarray:
    if prior is None:
        prior = infmat.build_prior_blocks(problem)
    info_blocks = infmat.construct_candidate_inf_blocks(problem)
    spectra = np.zeros((problem.num_candidates, problem.intrinsics_dim), dtype=float)
    for idx in range(problem.num_candidates):
        weights = np.zeros(problem.num_candidates, dtype=float)
        weights[idx] = 1.0
        h_cal = infmat.compute_calibration_schur_compact(problem, weights, info_blocks, prior=prior)
        spectra[idx] = np.linalg.eigvalsh(h_cal)
    return spectra


def before_after_calibration_summary(
    problem: infmat.CalibrationProblem,
    selected_indices: Sequence[int],
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
) -> Dict[str, object]:
    if prior is None:
        prior = infmat.build_prior_blocks(problem)
    info_blocks = infmat.construct_candidate_inf_blocks(problem)
    before_selection = np.zeros(problem.num_candidates, dtype=float)
    after_selection = selection_vector(problem.num_candidates, selected_indices)

    before_h_cal = infmat.compute_calibration_schur_compact(problem, before_selection, info_blocks, prior=prior)
    after_h_cal = infmat.compute_calibration_schur_compact(problem, after_selection, info_blocks, prior=prior)

    reg = 1e-12 * np.eye(problem.intrinsics_dim, dtype=float)
    before_eigvals = np.linalg.eigvalsh(before_h_cal)
    after_eigvals = np.linalg.eigvalsh(after_h_cal)
    before_cov = np.linalg.pinv(before_h_cal + reg)
    after_cov = np.linalg.pinv(after_h_cal + reg)
    before_std = np.sqrt(np.maximum(np.diag(before_cov), 0.0))
    after_std = np.sqrt(np.maximum(np.diag(after_cov), 0.0))

    return {
        "parameter_labels": intrinsic_parameter_labels(problem.intrinsics_dim),
        "before": {
            "h_cal": before_h_cal.tolist(),
            "eigvals": before_eigvals.tolist(),
            "min_eig": float(before_eigvals[0]),
            "covariance_diag": np.diag(before_cov).tolist(),
            "std_dev": before_std.tolist(),
        },
        "after": {
            "h_cal": after_h_cal.tolist(),
            "eigvals": after_eigvals.tolist(),
            "min_eig": float(after_eigvals[0]),
            "covariance_diag": np.diag(after_cov).tolist(),
            "std_dev": after_std.tolist(),
        },
    }


def intrinsic_parameter_labels(intrinsics_dim: int) -> List[str]:
    default = ["fx", "fy", "cx", "cy"]
    if intrinsics_dim <= len(default):
        return default[:intrinsics_dim]
    extra = [f"theta_{idx}" for idx in range(len(default), intrinsics_dim)]
    return default + extra


def random_baseline_report(
    problem: infmat.CalibrationProblem,
    select_k: int,
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
    num_trials: int = 50,
    seed: int = 0,
) -> Dict[str, object]:
    if prior is None:
        prior = infmat.build_prior_blocks(problem)
    rng = np.random.default_rng(seed)
    trials: List[Dict[str, float]] = []
    selections: List[List[int]] = []
    for _ in range(num_trials):
        indices = sorted(rng.choice(problem.num_candidates, size=select_k, replace=False).tolist())
        selections.append(indices)
        trials.append(evaluate_selection(problem, indices, prior=prior))

    min_eigs = np.array([trial["min_eig"] for trial in trials], dtype=float)
    best_idx = int(np.argmax(min_eigs))
    avg_report = {
        key: float(np.mean([trial[key] for trial in trials]))
        for key in trials[0].keys()
    }
    best_report = trials[best_idx]
    return {
        "average": avg_report,
        "best": best_report,
        "best_indices": selections[best_idx],
        "trials": trials,
    }


def compare_selected_vs_random(
    problem: infmat.CalibrationProblem,
    selected_indices: Sequence[int],
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
    num_random_trials: int = 50,
    seed: int = 0,
) -> Dict[str, object]:
    selected_report = evaluate_selection(problem, selected_indices, prior=prior)
    random_report = random_baseline_report(
        problem,
        select_k=len(selected_indices),
        prior=prior,
        num_trials=num_random_trials,
        seed=seed,
    )
    return {
        "selected": selected_report,
        "random_average": random_report["average"],
        "random_best": random_report["best"],
        "random_best_indices": random_report["best_indices"],
        "random_trials": random_report["trials"],
    }
