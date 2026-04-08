from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy import linalg as la


@dataclass
class CalibrationProblem:
    """Synthetic camera calibration problem used for pose selection.

    Attributes:
        target_points: Known calibration target points in the world frame, shape (M, 3).
        candidate_rotations: Candidate camera rotations R_wc, shape (N, 3, 3).
        candidate_translations: Candidate camera translations t_wc, shape (N, 3).
        measurements: Noisy image observations for each candidate pose, shape (N, M, 2).
            Invalid observations are stored as NaN.
        intrinsics_gt: Ground-truth intrinsic parameter vector [fx, fy, cx, cy].
        intrinsics_init: Initial intrinsic estimate used as the linearization point.
        image_size: Image size as (width, height).
        pixel_noise_sigma: Standard deviation of image noise in pixels.
    """

    target_points: np.ndarray
    candidate_rotations: np.ndarray
    candidate_translations: np.ndarray
    measurements: np.ndarray
    intrinsics_gt: np.ndarray
    intrinsics_init: np.ndarray
    image_size: Tuple[int, int]
    pixel_noise_sigma: float = 1.0
    candidate_board_rotations: Optional[np.ndarray] = None
    candidate_board_translations: Optional[np.ndarray] = None
    camera_is_fixed: bool = False

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_rotations.shape[0])

    @property
    def intrinsics_dim(self) -> int:
        return int(self.intrinsics_init.shape[0])

    @property
    def pose_dim(self) -> int:
        return 6

    @property
    def total_dim(self) -> int:
        return self.intrinsics_dim + self.pose_dim * self.num_candidates


@dataclass
class CandidateInfoBlocks:
    h_tt: np.ndarray
    h_tp: np.ndarray
    h_pp: np.ndarray
    visible_counts: np.ndarray


@dataclass
class CalibrationPrior:
    h_tt: np.ndarray
    h_pp: np.ndarray


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _exp_so3(w: np.ndarray) -> np.ndarray:
    theta = la.norm(w)
    if theta < 1e-12:
        return np.eye(3) + _skew(w)
    W = _skew(w)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * W + b * (W @ W)


def project_points(
    points_w: np.ndarray,
    rotation_wc: np.ndarray,
    translation_wc: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Project known target points into the image using a pinhole camera model."""
    fx, fy, cx, cy = intrinsics
    rotation_cw = rotation_wc.T
    points_c = (rotation_cw @ (points_w - translation_wc).T).T
    z = points_c[:, 2]
    proj = np.full((points_w.shape[0], 2), np.nan, dtype=float)
    valid = z > 1e-9
    if not np.any(valid):
        return proj

    x_norm = points_c[valid, 0] / z[valid]
    y_norm = points_c[valid, 1] / z[valid]
    proj[valid, 0] = fx * x_norm + cx
    proj[valid, 1] = fy * y_norm + cy
    return proj


def visible_measurement_mask(measurements: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    finite = np.isfinite(measurements).all(axis=1)
    in_bounds = (
        (measurements[:, 0] >= 0.0)
        & (measurements[:, 0] < width)
        & (measurements[:, 1] >= 0.0)
        & (measurements[:, 1] < height)
    )
    return finite & in_bounds


def perturb_pose(rotation_wc: np.ndarray, translation_wc: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rot_delta = _exp_so3(delta[:3])
    new_rotation = rotation_wc @ rot_delta
    new_translation = translation_wc + delta[3:]
    return new_rotation, new_translation


def numerical_jacobian_intrinsics(
    points_w: np.ndarray,
    rotation_wc: np.ndarray,
    translation_wc: np.ndarray,
    intrinsics: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    base = project_points(points_w, rotation_wc, translation_wc, intrinsics)[valid_mask].reshape(-1)
    jac = np.zeros((base.size, intrinsics.size), dtype=float)
    for idx in range(intrinsics.size):
        step = np.zeros_like(intrinsics)
        step[idx] = eps
        plus = project_points(points_w, rotation_wc, translation_wc, intrinsics + step)[valid_mask].reshape(-1)
        minus = project_points(points_w, rotation_wc, translation_wc, intrinsics - step)[valid_mask].reshape(-1)
        jac[:, idx] = (plus - minus) / (2.0 * eps)
    return jac


def numerical_jacobian_pose(
    points_w: np.ndarray,
    rotation_wc: np.ndarray,
    translation_wc: np.ndarray,
    intrinsics: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    base = project_points(points_w, rotation_wc, translation_wc, intrinsics)[valid_mask].reshape(-1)
    jac = np.zeros((base.size, 6), dtype=float)
    for idx in range(6):
        step = np.zeros(6, dtype=float)
        step[idx] = eps
        rot_plus, trans_plus = perturb_pose(rotation_wc, translation_wc, step)
        rot_minus, trans_minus = perturb_pose(rotation_wc, translation_wc, -step)
        plus = project_points(points_w, rot_plus, trans_plus, intrinsics)[valid_mask].reshape(-1)
        minus = project_points(points_w, rot_minus, trans_minus, intrinsics)[valid_mask].reshape(-1)
        jac[:, idx] = (plus - minus) / (2.0 * eps)
    return jac


def candidate_information_matrix(problem: CalibrationProblem, candidate_index: int) -> np.ndarray:
    """Build the Fisher information contribution for one candidate pose.

    The state is ordered as [intrinsics, pose_0, ..., pose_{N-1}]. Only the selected
    candidate's own 6-DoF nuisance pose block is populated.
    """
    measurements = problem.measurements[candidate_index]
    valid_mask = visible_measurement_mask(measurements, problem.image_size)
    if np.count_nonzero(valid_mask) < 4:
        return np.zeros((problem.total_dim, problem.total_dim), dtype=float)

    rotation_wc = problem.candidate_rotations[candidate_index]
    translation_wc = problem.candidate_translations[candidate_index]
    jac_intr = numerical_jacobian_intrinsics(
        problem.target_points,
        rotation_wc,
        translation_wc,
        problem.intrinsics_init,
        valid_mask,
    )
    jac_pose = numerical_jacobian_pose(
        problem.target_points,
        rotation_wc,
        translation_wc,
        problem.intrinsics_init,
        valid_mask,
    )

    jac = np.zeros((jac_intr.shape[0], problem.total_dim), dtype=float)
    jac[:, : problem.intrinsics_dim] = jac_intr
    pose_start = problem.intrinsics_dim + candidate_index * problem.pose_dim
    pose_end = pose_start + problem.pose_dim
    jac[:, pose_start:pose_end] = jac_pose

    weight = np.eye(jac.shape[0], dtype=float) / (problem.pixel_noise_sigma ** 2)
    return jac.T @ weight @ jac


def candidate_information_blocks(problem: CalibrationProblem, candidate_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build the compact intrinsics/pose blocks for one candidate pose."""
    measurements = problem.measurements[candidate_index]
    valid_mask = visible_measurement_mask(measurements, problem.image_size)
    visible = int(np.count_nonzero(valid_mask))
    if visible < 4:
        zeros_tt = np.zeros((problem.intrinsics_dim, problem.intrinsics_dim), dtype=float)
        zeros_tp = np.zeros((problem.intrinsics_dim, problem.pose_dim), dtype=float)
        zeros_pp = np.zeros((problem.pose_dim, problem.pose_dim), dtype=float)
        return zeros_tt, zeros_tp, zeros_pp, visible

    rotation_wc = problem.candidate_rotations[candidate_index]
    translation_wc = problem.candidate_translations[candidate_index]
    jac_intr = numerical_jacobian_intrinsics(
        problem.target_points,
        rotation_wc,
        translation_wc,
        problem.intrinsics_init,
        valid_mask,
    )
    jac_pose = numerical_jacobian_pose(
        problem.target_points,
        rotation_wc,
        translation_wc,
        problem.intrinsics_init,
        valid_mask,
    )

    weight = 1.0 / (problem.pixel_noise_sigma ** 2)
    h_tt = weight * (jac_intr.T @ jac_intr)
    h_tp = weight * (jac_intr.T @ jac_pose)
    h_pp = weight * (jac_pose.T @ jac_pose)
    return h_tt, h_tp, h_pp, visible


def construct_candidate_inf_blocks(problem: CalibrationProblem) -> CandidateInfoBlocks:
    h_tt = np.zeros((problem.num_candidates, problem.intrinsics_dim, problem.intrinsics_dim), dtype=float)
    h_tp = np.zeros((problem.num_candidates, problem.intrinsics_dim, problem.pose_dim), dtype=float)
    h_pp = np.zeros((problem.num_candidates, problem.pose_dim, problem.pose_dim), dtype=float)
    visible_counts = np.zeros(problem.num_candidates, dtype=int)
    for idx in range(problem.num_candidates):
        h_tt[idx], h_tp[idx], h_pp[idx], visible_counts[idx] = candidate_information_blocks(problem, idx)
    return CandidateInfoBlocks(h_tt=h_tt, h_tp=h_tp, h_pp=h_pp, visible_counts=visible_counts)


def construct_candidate_inf_mats(problem: CalibrationProblem) -> Tuple[np.ndarray, List[int]]:
    mats = np.zeros((problem.num_candidates, problem.total_dim, problem.total_dim), dtype=float)
    debug_visible_counts: List[int] = []
    for idx in range(problem.num_candidates):
        mats[idx] = candidate_information_matrix(problem, idx)
        visible = int(np.count_nonzero(visible_measurement_mask(problem.measurements[idx], problem.image_size)))
        debug_visible_counts.append(visible)
    return mats, debug_visible_counts


def build_prior_information(
    problem: CalibrationProblem,
    intrinsics_prior_sigma: Sequence[float] = (30.0, 30.0, 20.0, 20.0),
    pose_prior_sigma: Sequence[float] = (2.0, 2.0, 2.0, 0.5, 0.5, 0.5),
) -> np.ndarray:
    """Build a weak prior used to stabilize the Schur complement numerically."""
    prior = np.zeros((problem.total_dim, problem.total_dim), dtype=float)
    intr_diag = 1.0 / np.square(np.asarray(intrinsics_prior_sigma, dtype=float))
    prior[: problem.intrinsics_dim, : problem.intrinsics_dim] = np.diag(intr_diag)

    pose_diag = 1.0 / np.square(np.asarray(pose_prior_sigma, dtype=float))
    for idx in range(problem.num_candidates):
        start = problem.intrinsics_dim + idx * problem.pose_dim
        end = start + problem.pose_dim
        prior[start:end, start:end] = np.diag(pose_diag)
    return prior


def build_prior_blocks(
    problem: CalibrationProblem,
    intrinsics_prior_sigma: Sequence[float] = (30.0, 30.0, 20.0, 20.0),
    pose_prior_sigma: Sequence[float] = (2.0, 2.0, 2.0, 0.5, 0.5, 0.5),
) -> CalibrationPrior:
    intr_diag = 1.0 / np.square(np.asarray(intrinsics_prior_sigma, dtype=float))
    pose_diag = 1.0 / np.square(np.asarray(pose_prior_sigma, dtype=float))
    return CalibrationPrior(h_tt=np.diag(intr_diag), h_pp=np.diag(pose_diag))


def _coerce_prior_blocks(problem: CalibrationProblem, prior: Optional[np.ndarray | CalibrationPrior]) -> CalibrationPrior:
    if prior is None:
        return build_prior_blocks(problem)
    if isinstance(prior, CalibrationPrior):
        return prior

    h_tt = prior[: problem.intrinsics_dim, : problem.intrinsics_dim]
    pose_start = problem.intrinsics_dim
    pose_end = pose_start + problem.pose_dim
    if prior.shape[0] >= pose_end:
        h_pp = prior[pose_start:pose_end, pose_start:pose_end]
    else:
        h_pp = np.zeros((problem.pose_dim, problem.pose_dim), dtype=float)
    return CalibrationPrior(h_tt=np.array(h_tt, copy=True), h_pp=np.array(h_pp, copy=True))


def compute_calibration_schur_compact(
    problem: CalibrationProblem,
    selection: Sequence[int] | np.ndarray,
    info_blocks: CandidateInfoBlocks,
    prior: Optional[np.ndarray | CalibrationPrior] = None,
) -> np.ndarray:
    prior_blocks = _coerce_prior_blocks(problem, prior)
    weights = np.asarray(selection, dtype=float)
    h_cal = np.array(prior_blocks.h_tt, copy=True)
    reg = 1e-9 * np.eye(problem.pose_dim, dtype=float)

    for idx, weight in enumerate(weights):
        if weight == 0.0:
            continue
        weighted_h_tp = weight * info_blocks.h_tp[idx]
        h_cal += weight * info_blocks.h_tt[idx]
        pose_block = prior_blocks.h_pp + weight * info_blocks.h_pp[idx]
        h_cal -= weighted_h_tp @ np.linalg.pinv(pose_block + reg) @ weighted_h_tp.T

    return 0.5 * (h_cal + h_cal.T)


def compute_min_eig_score(
    problem: CalibrationProblem,
    selection: Sequence[int] | np.ndarray,
    info_blocks: CandidateInfoBlocks,
    prior: Optional[np.ndarray | CalibrationPrior] = None,
) -> float:
    h_cal = compute_calibration_schur_compact(problem, selection, info_blocks, prior=prior)
    return float(np.min(np.linalg.eigvalsh(h_cal)))


def compute_combined_fim(selection: np.ndarray, inf_mats: np.ndarray, prior: np.ndarray) -> np.ndarray:
    combined = prior.copy()
    for weight, mat in zip(selection, inf_mats):
        combined += float(weight) * mat
    return combined


def compute_calibration_schur(fim: np.ndarray, intrinsics_dim: int) -> np.ndarray:
    """Keep the intrinsics block and marginalize nuisance pose variables."""
    h_tt = fim[:intrinsics_dim, :intrinsics_dim]
    h_tn = fim[:intrinsics_dim, intrinsics_dim:]
    h_nt = fim[intrinsics_dim:, :intrinsics_dim]
    h_nn = fim[intrinsics_dim:, intrinsics_dim:]

    if h_nn.size == 0:
        h_cal = h_tt
    else:
        reg = 1e-9 * np.eye(h_nn.shape[0], dtype=float)
        h_nn_inv = np.linalg.pinv(h_nn + reg)
        h_cal = h_tt - h_tn @ h_nn_inv @ h_nt
    return 0.5 * (h_cal + h_cal.T)


def compute_info_metric(problem: CalibrationProblem, selection: Sequence[int], prior: Optional[np.ndarray] = None) -> float:
    info_blocks = construct_candidate_inf_blocks(problem)
    weights = np.zeros(problem.num_candidates, dtype=float)
    weights[np.asarray(selection, dtype=int)] = 1.0
    return compute_min_eig_score(problem, weights, info_blocks, prior=prior)


def find_min_eig_pair(inf_mats: np.ndarray, selection: np.ndarray, prior: np.ndarray, intrinsics_dim: int):
    final_fim = compute_combined_fim(selection, inf_mats, prior)
    h_cal = compute_calibration_schur(final_fim, intrinsics_dim)
    eigvals, eigvecs = la.eigh(h_cal)
    return eigvals[0], eigvecs[:, 0], final_fim
