from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from ..OASIS.FIM import CalibrationProblem, project_points


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def look_at_rotation(camera_position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return R_wc for a camera whose +z axis points toward the target."""
    forward = _normalize(target - camera_position)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(forward, fallback_up)
        if np.linalg.norm(right) < 1e-9:
            fallback_up = np.array([1.0, 0.0, 0.0], dtype=float)
            right = np.cross(forward, fallback_up)
    right = _normalize(right)
    camera_down = np.cross(forward, right)
    rotation_wc = np.column_stack((right, camera_down, forward))
    return rotation_wc


def _rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rotation_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rotation_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def board_facing_camera_rotation(
    board_position: np.ndarray,
    camera_position: np.ndarray | None = None,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Return R_cb that maps board-frame points into the camera frame."""
    if camera_position is None:
        camera_position = np.zeros(3, dtype=float)
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=float)

    board_normal_c = _normalize(camera_position - board_position)
    board_x_c = np.cross(up, board_normal_c)
    if np.linalg.norm(board_x_c) < 1e-9:
        board_x_c = np.cross(np.array([1.0, 0.0, 0.0], dtype=float), board_normal_c)
    board_x_c = _normalize(board_x_c)
    board_y_c = _normalize(np.cross(board_normal_c, board_x_c))
    return np.column_stack((board_x_c, board_y_c, board_normal_c))


def board_pose_to_camera_pose(rotation_cb: np.ndarray, translation_cb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a moving-board pose with a fixed camera into an equivalent moving-camera pose."""
    rotation_wc = rotation_cb.T
    translation_wc = -rotation_wc @ translation_cb
    return rotation_wc, translation_wc


def create_checkerboard_points(
    rows: int = 6,
    cols: int = 8,
    square_size: float = 0.04,
    center: bool = True,
) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack((xs, ys, np.zeros_like(xs)), axis=-1).reshape(-1, 3).astype(float)
    points[:, 0] *= square_size
    points[:, 1] *= square_size
    if center:
        points[:, 0] -= 0.5 * (cols - 1) * square_size
        points[:, 1] -= 0.5 * (rows - 1) * square_size
    return points


def structured_camera_bank_aim_points(
    target_points: np.ndarray,
    target: np.ndarray | None = None,
    mode: str = "center",
) -> list[tuple[str, np.ndarray]]:
    if target is None:
        target = np.zeros(3, dtype=float)

    points = np.asarray(target_points, dtype=float)
    min_x, max_x = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
    min_y, max_y = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
    mid_x = 0.5 * (min_x + max_x)
    mid_y = 0.5 * (min_y + max_y)

    anchors_local = {
        "center": np.array([mid_x, mid_y, 0.0], dtype=float),
        "top_left": np.array([min_x, max_y, 0.0], dtype=float),
        "top_center": np.array([mid_x, max_y, 0.0], dtype=float),
        "top_right": np.array([max_x, max_y, 0.0], dtype=float),
        "center_left": np.array([min_x, mid_y, 0.0], dtype=float),
        "center_right": np.array([max_x, mid_y, 0.0], dtype=float),
        "bottom_left": np.array([min_x, min_y, 0.0], dtype=float),
        "bottom_center": np.array([mid_x, min_y, 0.0], dtype=float),
        "bottom_right": np.array([max_x, min_y, 0.0], dtype=float),
    }

    if mode == "center":
        labels = ["center"]
    elif mode == "board_anchors":
        labels = [
            "center",
            "top_left",
            "top_center",
            "top_right",
            "center_left",
            "center_right",
            "bottom_left",
            "bottom_center",
            "bottom_right",
        ]
    else:
        raise ValueError(f"Unknown aim-point mode '{mode}'.")

    return [(label, target + anchors_local[label]) for label in labels]


def generate_candidate_camera_poses(
    azimuth_samples: int = 8,
    elevation_samples: int = 3,
    radius_range: Tuple[float, float] = (0.6, 1.2),
    radius_samples: int = 1,
    num_candidate_poses: int | None = None,
    azimuth_span_deg: float = 360.0,
    azimuth_center_deg: float = 0.0,
    target: np.ndarray | None = None,
    up: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if target is None:
        target = np.array([0.0, 0.0, 0.0], dtype=float)
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)

    rotations = []
    translations = []
    elevations = np.linspace(math.radians(15.0), math.radians(60.0), elevation_samples)
    radii = np.linspace(radius_range[0], radius_range[1], radius_samples)
    azimuth_span_rad = math.radians(float(np.clip(azimuth_span_deg, 1e-6, 360.0)))
    azimuth_center_rad = math.radians(azimuth_center_deg)
    azimuth_start = azimuth_center_rad - 0.5 * azimuth_span_rad

    if num_candidate_poses is not None:
        golden_ratio_conjugate = (math.sqrt(5.0) - 1.0) / 2.0
        shell_counts = np.full(radius_samples, num_candidate_poses // radius_samples, dtype=int)
        shell_counts[: num_candidate_poses % radius_samples] += 1
        sin_min = math.sin(math.radians(15.0))
        sin_max = math.sin(math.radians(60.0))

        for shell_idx, (radius, shell_count) in enumerate(zip(radii, shell_counts)):
            for idx in range(shell_count):
                u = (idx + 0.5) / shell_count
                # Offset each shell so radius discretization does not create stacked rays.
                azimuth_unit = ((idx + shell_idx / max(radius_samples, 1)) * golden_ratio_conjugate) % 1.0
                azimuth = azimuth_start + azimuth_unit * azimuth_span_rad
                elevation = math.asin(sin_min + u * (sin_max - sin_min))
                position = np.array(
                    [
                        radius * math.cos(elevation) * math.cos(azimuth),
                        radius * math.cos(elevation) * math.sin(azimuth),
                        radius * math.sin(elevation),
                    ],
                    dtype=float,
                )
                rotations.append(look_at_rotation(position, target, up))
                translations.append(position)
        return np.asarray(rotations), np.asarray(translations)

    azimuths = np.linspace(azimuth_start, azimuth_start + azimuth_span_rad, azimuth_samples, endpoint=False)
    for radius in radii:
        for elevation in elevations:
            for azimuth in azimuths:
                position = np.array(
                    [
                        radius * math.cos(elevation) * math.cos(azimuth),
                        radius * math.cos(elevation) * math.sin(azimuth),
                        radius * math.sin(elevation),
                    ],
                    dtype=float,
                )
                rotations.append(look_at_rotation(position, target, up))
                translations.append(position)
    return np.asarray(rotations), np.asarray(translations)


def generate_structured_candidate_camera_bank(
    image_size: Tuple[int, int] = (1280, 960),
    intrinsics: np.ndarray | None = None,
    radius_levels_m: Sequence[float] = (0.35, 0.50, 0.70, 0.95),
    azimuth_levels_deg: Sequence[float] = (-30.0, -15.0, 0.0, 15.0, 30.0),
    elevation_levels_deg: Sequence[float] = (-25.0, -10.0, 0.0, 10.0, 25.0),
    roll_levels_deg: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
    yaw_perturb_levels_deg: Sequence[float] = (-10.0, -5.0, 0.0, 5.0, 10.0),
    pitch_perturb_levels_deg: Sequence[float] = (-10.0, -5.0, 0.0, 5.0, 10.0),
    board_rows: int = 9,
    board_cols: int = 9,
    board_square_size: float = 0.125,
    aim_point_mode: str = "center",
    target: np.ndarray | None = None,
    up: np.ndarray | None = None,
    min_target_area_fraction: float = 0.02,
    max_target_area_fraction: float = 0.55,
    max_slant_deg: float = 65.0,
    min_projected_corner_spread_px: float = 8.0,
) -> dict:
    if intrinsics is None:
        intrinsics = np.array([820.0, 815.0, image_size[0] / 2.0, image_size[1] / 2.0], dtype=float)
    if target is None:
        target = np.zeros(3, dtype=float)
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)

    target_points = create_checkerboard_points(rows=board_rows, cols=board_cols, square_size=board_square_size)
    aim_points = structured_camera_bank_aim_points(target_points, target=target, mode=aim_point_mode)
    rotations = []
    translations = []
    metadata = []

    for radius in radius_levels_m:
        for azimuth_deg in azimuth_levels_deg:
            azimuth = math.radians(azimuth_deg)
            for elevation_deg in elevation_levels_deg:
                elevation = math.radians(elevation_deg)
                direction = np.array(
                    [
                        math.tan(azimuth),
                        math.tan(elevation),
                        1.0,
                    ],
                    dtype=float,
                )
                direction = _normalize(direction)
                position = target + radius * direction
                for aim_label, aim_point in aim_points:
                    base_rotation = look_at_rotation(position, aim_point, up)
                    for roll_deg in roll_levels_deg:
                        roll_rad = math.radians(roll_deg)
                        for yaw_deg in yaw_perturb_levels_deg:
                            yaw_rad = math.radians(yaw_deg)
                            for pitch_deg in pitch_perturb_levels_deg:
                                pitch_rad = math.radians(pitch_deg)
                                local_rotation = _rotation_z(roll_rad) @ _rotation_y(yaw_rad) @ _rotation_x(pitch_rad)
                                rotation = base_rotation @ local_rotation
                                rotations.append(rotation)
                                translations.append(position)
                                metadata.append(
                                    {
                                        "radius_m": float(radius),
                                        "azimuth_deg": float(azimuth_deg),
                                        "elevation_deg": float(elevation_deg),
                                        "roll_deg": float(roll_deg),
                                        "yaw_perturb_deg": float(yaw_deg),
                                        "pitch_perturb_deg": float(pitch_deg),
                                        "aim_point_label": aim_label,
                                        "aim_point_world": np.asarray(aim_point, dtype=float).tolist(),
                                    }
                                )

    rotations = np.asarray(rotations, dtype=float)
    translations = np.asarray(translations, dtype=float)
    measurements = generate_measurements(
        target_points,
        rotations,
        translations,
        np.asarray(intrinsics, dtype=float),
        image_size,
        pixel_noise_sigma=0.0,
    )

    valid_mask = np.ones(rotations.shape[0], dtype=bool)
    diagnostics = []
    reason_counts = {
        "all_corners_visible": 0,
        "target_area_fraction": 0,
        "slant_angle": 0,
        "corner_spread": 0,
    }
    board_normal_world = np.array([0.0, 0.0, 1.0], dtype=float)

    for idx, (rotation_wc, translation_wc, measurement) in enumerate(zip(rotations, translations, measurements)):
        all_corners_visible = bool(np.isfinite(measurement).all())
        area_fraction = compute_projected_bbox_area_fraction(measurement, image_size)
        viewing_dir_world = _normalize(target - translation_wc)
        slant_angle_deg = float(np.degrees(np.arccos(np.clip(abs(np.dot(viewing_dir_world, board_normal_world)), -1.0, 1.0))))
        min_corner_spread_px = compute_min_projected_corner_spread(measurement, board_rows, board_cols)

        area_ok = min_target_area_fraction <= area_fraction <= max_target_area_fraction
        slant_ok = slant_angle_deg <= max_slant_deg
        spread_ok = min_corner_spread_px >= min_projected_corner_spread_px
        is_valid = all_corners_visible and area_ok and slant_ok and spread_ok
        valid_mask[idx] = is_valid

        if not all_corners_visible:
            reason_counts["all_corners_visible"] += 1
        if not area_ok:
            reason_counts["target_area_fraction"] += 1
        if not slant_ok:
            reason_counts["slant_angle"] += 1
        if not spread_ok:
            reason_counts["corner_spread"] += 1

        diagnostics.append(
            {
                **metadata[idx],
                "all_corners_visible": all_corners_visible,
                "target_area_fraction": area_fraction,
                "slant_angle_deg": slant_angle_deg,
                "min_projected_corner_spread_px": min_corner_spread_px,
                "valid": bool(is_valid),
            }
        )

    return {
        "target_points": target_points,
        "intrinsics": np.asarray(intrinsics, dtype=float),
        "candidate_rotations": rotations,
        "candidate_translations": translations,
        "measurements": measurements,
        "valid_mask": valid_mask,
        "diagnostics": diagnostics,
        "reason_counts": reason_counts,
        "filter_parameters": {
            "min_target_area_fraction": float(min_target_area_fraction),
            "max_target_area_fraction": float(max_target_area_fraction),
            "max_slant_deg": float(max_slant_deg),
            "min_projected_corner_spread_px": float(min_projected_corner_spread_px),
        },
        "aim_point_mode": aim_point_mode,
    }


def generate_random_candidate_camera_bank(
    num_candidates: int,
    image_size: Tuple[int, int] = (1280, 960),
    intrinsics: np.ndarray | None = None,
    radius_range_m: Tuple[float, float] = (0.5, 10.0),
    azimuth_range_deg: Tuple[float, float] = (-30.0, 30.0),
    elevation_range_deg: Tuple[float, float] = (-25.0, 25.0),
    roll_range_deg: Tuple[float, float] = (0.0, 135.0),
    yaw_perturb_range_deg: Tuple[float, float] = (-10.0, 10.0),
    pitch_perturb_range_deg: Tuple[float, float] = (-10.0, 10.0),
    board_rows: int = 9,
    board_cols: int = 9,
    board_square_size: float = 0.125,
    aim_point_mode: str = "center",
    target: np.ndarray | None = None,
    up: np.ndarray | None = None,
    min_target_area_fraction: float = 0.02,
    max_target_area_fraction: float = 0.55,
    max_slant_deg: float = 65.0,
    min_projected_corner_spread_px: float = 8.0,
    seed: int = 0,
) -> dict:
    if intrinsics is None:
        intrinsics = np.array([820.0, 815.0, image_size[0] / 2.0, image_size[1] / 2.0], dtype=float)
    if target is None:
        target = np.zeros(3, dtype=float)
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)

    rng = np.random.default_rng(seed)
    target_points = create_checkerboard_points(rows=board_rows, cols=board_cols, square_size=board_square_size)
    aim_points = structured_camera_bank_aim_points(target_points, target=target, mode=aim_point_mode)
    aim_labels = [label for label, _ in aim_points]
    aim_targets = [point for _, point in aim_points]

    rotations = []
    translations = []
    metadata = []

    for _ in range(int(num_candidates)):
        radius = float(rng.uniform(radius_range_m[0], radius_range_m[1]))
        azimuth_deg = float(rng.uniform(azimuth_range_deg[0], azimuth_range_deg[1]))
        elevation_deg = float(rng.uniform(elevation_range_deg[0], elevation_range_deg[1]))
        roll_deg = float(rng.uniform(roll_range_deg[0], roll_range_deg[1]))
        yaw_deg = float(rng.uniform(yaw_perturb_range_deg[0], yaw_perturb_range_deg[1]))
        pitch_deg = float(rng.uniform(pitch_perturb_range_deg[0], pitch_perturb_range_deg[1]))
        aim_idx = int(rng.integers(0, len(aim_points)))

        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        direction = np.array([math.tan(azimuth), math.tan(elevation), 1.0], dtype=float)
        direction = _normalize(direction)
        position = target + radius * direction

        base_rotation = look_at_rotation(position, aim_targets[aim_idx], up)
        local_rotation = (
            _rotation_z(math.radians(roll_deg))
            @ _rotation_y(math.radians(yaw_deg))
            @ _rotation_x(math.radians(pitch_deg))
        )
        rotation = base_rotation @ local_rotation

        rotations.append(rotation)
        translations.append(position)
        metadata.append(
            {
                "radius_m": radius,
                "azimuth_deg": azimuth_deg,
                "elevation_deg": elevation_deg,
                "roll_deg": roll_deg,
                "yaw_perturb_deg": yaw_deg,
                "pitch_perturb_deg": pitch_deg,
                "aim_point_label": aim_labels[aim_idx],
                "aim_point_world": np.asarray(aim_targets[aim_idx], dtype=float).tolist(),
            }
        )

    rotations = np.asarray(rotations, dtype=float)
    translations = np.asarray(translations, dtype=float)
    measurements = generate_measurements(
        target_points,
        rotations,
        translations,
        np.asarray(intrinsics, dtype=float),
        image_size,
        pixel_noise_sigma=0.0,
    )

    valid_mask = np.ones(rotations.shape[0], dtype=bool)
    diagnostics = []
    reason_counts = {
        "all_corners_visible": 0,
        "target_area_fraction": 0,
        "slant_angle": 0,
        "corner_spread": 0,
    }
    board_normal_world = np.array([0.0, 0.0, 1.0], dtype=float)

    for idx, (rotation_wc, translation_wc, measurement) in enumerate(zip(rotations, translations, measurements)):
        all_corners_visible = bool(np.isfinite(measurement).all())
        area_fraction = compute_projected_bbox_area_fraction(measurement, image_size)
        viewing_dir_world = _normalize(target - translation_wc)
        slant_angle_deg = float(np.degrees(np.arccos(np.clip(abs(np.dot(viewing_dir_world, board_normal_world)), -1.0, 1.0))))
        min_corner_spread_px = compute_min_projected_corner_spread(measurement, board_rows, board_cols)

        area_ok = min_target_area_fraction <= area_fraction <= max_target_area_fraction
        slant_ok = slant_angle_deg <= max_slant_deg
        spread_ok = min_corner_spread_px >= min_projected_corner_spread_px
        is_valid = all_corners_visible and area_ok and slant_ok and spread_ok
        valid_mask[idx] = is_valid

        if not all_corners_visible:
            reason_counts["all_corners_visible"] += 1
        if not area_ok:
            reason_counts["target_area_fraction"] += 1
        if not slant_ok:
            reason_counts["slant_angle"] += 1
        if not spread_ok:
            reason_counts["corner_spread"] += 1

        diagnostics.append(
            {
                **metadata[idx],
                "all_corners_visible": all_corners_visible,
                "target_area_fraction": area_fraction,
                "slant_angle_deg": slant_angle_deg,
                "min_projected_corner_spread_px": min_corner_spread_px,
                "valid": bool(is_valid),
            }
        )

    return {
        "target_points": target_points,
        "intrinsics": np.asarray(intrinsics, dtype=float),
        "candidate_rotations": rotations,
        "candidate_translations": translations,
        "measurements": measurements,
        "valid_mask": valid_mask,
        "diagnostics": diagnostics,
        "reason_counts": reason_counts,
        "filter_parameters": {
            "min_target_area_fraction": float(min_target_area_fraction),
            "max_target_area_fraction": float(max_target_area_fraction),
            "max_slant_deg": float(max_slant_deg),
            "min_projected_corner_spread_px": float(min_projected_corner_spread_px),
        },
        "aim_point_mode": aim_point_mode,
        "sampler_mode": "random",
        "random_parameters": {
            "num_candidates": int(num_candidates),
            "radius_range_m": [float(v) for v in radius_range_m],
            "azimuth_range_deg": [float(v) for v in azimuth_range_deg],
            "elevation_range_deg": [float(v) for v in elevation_range_deg],
            "roll_range_deg": [float(v) for v in roll_range_deg],
            "yaw_perturb_range_deg": [float(v) for v in yaw_perturb_range_deg],
            "pitch_perturb_range_deg": [float(v) for v in pitch_perturb_range_deg],
            "seed": int(seed),
        },
    }


def build_camera_problem_from_candidate_bank(
    bank: dict,
    image_size: Tuple[int, int] = (1280, 960),
    pixel_noise_sigma: float = 1.0,
    seed: int = 0,
    filtered_only: bool = True,
) -> CalibrationProblem:
    rng = np.random.default_rng(seed)
    mask = bank["valid_mask"] if filtered_only else np.ones(bank["valid_mask"].shape[0], dtype=bool)
    target_points = np.asarray(bank["target_points"], dtype=float)
    intrinsics_gt = np.asarray(bank["intrinsics"], dtype=float)
    intrinsics_init = intrinsics_gt + np.array([25.0, -20.0, 8.0, -6.0], dtype=float)
    candidate_rotations = np.asarray(bank["candidate_rotations"], dtype=float)[mask]
    candidate_translations = np.asarray(bank["candidate_translations"], dtype=float)[mask]
    measurements = generate_measurements(
        target_points,
        candidate_rotations,
        candidate_translations,
        intrinsics_gt,
        image_size,
        pixel_noise_sigma=pixel_noise_sigma,
        rng=rng,
    )

    return CalibrationProblem(
        target_points=target_points,
        candidate_rotations=candidate_rotations,
        candidate_translations=candidate_translations,
        measurements=measurements,
        intrinsics_gt=intrinsics_gt,
        intrinsics_init=intrinsics_init,
        image_size=image_size,
        pixel_noise_sigma=pixel_noise_sigma,
        camera_is_fixed=False,
    )


def generate_candidate_checkerboard_poses_fixed_camera(
    num_candidate_poses: int,
    radius_range: Tuple[float, float] = (4.0, 6.0),
    azimuth_span_deg: float = 360.0,
    azimuth_center_deg: float = 0.0,
    offaxis_angle_range_deg: Tuple[float, float] = (2.0, 25.0),
    tilt_span_deg: float = 30.0,
    azimuth_anchor_count: int | None = None,
    symmetry_strength: float = 0.0,
    camera_position: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if camera_position is None:
        camera_position = np.zeros(3, dtype=float)

    symmetry_strength = float(np.clip(symmetry_strength, 0.0, 1.0))
    azimuth_span_rad = math.radians(float(np.clip(azimuth_span_deg, 1e-6, 360.0)))
    azimuth_center_rad = math.radians(azimuth_center_deg)
    azimuth_start = azimuth_center_rad - 0.5 * azimuth_span_rad
    offaxis_min_rad = math.radians(offaxis_angle_range_deg[0])
    offaxis_max_rad = math.radians(offaxis_angle_range_deg[1])
    tilt_span_rad = math.radians(tilt_span_deg)
    golden_ratio_conjugate = (math.sqrt(5.0) - 1.0) / 2.0

    rotations_cb = []
    translations_cb = []
    pair_count = max((num_candidate_poses + 1) // 2, 1)
    anchor_count = None if azimuth_anchor_count is None else max(int(azimuth_anchor_count), 1)

    for idx in range(num_candidate_poses):
        pair_idx = idx // 2
        mirror_sign = 1.0 if (idx % 2 == 0) else -1.0
        radius_u = ((idx + 0.5) * golden_ratio_conjugate) % 1.0
        offaxis_u = ((idx + 0.5) / num_candidate_poses)
        tilt_u = ((idx + 0.25) * golden_ratio_conjugate) % 1.0
        tilt_v = ((idx + 0.75) * golden_ratio_conjugate) % 1.0

        asym_azimuth = azimuth_start + (((idx + 0.5) * golden_ratio_conjugate) % 1.0) * azimuth_span_rad
        if anchor_count is not None:
            anchor_phase = (pair_idx + 0.5) / anchor_count
            sym_azimuth = azimuth_start + (anchor_phase % 1.0) * azimuth_span_rad
        else:
            sym_abs = ((pair_idx + 0.5) / pair_count) * 0.5 * azimuth_span_rad
            sym_azimuth = azimuth_center_rad + mirror_sign * sym_abs
        azimuth = (1.0 - symmetry_strength) * asym_azimuth + symmetry_strength * sym_azimuth

        radius = radius_range[0] + radius_u * (radius_range[1] - radius_range[0])
        offaxis = offaxis_min_rad + offaxis_u * (offaxis_max_rad - offaxis_min_rad)
        position_cb = np.array(
            [
                radius * math.sin(offaxis) * math.cos(azimuth),
                radius * math.sin(offaxis) * math.sin(azimuth),
                radius * math.cos(offaxis),
            ],
            dtype=float,
        )

        base_rotation_cb = board_facing_camera_rotation(position_cb, camera_position=camera_position)
        asym_tilt_x = (2.0 * tilt_u - 1.0) * tilt_span_rad
        asym_tilt_y = (2.0 * tilt_v - 1.0) * tilt_span_rad
        sym_tilt_x = mirror_sign * ((pair_idx + 0.5) / pair_count) * tilt_span_rad
        tilt_x = (1.0 - symmetry_strength) * asym_tilt_x + symmetry_strength * sym_tilt_x
        tilt_y = (1.0 - symmetry_strength) * asym_tilt_y
        local_tilt = _rotation_y(tilt_y) @ _rotation_x(tilt_x)
        rotation_cb = base_rotation_cb @ local_tilt

        rotations_cb.append(rotation_cb)
        translations_cb.append(position_cb)

    return np.asarray(rotations_cb), np.asarray(translations_cb)


def generate_fixed_camera_measurements(
    target_points_board: np.ndarray,
    candidate_board_rotations: np.ndarray,
    candidate_board_translations: np.ndarray,
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
    pixel_noise_sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    width, height = image_size
    measurements = np.full((candidate_board_rotations.shape[0], target_points_board.shape[0], 2), np.nan, dtype=float)
    for idx, (rotation_cb, translation_cb) in enumerate(zip(candidate_board_rotations, candidate_board_translations)):
        points_c = (rotation_cb @ target_points_board.T).T + translation_cb
        z = points_c[:, 2]
        proj = np.full((target_points_board.shape[0], 2), np.nan, dtype=float)
        valid = z > 1e-9
        if np.any(valid):
            x_norm = points_c[valid, 0] / z[valid]
            y_norm = points_c[valid, 1] / z[valid]
            proj[valid, 0] = intrinsics[0] * x_norm + intrinsics[2]
            proj[valid, 1] = intrinsics[1] * y_norm + intrinsics[3]

        valid = (
            np.isfinite(proj).all(axis=1)
            & (proj[:, 0] >= 0.0)
            & (proj[:, 0] < width)
            & (proj[:, 1] >= 0.0)
            & (proj[:, 1] < height)
        )
        noisy = proj.copy()
        noisy[valid] += rng.normal(scale=pixel_noise_sigma, size=(np.count_nonzero(valid), 2))
        noisy[~valid] = np.nan
        measurements[idx] = noisy
    return measurements


def compute_projected_bbox_area_fraction(measurements: np.ndarray, image_size: Tuple[int, int]) -> float:
    valid = np.isfinite(measurements).all(axis=1)
    if not np.any(valid):
        return 0.0
    coords = measurements[valid]
    bbox_w = float(np.max(coords[:, 0]) - np.min(coords[:, 0]))
    bbox_h = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))
    image_area = float(image_size[0] * image_size[1])
    if image_area <= 0.0:
        return 0.0
    return (bbox_w * bbox_h) / image_area


def compute_min_projected_corner_spread(
    measurements: np.ndarray,
    board_rows: int,
    board_cols: int,
) -> float:
    points = measurements.reshape(board_rows, board_cols, 2)
    spreads = []
    for row in range(board_rows):
        for col in range(board_cols - 1):
            p0 = points[row, col]
            p1 = points[row, col + 1]
            if np.isfinite(p0).all() and np.isfinite(p1).all():
                spreads.append(float(np.linalg.norm(p1 - p0)))
    for row in range(board_rows - 1):
        for col in range(board_cols):
            p0 = points[row, col]
            p1 = points[row + 1, col]
            if np.isfinite(p0).all() and np.isfinite(p1).all():
                spreads.append(float(np.linalg.norm(p1 - p0)))
    if not spreads:
        return 0.0
    return float(min(spreads))


def generate_discrete_candidate_checkerboard_bank_fixed_camera(
    image_size: Tuple[int, int] = (1280, 960),
    intrinsics: np.ndarray | None = None,
    depths_m: Sequence[float] = (0.35, 0.50, 0.70, 0.95),
    region_grid_shape: Tuple[int, int] = (3, 3),
    tilts_deg: Sequence[float] = (0.0, -15.0, 15.0, -30.0, 30.0),
    rolls_deg: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
    board_rows: int = 9,
    board_cols: int = 9,
    board_square_size: float = 0.125,
    min_target_area_fraction: float = 0.02,
    max_target_area_fraction: float = 0.65,
    max_slant_deg: float = 60.0,
    min_projected_corner_spread_px: float = 12.0,
) -> dict:
    if intrinsics is None:
        intrinsics = np.array([820.0, 815.0, image_size[0] / 2.0, image_size[1] / 2.0], dtype=float)
    fx, fy, cx, cy = [float(v) for v in intrinsics]
    board_points = create_checkerboard_points(rows=board_rows, cols=board_cols, square_size=board_square_size)

    grid_rows, grid_cols = region_grid_shape
    u_centers = (np.arange(grid_cols, dtype=float) + 0.5) * image_size[0] / grid_cols
    v_centers = (np.arange(grid_rows, dtype=float) + 0.5) * image_size[1] / grid_rows

    board_rotations = []
    board_translations = []
    metadata = []

    for depth in depths_m:
        for grid_row, v_center in enumerate(v_centers):
            for grid_col, u_center in enumerate(u_centers):
                x_norm = (u_center - cx) / fx
                y_norm = (v_center - cy) / fy
                translation_cb = np.array([x_norm * depth, y_norm * depth, depth], dtype=float)
                base_rotation_cb = board_facing_camera_rotation(translation_cb, camera_position=np.zeros(3, dtype=float))
                for tilt_deg in tilts_deg:
                    for roll_deg in rolls_deg:
                        local_rotation = _rotation_z(math.radians(roll_deg)) @ _rotation_x(math.radians(tilt_deg))
                        rotation_cb = base_rotation_cb @ local_rotation
                        board_rotations.append(rotation_cb)
                        board_translations.append(translation_cb)
                        metadata.append(
                            {
                                "depth_m": float(depth),
                                "grid_row": int(grid_row),
                                "grid_col": int(grid_col),
                                "image_center_px": [float(u_center), float(v_center)],
                                "tilt_deg": float(tilt_deg),
                                "roll_deg": float(roll_deg),
                            }
                        )

    board_rotations = np.asarray(board_rotations, dtype=float)
    board_translations = np.asarray(board_translations, dtype=float)
    measurements = generate_fixed_camera_measurements(
        board_points,
        board_rotations,
        board_translations,
        intrinsics,
        image_size,
        pixel_noise_sigma=0.0,
    )

    valid_mask = np.ones(board_rotations.shape[0], dtype=bool)
    reason_counts = {
        "all_corners_visible": 0,
        "target_area_fraction": 0,
        "slant_angle": 0,
        "corner_spread": 0,
    }
    per_candidate_diagnostics = []
    camera_rotations = []
    camera_translations = []

    for idx, (rotation_cb, translation_cb, measurement) in enumerate(zip(board_rotations, board_translations, measurements)):
        all_corners_visible = bool(np.isfinite(measurement).all())
        area_fraction = compute_projected_bbox_area_fraction(measurement, image_size)
        board_normal_c = rotation_cb[:, 2]
        slant_angle_deg = float(np.degrees(np.arccos(np.clip(-board_normal_c[2], -1.0, 1.0))))
        min_corner_spread_px = compute_min_projected_corner_spread(measurement, board_rows, board_cols)

        area_ok = min_target_area_fraction <= area_fraction <= max_target_area_fraction
        slant_ok = slant_angle_deg <= max_slant_deg
        spread_ok = min_corner_spread_px >= min_projected_corner_spread_px
        is_valid = all_corners_visible and area_ok and slant_ok and spread_ok
        valid_mask[idx] = is_valid

        if not all_corners_visible:
            reason_counts["all_corners_visible"] += 1
        if not area_ok:
            reason_counts["target_area_fraction"] += 1
        if not slant_ok:
            reason_counts["slant_angle"] += 1
        if not spread_ok:
            reason_counts["corner_spread"] += 1

        image_center = np.nanmean(measurement, axis=0)
        camera_rotation, camera_translation = board_pose_to_camera_pose(rotation_cb, translation_cb)
        camera_rotations.append(camera_rotation)
        camera_translations.append(camera_translation)
        per_candidate_diagnostics.append(
            {
                **metadata[idx],
                "all_corners_visible": all_corners_visible,
                "target_area_fraction": area_fraction,
                "slant_angle_deg": slant_angle_deg,
                "min_projected_corner_spread_px": min_corner_spread_px,
                "valid": bool(is_valid),
                "projected_center_px": image_center.tolist(),
            }
        )

    return {
        "target_points": board_points,
        "intrinsics": np.asarray(intrinsics, dtype=float),
        "board_rotations": board_rotations,
        "board_translations": board_translations,
        "camera_rotations": np.asarray(camera_rotations, dtype=float),
        "camera_translations": np.asarray(camera_translations, dtype=float),
        "measurements": measurements,
        "valid_mask": valid_mask,
        "diagnostics": per_candidate_diagnostics,
        "reason_counts": reason_counts,
        "filter_parameters": {
            "min_target_area_fraction": float(min_target_area_fraction),
            "max_target_area_fraction": float(max_target_area_fraction),
            "max_slant_deg": float(max_slant_deg),
            "min_projected_corner_spread_px": float(min_projected_corner_spread_px),
        },
    }


def build_fixed_camera_problem_from_candidate_bank(
    bank: dict,
    image_size: Tuple[int, int] = (1280, 960),
    pixel_noise_sigma: float = 1.0,
    seed: int = 0,
    filtered_only: bool = True,
) -> CalibrationProblem:
    rng = np.random.default_rng(seed)
    mask = bank["valid_mask"] if filtered_only else np.ones(bank["valid_mask"].shape[0], dtype=bool)
    target_points = np.asarray(bank["target_points"], dtype=float)
    intrinsics_gt = np.asarray(bank["intrinsics"], dtype=float)
    intrinsics_init = intrinsics_gt + np.array([25.0, -20.0, 8.0, -6.0], dtype=float)

    board_rotations = np.asarray(bank["board_rotations"], dtype=float)[mask]
    board_translations = np.asarray(bank["board_translations"], dtype=float)[mask]
    camera_rotations = np.asarray(bank["camera_rotations"], dtype=float)[mask]
    camera_translations = np.asarray(bank["camera_translations"], dtype=float)[mask]
    measurements = generate_fixed_camera_measurements(
        target_points,
        board_rotations,
        board_translations,
        intrinsics_gt,
        image_size,
        pixel_noise_sigma=pixel_noise_sigma,
        rng=rng,
    )

    return CalibrationProblem(
        target_points=target_points,
        candidate_rotations=camera_rotations,
        candidate_translations=camera_translations,
        measurements=measurements,
        intrinsics_gt=intrinsics_gt,
        intrinsics_init=intrinsics_init,
        image_size=image_size,
        pixel_noise_sigma=pixel_noise_sigma,
        candidate_board_rotations=board_rotations,
        candidate_board_translations=board_translations,
        camera_is_fixed=True,
    )


def generate_measurements(
    target_points: np.ndarray,
    candidate_rotations: np.ndarray,
    candidate_translations: np.ndarray,
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
    pixel_noise_sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    width, height = image_size
    measurements = np.full((candidate_rotations.shape[0], target_points.shape[0], 2), np.nan, dtype=float)
    for idx, (rotation_wc, translation_wc) in enumerate(zip(candidate_rotations, candidate_translations)):
        projected = project_points(target_points, rotation_wc, translation_wc, intrinsics)
        valid = (
            np.isfinite(projected).all(axis=1)
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < height)
        )
        noisy = projected.copy()
        noisy[valid] += rng.normal(scale=pixel_noise_sigma, size=(np.count_nonzero(valid), 2))
        noisy[~valid] = np.nan
        measurements[idx] = noisy
    return measurements


def generate_calibration_problem(
    image_size: Tuple[int, int] = (1280, 960),
    pixel_noise_sigma: float = 1.0,
    seed: int = 0,
    num_candidate_poses: int | None = None,
    azimuth_samples: int = 8,
    elevation_samples: int = 3,
    radius_samples: int = 1,
    radius_range: Tuple[float, float] = (0.6, 1.2),
    azimuth_span_deg: float = 360.0,
    azimuth_center_deg: float = 0.0,
    board_rows: int = 9,
    board_cols: int = 9,
    board_square_size: float = 0.125,
) -> CalibrationProblem:
    rng = np.random.default_rng(seed)
    intrinsics_gt = np.array([820.0, 815.0, image_size[0] / 2.0, image_size[1] / 2.0], dtype=float)
    intrinsics_init = intrinsics_gt + np.array([25.0, -20.0, 8.0, -6.0], dtype=float)

    target_points = create_checkerboard_points(rows=board_rows, cols=board_cols, square_size=board_square_size)
    candidate_rotations, candidate_translations = generate_candidate_camera_poses(
        azimuth_samples=azimuth_samples,
        elevation_samples=elevation_samples,
        radius_range=radius_range,
        radius_samples=radius_samples,
        num_candidate_poses=num_candidate_poses,
        azimuth_span_deg=azimuth_span_deg,
        azimuth_center_deg=azimuth_center_deg,
    )
    measurements = generate_measurements(
        target_points,
        candidate_rotations,
        candidate_translations,
        intrinsics_gt,
        image_size,
        pixel_noise_sigma=pixel_noise_sigma,
        rng=rng,
    )

    return CalibrationProblem(
        target_points=target_points,
        candidate_rotations=candidate_rotations,
        candidate_translations=candidate_translations,
        measurements=measurements,
        intrinsics_gt=intrinsics_gt,
        intrinsics_init=intrinsics_init,
        image_size=image_size,
        pixel_noise_sigma=pixel_noise_sigma,
    )


def generate_problem_set(
    num_runs: int,
    image_size: Tuple[int, int] = (1280, 960),
    pixel_noise_sigma: float = 1.0,
    num_candidate_poses: int | None = None,
    azimuth_samples: int = 8,
    elevation_samples: int = 3,
    radius_samples: int = 1,
    radius_range: Tuple[float, float] = (0.6, 1.2),
    azimuth_span_deg: float = 360.0,
    azimuth_center_deg: float = 0.0,
    board_rows: int = 9,
    board_cols: int = 9,
    board_square_size: float = 0.125,
):
    return [
        generate_calibration_problem(
            image_size=image_size,
            pixel_noise_sigma=pixel_noise_sigma,
            seed=seed,
            num_candidate_poses=num_candidate_poses,
            azimuth_samples=azimuth_samples,
            elevation_samples=elevation_samples,
            radius_samples=radius_samples,
            radius_range=radius_range,
            azimuth_span_deg=azimuth_span_deg,
            azimuth_center_deg=azimuth_center_deg,
            board_rows=board_rows,
            board_cols=board_cols,
            board_square_size=board_square_size,
        )
        for seed in range(num_runs)
    ]


def generate_fixed_camera_calibration_problem(
    image_size: Tuple[int, int] = (1280, 960),
    pixel_noise_sigma: float = 1.0,
    seed: int = 0,
    num_candidate_poses: int = 1000,
    radius_range: Tuple[float, float] = (4.0, 6.0),
    azimuth_span_deg: float = 360.0,
    azimuth_center_deg: float = 0.0,
    offaxis_angle_range_deg: Tuple[float, float] = (2.0, 25.0),
    tilt_span_deg: float = 30.0,
    azimuth_anchor_count: int | None = None,
    symmetry_strength: float = 0.0,
    board_rows: int = 9,
    board_cols: int = 9,
    board_square_size: float = 0.125,
) -> CalibrationProblem:
    rng = np.random.default_rng(seed)
    intrinsics_gt = np.array([820.0, 815.0, image_size[0] / 2.0, image_size[1] / 2.0], dtype=float)
    intrinsics_init = intrinsics_gt + np.array([25.0, -20.0, 8.0, -6.0], dtype=float)
    target_points = create_checkerboard_points(rows=board_rows, cols=board_cols, square_size=board_square_size)
    board_rotations, board_translations = generate_candidate_checkerboard_poses_fixed_camera(
        num_candidate_poses=num_candidate_poses,
        radius_range=radius_range,
        azimuth_span_deg=azimuth_span_deg,
        azimuth_center_deg=azimuth_center_deg,
        offaxis_angle_range_deg=offaxis_angle_range_deg,
        tilt_span_deg=tilt_span_deg,
        azimuth_anchor_count=azimuth_anchor_count,
        symmetry_strength=symmetry_strength,
    )
    measurements = generate_fixed_camera_measurements(
        target_points,
        board_rotations,
        board_translations,
        intrinsics_gt,
        image_size,
        pixel_noise_sigma=pixel_noise_sigma,
        rng=rng,
    )
    camera_rotations = []
    camera_translations = []
    for rotation_cb, translation_cb in zip(board_rotations, board_translations):
        rotation_wc, translation_wc = board_pose_to_camera_pose(rotation_cb, translation_cb)
        camera_rotations.append(rotation_wc)
        camera_translations.append(translation_wc)

    return CalibrationProblem(
        target_points=target_points,
        candidate_rotations=np.asarray(camera_rotations),
        candidate_translations=np.asarray(camera_translations),
        measurements=measurements,
        intrinsics_gt=intrinsics_gt,
        intrinsics_init=intrinsics_init,
        image_size=image_size,
        pixel_noise_sigma=pixel_noise_sigma,
        candidate_board_rotations=board_rotations,
        candidate_board_translations=board_translations,
        camera_is_fixed=True,
    )


def generate_fixed_camera_problem_set(
    num_runs: int,
    image_size: Tuple[int, int] = (1280, 960),
    pixel_noise_sigma: float = 1.0,
    num_candidate_poses: int = 1000,
    radius_range: Tuple[float, float] = (4.0, 6.0),
    azimuth_span_deg: float = 360.0,
    azimuth_center_deg: float = 0.0,
    offaxis_angle_range_deg: Tuple[float, float] = (2.0, 25.0),
    tilt_span_deg: float = 30.0,
    azimuth_anchor_count: int | None = None,
    symmetry_strength: float = 0.0,
    board_rows: int = 9,
    board_cols: int = 9,
    board_square_size: float = 0.125,
):
    return [
        generate_fixed_camera_calibration_problem(
            image_size=image_size,
            pixel_noise_sigma=pixel_noise_sigma,
            seed=seed,
            num_candidate_poses=num_candidate_poses,
            radius_range=radius_range,
            azimuth_span_deg=azimuth_span_deg,
            azimuth_center_deg=azimuth_center_deg,
            offaxis_angle_range_deg=offaxis_angle_range_deg,
            tilt_span_deg=tilt_span_deg,
            azimuth_anchor_count=azimuth_anchor_count,
            symmetry_strength=symmetry_strength,
            board_rows=board_rows,
            board_cols=board_cols,
            board_square_size=board_square_size,
        )
        for seed in range(num_runs)
    ]
