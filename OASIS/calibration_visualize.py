from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from . import FIM as infmat


REPORT_KEYS = ["min_eig", "logdet", "trace_cov", "cond", "visible_points"]


def _camera_forward(rotation_wc: np.ndarray) -> np.ndarray:
    return rotation_wc[:, 2]


def _checkerboard_corners(problem: infmat.CalibrationProblem) -> np.ndarray:
    points = np.asarray(problem.target_points, dtype=float)
    min_x, max_x = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
    min_y, max_y = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
    return np.array(
        [
            [min_x, min_y, 0.0],
            [max_x, min_y, 0.0],
            [max_x, max_y, 0.0],
            [min_x, max_y, 0.0],
        ],
        dtype=float,
    )


def _plot_checkerboard_plane(
    ax,
    corners: np.ndarray,
    rotation_cb: np.ndarray,
    translation_cb: np.ndarray,
    facecolor: str,
    alpha: float,
):
    plane = (rotation_cb @ corners.T).T + translation_cb
    x = plane[:, 0].reshape(2, 2)
    y = plane[:, 1].reshape(2, 2)
    z = plane[:, 2].reshape(2, 2)
    ax.plot_surface(x, y, z, color=facecolor, alpha=alpha, shade=False, linewidth=0.0)


def plot_fixed_camera_candidate_bank_3d(
    problem: infmat.CalibrationProblem,
    valid_mask: Sequence[bool],
    title: str,
    save_path: str | None = None,
):
    valid_mask = np.asarray(valid_mask, dtype=bool)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    board_corners = _checkerboard_corners(problem)
    translations = np.asarray(problem.candidate_board_translations, dtype=float)
    rotations = np.asarray(problem.candidate_board_rotations, dtype=float)

    ax.scatter([0.0], [0.0], [0.0], c="#111827", s=70, label="fixed camera")
    ax.scatter(
        translations[~valid_mask, 0],
        translations[~valid_mask, 1],
        translations[~valid_mask, 2],
        c="#cbd5e1",
        s=16,
        alpha=0.35,
        label="filtered out",
    )
    ax.scatter(
        translations[valid_mask, 0],
        translations[valid_mask, 1],
        translations[valid_mask, 2],
        c="#059669",
        s=26,
        alpha=0.8,
        label="kept",
    )

    sample_valid = np.flatnonzero(valid_mask)[: min(12, np.count_nonzero(valid_mask))]
    sample_invalid = np.flatnonzero(~valid_mask)[: min(12, np.count_nonzero(~valid_mask))]
    for idx in sample_valid:
        _plot_checkerboard_plane(ax, board_corners, rotations[idx], translations[idx], "#86efac", 0.16)
    for idx in sample_invalid:
        _plot_checkerboard_plane(ax, board_corners, rotations[idx], translations[idx], "#cbd5e1", 0.08)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 0.7))
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_camera_candidate_bank_3d(
    problem: infmat.CalibrationProblem,
    valid_mask: Sequence[bool],
    title: str,
    max_valid_axes: int | None = 300,
    max_invalid_axes: int | None = 120,
    save_path: str | None = None,
):
    valid_mask = np.asarray(valid_mask, dtype=bool)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    board = problem.target_points
    ax.scatter(board[:, 0], board[:, 1], board[:, 2], c="#111827", s=18, alpha=0.9, label="checkerboard")

    translations = np.asarray(problem.candidate_translations, dtype=float)
    rotations = np.asarray(problem.candidate_rotations, dtype=float)

    valid_indices = np.flatnonzero(valid_mask)
    invalid_indices = np.flatnonzero(~valid_mask)
    if max_valid_axes is not None and valid_indices.size > max_valid_axes:
        valid_step = max(1, valid_indices.size // max_valid_axes)
        valid_indices = valid_indices[::valid_step][:max_valid_axes]
    if max_invalid_axes is not None and invalid_indices.size > max_invalid_axes:
        invalid_step = max(1, invalid_indices.size // max_invalid_axes)
        invalid_indices = invalid_indices[::invalid_step][:max_invalid_axes]

    def draw_axes(indices: np.ndarray, alpha: float, axis_length: float):
        for idx in indices:
            origin = translations[int(idx)]
            rotation = rotations[int(idx)]
            x_axis = rotation[:, 0]
            y_axis = rotation[:, 1]
            z_axis = rotation[:, 2]
            ax.quiver(
                [origin[0]], [origin[1]], [origin[2]],
                [x_axis[0]], [x_axis[1]], [x_axis[2]],
                length=axis_length, normalize=True, color="#dc2626",
                alpha=alpha, linewidths=1.6, arrow_length_ratio=0.26,
            )
            ax.quiver(
                [origin[0]], [origin[1]], [origin[2]],
                [y_axis[0]], [y_axis[1]], [y_axis[2]],
                length=axis_length, normalize=True, color="#2563eb",
                alpha=alpha, linewidths=1.6, arrow_length_ratio=0.26,
            )
            ax.quiver(
                [origin[0]], [origin[1]], [origin[2]],
                [z_axis[0]], [z_axis[1]], [z_axis[2]],
                length=axis_length, normalize=True, color="#059669",
                alpha=alpha, linewidths=2.0, arrow_length_ratio=0.3,
            )

    if valid_indices.size > 0:
        draw_axes(valid_indices, alpha=0.95, axis_length=0.45)
        ax.scatter(
            translations[valid_indices, 0],
            translations[valid_indices, 1],
            translations[valid_indices, 2],
            c="#065f46",
            s=18,
            alpha=0.95,
            label="kept camera axes",
        )
    if invalid_indices.size > 0:
        draw_axes(invalid_indices, alpha=0.45, axis_length=0.32)
        ax.scatter(
            translations[invalid_indices, 0],
            translations[invalid_indices, 1],
            translations[invalid_indices, 2],
            c="#64748b",
            s=12,
            alpha=0.85,
            label="filtered camera axes",
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 0.7))
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_camera_candidate_bank_3d_plotly(
    problem: infmat.CalibrationProblem,
    valid_mask: Sequence[bool],
    title: str,
    max_valid_axes: int | None = 300,
    max_invalid_axes: int | None = 120,
    save_path: str | None = None,
):
    valid_mask = np.asarray(valid_mask, dtype=bool)
    board = np.asarray(problem.target_points, dtype=float)
    translations = np.asarray(problem.candidate_translations, dtype=float)
    rotations = np.asarray(problem.candidate_rotations, dtype=float)

    valid_indices = np.flatnonzero(valid_mask)
    invalid_indices = np.flatnonzero(~valid_mask)
    if max_valid_axes is not None and valid_indices.size > max_valid_axes:
        valid_step = max(1, valid_indices.size // max_valid_axes)
        valid_indices = valid_indices[::valid_step][:max_valid_axes]
    if max_invalid_axes is not None and invalid_indices.size > max_invalid_axes:
        invalid_step = max(1, invalid_indices.size // max_invalid_axes)
        invalid_indices = invalid_indices[::invalid_step][:max_invalid_axes]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=board[:, 0],
            y=board[:, 1],
            z=board[:, 2],
            mode="markers",
            name="checkerboard",
            marker=dict(size=4, color="#111827"),
            hovertemplate="board point<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        )
    )

    def add_axis_lines(indices: np.ndarray, axis_length: float, opacity: float, suffix: str):
        if indices.size == 0:
            return

        axis_specs = [
            ("camera x", 0, "#dc2626"),
            ("camera y", 1, "#2563eb"),
            ("camera z", 2, "#059669"),
        ]
        for axis_name, axis_idx, color in axis_specs:
            x_lines: list[float | None] = []
            y_lines: list[float | None] = []
            z_lines: list[float | None] = []
            for idx in indices:
                origin = translations[int(idx)]
                direction = rotations[int(idx)][:, axis_idx]
                end = origin + axis_length * direction
                x_lines.extend([float(origin[0]), float(end[0]), None])
                y_lines.extend([float(origin[1]), float(end[1]), None])
                z_lines.extend([float(origin[2]), float(end[2]), None])
            fig.add_trace(
                go.Scatter3d(
                    x=x_lines,
                    y=y_lines,
                    z=z_lines,
                    mode="lines",
                    name=f"{axis_name} ({suffix})",
                    line=dict(color=color, width=5),
                    opacity=opacity,
                    hoverinfo="skip",
                )
            )

        sampled = translations[indices]
        fig.add_trace(
            go.Scatter3d(
                x=sampled[:, 0],
                y=sampled[:, 1],
                z=sampled[:, 2],
                mode="markers",
                name=f"{suffix} camera origins",
                marker=dict(
                    size=3 if suffix == "filtered" else 4,
                    color="#64748b" if suffix == "filtered" else "#065f46",
                    opacity=opacity,
                ),
                customdata=np.asarray(indices, dtype=int)[:, None],
                hovertemplate=(
                    "candidate %{customdata[0]}<br>"
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
                ),
            )
        )

    add_axis_lines(valid_indices, axis_length=0.45, opacity=0.95, suffix="kept")
    add_axis_lines(invalid_indices, axis_length=0.32, opacity=0.35, suffix="filtered")

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(fig, str(path), auto_open=False, include_plotlyjs=True)
    return fig


def plot_fixed_camera_candidate_bank_image_plane(
    diagnostics: Sequence[Dict[str, object]],
    image_size: Sequence[int],
    valid_mask: Sequence[bool],
    title: str,
    save_path: str | None = None,
):
    valid_mask = np.asarray(valid_mask, dtype=bool)
    centers = np.asarray([item["projected_center_px"] for item in diagnostics], dtype=float)
    width, height = [float(v) for v in image_size]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        centers[~valid_mask, 0],
        centers[~valid_mask, 1],
        c="#cbd5e1",
        s=20,
        alpha=0.45,
        label="filtered out",
    )
    ax.scatter(
        centers[valid_mask, 0],
        centers[valid_mask, 1],
        c="#059669",
        s=28,
        alpha=0.85,
        label="kept",
    )
    for frac in (1.0 / 3.0, 2.0 / 3.0):
        ax.axvline(width * frac, color="#94a3b8", linewidth=1.0, alpha=0.6)
        ax.axhline(height * frac, color="#94a3b8", linewidth=1.0, alpha=0.6)

    ax.set_xlim(0.0, width)
    ax.set_ylim(height, 0.0)
    ax.set_title(title)
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.legend(loc="best")
    ax.grid(alpha=0.15, linewidth=0.5)
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_fixed_camera_projected_corner_bank(
    measurements: np.ndarray,
    valid_mask: Sequence[bool],
    image_size: Sequence[int],
    title: str,
    save_path: str | None = None,
):
    valid_mask = np.asarray(valid_mask, dtype=bool)
    width, height = [float(v) for v in image_size]
    measurements = np.asarray(measurements, dtype=float)

    rejected_points = measurements[~valid_mask].reshape(-1, 2)
    kept_points = measurements[valid_mask].reshape(-1, 2)
    rejected_points = rejected_points[np.isfinite(rejected_points).all(axis=1)]
    kept_points = kept_points[np.isfinite(kept_points).all(axis=1)]

    fig, ax = plt.subplots(figsize=(8, 6))
    if rejected_points.size > 0:
        ax.scatter(
            rejected_points[:, 0],
            rejected_points[:, 1],
            c="#cbd5e1",
            s=6,
            alpha=0.2,
            label="filtered out corners",
        )
    if kept_points.size > 0:
        ax.scatter(
            kept_points[:, 0],
            kept_points[:, 1],
            c="#059669",
            s=8,
            alpha=0.35,
            label="kept corners",
        )

    for frac in (1.0 / 3.0, 2.0 / 3.0):
        ax.axvline(width * frac, color="#94a3b8", linewidth=1.0, alpha=0.6)
        ax.axhline(height * frac, color="#94a3b8", linewidth=1.0, alpha=0.6)

    ax.set_xlim(0.0, width)
    ax.set_ylim(height, 0.0)
    ax.set_title(title)
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.legend(loc="best")
    ax.grid(alpha=0.15, linewidth=0.5)
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_pose_selection(
    problem: infmat.CalibrationProblem,
    selected_indices: Sequence[int],
    random_indices: Sequence[int] | None = None,
    save_path: str | None = None,
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if problem.camera_is_fixed and problem.candidate_board_translations is not None:
        camera_origin = np.zeros(3, dtype=float)
        ax.scatter([camera_origin[0]], [camera_origin[1]], [camera_origin[2]], c="#111827", s=60, label="fixed camera")
        board_corners = _checkerboard_corners(problem)

        all_t = problem.candidate_board_translations
        ax.scatter(all_t[:, 0], all_t[:, 1], all_t[:, 2], c="#cbd5e1", s=18, alpha=0.7, label="checkerboard candidates")

        selected_indices = list(selected_indices)
        if selected_indices:
            sel_t = problem.candidate_board_translations[selected_indices]
            sel_dirs = np.array([problem.candidate_board_rotations[idx][:, 2] for idx in selected_indices])
            ax.scatter(sel_t[:, 0], sel_t[:, 1], sel_t[:, 2], c="#059669", s=50, label="selected boards")
            ax.quiver(
                sel_t[:, 0],
                sel_t[:, 1],
                sel_t[:, 2],
                sel_dirs[:, 0],
                sel_dirs[:, 1],
                sel_dirs[:, 2],
                length=0.3,
                normalize=True,
                color="#059669",
            )
            for idx in selected_indices:
                _plot_checkerboard_plane(
                    ax,
                    board_corners,
                    problem.candidate_board_rotations[idx],
                    problem.candidate_board_translations[idx],
                    facecolor="#86efac",
                    alpha=0.18,
                )

        if random_indices:
            random_indices = list(random_indices)
            rnd_t = problem.candidate_board_translations[random_indices]
            rnd_dirs = np.array([problem.candidate_board_rotations[idx][:, 2] for idx in random_indices])
            ax.scatter(rnd_t[:, 0], rnd_t[:, 1], rnd_t[:, 2], c="#d97706", s=45, label="random best boards")
            ax.quiver(
                rnd_t[:, 0],
                rnd_t[:, 1],
                rnd_t[:, 2],
                rnd_dirs[:, 0],
                rnd_dirs[:, 1],
                rnd_dirs[:, 2],
                length=0.3,
                normalize=True,
                color="#d97706",
            )
            for idx in random_indices:
                _plot_checkerboard_plane(
                    ax,
                    board_corners,
                    problem.candidate_board_rotations[idx],
                    problem.candidate_board_translations[idx],
                    facecolor="#fdba74",
                    alpha=0.14,
                )

        ax.set_title(
            f"Candidate Checkerboard Poses With Fixed Camera "
            f"(sampled={problem.num_candidates}, selected={len(selected_indices)})"
        )
    else:
        board = problem.target_points
        ax.scatter(board[:, 0], board[:, 1], board[:, 2], c="#1f2937", s=20, label="checkerboard")

        all_t = problem.candidate_translations
        ax.scatter(all_t[:, 0], all_t[:, 1], all_t[:, 2], c="#cbd5e1", s=18, alpha=0.7, label="candidates")

        selected_indices = list(selected_indices)
        if selected_indices:
            sel_t = problem.candidate_translations[selected_indices]
            sel_dirs = np.array([_camera_forward(problem.candidate_rotations[idx]) for idx in selected_indices])
            ax.scatter(sel_t[:, 0], sel_t[:, 1], sel_t[:, 2], c="#059669", s=50, label="selected")
            ax.quiver(sel_t[:, 0], sel_t[:, 1], sel_t[:, 2], sel_dirs[:, 0], sel_dirs[:, 1], sel_dirs[:, 2],
                      length=0.12, normalize=True, color="#059669")

        if random_indices:
            random_indices = list(random_indices)
            rnd_t = problem.candidate_translations[random_indices]
            rnd_dirs = np.array([_camera_forward(problem.candidate_rotations[idx]) for idx in random_indices])
            ax.scatter(rnd_t[:, 0], rnd_t[:, 1], rnd_t[:, 2], c="#d97706", s=45, label="random best")
            ax.quiver(rnd_t[:, 0], rnd_t[:, 1], rnd_t[:, 2], rnd_dirs[:, 0], rnd_dirs[:, 1], rnd_dirs[:, 2],
                      length=0.12, normalize=True, color="#d97706")

        ax.set_title(
            f"Candidate Poses And Selected Calibration Views "
            f"(sampled={problem.num_candidates}, selected={len(selected_indices)})"
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left")
    ax.set_box_aspect((1, 1, 0.7))
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_selection_report(report: Dict[str, object], save_path: str | None = None):
    fig, axes = plt.subplots(1, len(REPORT_KEYS), figsize=(16, 4))
    selected = report["selected"]
    random_avg = report["random_average"]

    for ax, key in zip(axes, REPORT_KEYS):
        values = [selected[key], random_avg[key]]
        colors = ["#059669", "#d97706"]
        ax.bar(["selected", "random avg"], values, color=colors)
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=20)
        if key in {"trace_cov", "cond"}:
            ax.set_yscale("log")

    fig.suptitle("Selected Pose Set Versus Random Pose Baseline")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_candidate_eigenvalues(
    scores: Sequence[float],
    selected_indices: Sequence[int] | None = None,
    random_indices: Sequence[int] | None = None,
    save_path: str | None = None,
):
    scores = np.asarray(scores, dtype=float)
    x = np.arange(scores.size)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, scores, color="#0f172a", linewidth=1.4, alpha=0.9, label="candidate min eig")
    ax.scatter(x, scores, color="#94a3b8", s=12, alpha=0.65)

    if selected_indices:
        selected_indices = np.asarray(list(selected_indices), dtype=int)
        ax.scatter(
            selected_indices,
            scores[selected_indices],
            color="#059669",
            s=50,
            zorder=3,
            label="selected",
        )

    if random_indices:
        random_indices = np.asarray(list(random_indices), dtype=int)
        ax.scatter(
            random_indices,
            scores[random_indices],
            color="#d97706",
            s=42,
            zorder=3,
            label="random best",
        )

    ax.set_title("Minimum Eigenvalue Score For Each Candidate Pose")
    ax.set_xlabel("candidate pose index")
    ax.set_ylabel("min eig(H_cal)")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_candidate_eigenvalues_3d(
    problem: infmat.CalibrationProblem,
    scores: Sequence[float],
    selected_indices: Sequence[int] | None = None,
    random_indices: Sequence[int] | None = None,
    save_path: str | None = None,
):
    scores = np.asarray(scores, dtype=float)
    translations = (
        problem.candidate_board_translations
        if problem.camera_is_fixed and problem.candidate_board_translations is not None
        else problem.candidate_translations
    )

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        c=scores,
        cmap="viridis",
        s=28,
        alpha=0.85,
    )

    if selected_indices is not None and len(selected_indices) > 0:
        selected_indices = np.asarray(list(selected_indices), dtype=int)
        ax.scatter(
            translations[selected_indices, 0],
            translations[selected_indices, 1],
            translations[selected_indices, 2],
            facecolors="none",
            edgecolors="#ef4444",
            linewidths=1.8,
            s=120,
            label="selected",
        )

    if random_indices is not None and len(random_indices) > 0:
        random_indices = np.asarray(list(random_indices), dtype=int)
        ax.scatter(
            translations[random_indices, 0],
            translations[random_indices, 1],
            translations[random_indices, 2],
            facecolors="none",
            edgecolors="#f59e0b",
            linewidths=1.4,
            s=95,
            label="random best",
        )

    if problem.camera_is_fixed:
        ax.scatter([0.0], [0.0], [0.0], c="#111827", s=36, alpha=0.95, label="fixed camera")
        board_corners = _checkerboard_corners(problem)
        if selected_indices is not None and len(selected_indices) > 0:
            for idx in selected_indices:
                _plot_checkerboard_plane(
                    ax,
                    board_corners,
                    problem.candidate_board_rotations[int(idx)],
                    problem.candidate_board_translations[int(idx)],
                    facecolor="#86efac",
                    alpha=0.16,
                )
        if random_indices is not None and len(random_indices) > 0:
            for idx in random_indices:
                _plot_checkerboard_plane(
                    ax,
                    board_corners,
                    problem.candidate_board_rotations[int(idx)],
                    problem.candidate_board_translations[int(idx)],
                    facecolor="#fdba74",
                    alpha=0.12,
                )
    else:
        board = problem.target_points
        ax.scatter(board[:, 0], board[:, 1], board[:, 2], c="#111827", s=16, alpha=0.9, label="checkerboard")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.08, shrink=0.8)
    cbar.set_label("single-pose min eig(H_cal)")

    if problem.camera_is_fixed:
        ax.set_title("3D Candidate Checkerboard Eigenvalue Map")
    else:
        ax.set_title("3D Candidate Pose Eigenvalue Map")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 0.7))
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_candidate_eigenvalue_spectra(
    spectra: np.ndarray,
    selected_indices: Sequence[int] | None = None,
    random_indices: Sequence[int] | None = None,
    save_path: str | None = None,
):
    spectra = np.asarray(spectra, dtype=float)
    x = np.arange(spectra.shape[0])

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#0f172a", "#2563eb", "#0891b2", "#7c3aed"]
    for eig_idx in range(spectra.shape[1]):
        ax.plot(
            x,
            spectra[:, eig_idx],
            color=colors[eig_idx % len(colors)],
            linewidth=1.4,
            alpha=0.9,
            label=f"eig {eig_idx + 1}",
        )

    if selected_indices:
        selected_indices = np.asarray(list(selected_indices), dtype=int)
        for eig_idx in range(spectra.shape[1]):
            ax.scatter(
                selected_indices,
                spectra[selected_indices, eig_idx],
                color="#059669",
                s=28,
                alpha=0.75,
                zorder=3,
            )

    if random_indices:
        random_indices = np.asarray(list(random_indices), dtype=int)
        for eig_idx in range(spectra.shape[1]):
            ax.scatter(
                random_indices,
                spectra[random_indices, eig_idx],
                color="#d97706",
                s=24,
                alpha=0.65,
                zorder=3,
            )

    ax.set_title("Eigenvalue Spectrum For All Candidate Poses")
    ax.set_xlabel("candidate pose index")
    ax.set_ylabel("eigenvalue")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best", ncol=min(4, spectra.shape[1]))
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_parameter_uncertainty_before_after(
    parameter_labels: Sequence[str],
    before_std: Sequence[float],
    after_std: Sequence[float],
    save_path: str | None = None,
):
    x = np.arange(len(parameter_labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, before_std, width=width, color="#94a3b8", label="before")
    ax.bar(x + width / 2, after_std, width=width, color="#059669", label="after")
    ax.set_title("Parameter Uncertainty Before And After Calibration")
    ax.set_xlabel("intrinsic parameter")
    ax.set_ylabel("std dev")
    ax.set_xticks(x, parameter_labels)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_min_eigenvalue_before_after(
    before_min_eig: float,
    after_min_eig: float,
    save_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(["before", "after"], [before_min_eig, after_min_eig], marker="o", color="#0f172a", linewidth=2.0)
    ax.scatter(["before"], [before_min_eig], color="#94a3b8", s=60, zorder=3)
    ax.scatter(["after"], [after_min_eig], color="#059669", s=60, zorder=3)
    ax.set_title("Minimum Eigenvalue Before And After Calibration")
    ax.set_ylabel("min eig(H_cal)")
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_eigenvalues_before_after(
    before_eigvals: Sequence[float],
    after_eigvals: Sequence[float],
    save_path: str | None = None,
):
    before_eigvals = np.asarray(before_eigvals, dtype=float)
    after_eigvals = np.asarray(after_eigvals, dtype=float)
    x = np.arange(before_eigvals.size)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, before_eigvals, marker="o", color="#94a3b8", linewidth=1.8, label="before")
    ax.plot(x, after_eigvals, marker="o", color="#059669", linewidth=1.8, label="after")
    ax.set_title("Calibration Eigenvalues Before And After")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue")
    ax.set_xticks(x, [f"eig {idx + 1}" for idx in x])
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_spacing_sweep_metric(
    spacings: Sequence[float],
    selected_values: Sequence[float],
    random_values: Sequence[float] | None = None,
    ylabel: str = "metric",
    title: str = "Checkerboard Spacing Sweep",
    save_path: str | None = None,
):
    spacings = np.asarray(spacings, dtype=float)
    selected_values = np.asarray(selected_values, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(spacings, selected_values, marker="o", linewidth=2.0, color="#059669", label="selected")

    if random_values is not None:
        random_values = np.asarray(random_values, dtype=float)
        ax.plot(spacings, random_values, marker="o", linewidth=1.8, color="#d97706", label="random avg")

    ax.set_title(title)
    ax.set_xlabel("checkerboard square size")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_parameter_uncertainty_vs_spacing(
    spacings: Sequence[float],
    parameter_labels: Sequence[str],
    uncertainty_matrix: np.ndarray,
    save_path: str | None = None,
):
    spacings = np.asarray(spacings, dtype=float)
    uncertainty_matrix = np.asarray(uncertainty_matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#0f172a", "#2563eb", "#0891b2", "#7c3aed"]
    for idx, label in enumerate(parameter_labels):
        ax.plot(
            spacings,
            uncertainty_matrix[:, idx],
            marker="o",
            linewidth=2.0,
            color=colors[idx % len(colors)],
            label=label,
        )

    ax.set_title("Intrinsic Parameter Uncertainty Vs Checkerboard Spacing")
    ax.set_xlabel("checkerboard square size")
    ax.set_ylabel("std dev after calibration")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_parameter_uncertainty_vs_noise(
    noise_levels: Sequence[float],
    parameter_labels: Sequence[str],
    uncertainty_matrix: np.ndarray,
    save_path: str | None = None,
):
    noise_levels = np.asarray(noise_levels, dtype=float)
    uncertainty_matrix = np.asarray(uncertainty_matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#0f172a", "#2563eb", "#0891b2", "#7c3aed"]
    for idx, label in enumerate(parameter_labels):
        ax.plot(
            noise_levels,
            uncertainty_matrix[:, idx],
            marker="o",
            linewidth=2.0,
            color=colors[idx % len(colors)],
            label=label,
        )

    ax.set_title("Intrinsic Parameter Uncertainty Vs Pixel Noise")
    ax.set_xlabel("pixel noise sigma")
    ax.set_ylabel("std dev after calibration")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_noise_sweep_metric(
    noise_levels: Sequence[float],
    selected_values: Sequence[float],
    random_values: Sequence[float] | None = None,
    ylabel: str = "metric",
    title: str = "Pixel Noise Sweep",
    save_path: str | None = None,
):
    noise_levels = np.asarray(noise_levels, dtype=float)
    selected_values = np.asarray(selected_values, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(noise_levels, selected_values, marker="o", linewidth=2.0, color="#059669", label="selected")

    if random_values is not None:
        random_values = np.asarray(random_values, dtype=float)
        ax.plot(noise_levels, random_values, marker="o", linewidth=1.8, color="#d97706", label="random avg")

    ax.set_title(title)
    ax.set_xlabel("pixel noise sigma")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_eigenvalues_vs_noise(
    noise_levels: Sequence[float],
    eigenvalue_matrix: np.ndarray,
    save_path: str | None = None,
):
    noise_levels = np.asarray(noise_levels, dtype=float)
    eigenvalue_matrix = np.asarray(eigenvalue_matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#0f172a", "#2563eb", "#0891b2", "#7c3aed"]
    for idx in range(eigenvalue_matrix.shape[1]):
        ax.plot(
            noise_levels,
            eigenvalue_matrix[:, idx],
            marker="o",
            linewidth=2.0,
            color=colors[idx % len(colors)],
            label=f"eig {idx + 1}",
        )

    ax.set_title("Calibration Eigenvalues Vs Pixel Noise")
    ax.set_xlabel("pixel noise sigma")
    ax.set_ylabel("eigenvalue after calibration")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_azimuth_span_metric(
    azimuth_spans_deg: Sequence[float],
    selected_values: Sequence[float],
    random_values: Sequence[float] | None = None,
    ylabel: str = "metric",
    title: str = "Azimuth Span Sweep",
    save_path: str | None = None,
):
    azimuth_spans_deg = np.asarray(azimuth_spans_deg, dtype=float)
    selected_values = np.asarray(selected_values, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(azimuth_spans_deg, selected_values, marker="o", linewidth=2.0, color="#059669", label="selected")

    if random_values is not None:
        random_values = np.asarray(random_values, dtype=float)
        ax.plot(azimuth_spans_deg, random_values, marker="o", linewidth=1.8, color="#d97706", label="random avg")

    ax.set_title(title)
    ax.set_xlabel("candidate azimuth span (deg)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_parameter_uncertainty_vs_azimuth_span(
    azimuth_spans_deg: Sequence[float],
    parameter_labels: Sequence[str],
    uncertainty_matrix: np.ndarray,
    save_path: str | None = None,
):
    azimuth_spans_deg = np.asarray(azimuth_spans_deg, dtype=float)
    uncertainty_matrix = np.asarray(uncertainty_matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#0f172a", "#2563eb", "#0891b2", "#7c3aed"]
    for idx, label in enumerate(parameter_labels):
        ax.plot(
            azimuth_spans_deg,
            uncertainty_matrix[:, idx],
            marker="o",
            linewidth=2.0,
            color=colors[idx % len(colors)],
            label=label,
        )

    ax.set_title("Intrinsic Parameter Uncertainty Vs Candidate Azimuth Span")
    ax.set_xlabel("candidate azimuth span (deg)")
    ax.set_ylabel("std dev after calibration")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_eigenvalues_vs_azimuth_span(
    azimuth_spans_deg: Sequence[float],
    eigenvalue_matrix: np.ndarray,
    save_path: str | None = None,
):
    azimuth_spans_deg = np.asarray(azimuth_spans_deg, dtype=float)
    eigenvalue_matrix = np.asarray(eigenvalue_matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#0f172a", "#2563eb", "#0891b2", "#7c3aed"]
    for idx in range(eigenvalue_matrix.shape[1]):
        ax.plot(
            azimuth_spans_deg,
            eigenvalue_matrix[:, idx],
            marker="o",
            linewidth=2.0,
            color=colors[idx % len(colors)],
            label=f"eig {idx + 1}",
        )

    ax.set_title("Calibration Eigenvalues Vs Candidate Azimuth Span")
    ax.set_xlabel("candidate azimuth span (deg)")
    ax.set_ylabel("eigenvalue after calibration")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_metric_heatmap(
    x_values: Sequence[float],
    y_values: Sequence[float],
    matrix: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    colorbar_label: str,
    save_path: str | None = None,
):
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(x_values)), [str(v) for v in x_values])
    ax.set_yticks(np.arange(len(y_values)), [str(v) for v in y_values])
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(colorbar_label)
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_metric_curves_by_series(
    x_values: Sequence[float],
    series_to_values: Dict[str, Sequence[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str | None = None,
):
    x_values = np.asarray(x_values, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#0f172a", "#2563eb", "#0891b2", "#7c3aed", "#059669", "#d97706", "#dc2626", "#7c2d12"]

    for idx, (label, values) in enumerate(series_to_values.items()):
        ax.plot(
            x_values,
            np.asarray(values, dtype=float),
            marker="o",
            linewidth=2.0,
            color=colors[idx % len(colors)],
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_scalar_sweep_metric(
    x_values: Sequence[float],
    selected_values: Sequence[float],
    random_values: Sequence[float] | None = None,
    xlabel: str = "sweep value",
    ylabel: str = "metric",
    title: str = "Metric Sweep",
    save_path: str | None = None,
):
    x_values = np.asarray(x_values, dtype=float)
    selected_values = np.asarray(selected_values, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_values, selected_values, marker="o", linewidth=2.0, color="#059669", label="selected")
    if random_values is not None:
        random_values = np.asarray(random_values, dtype=float)
        ax.plot(x_values, random_values, marker="o", linewidth=1.8, color="#d97706", label="random avg")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_parameter_uncertainty_vs_scalar(
    x_values: Sequence[float],
    parameter_labels: Sequence[str],
    uncertainty_matrix: np.ndarray,
    xlabel: str,
    title: str,
    save_path: str | None = None,
):
    x_values = np.asarray(x_values, dtype=float)
    uncertainty_matrix = np.asarray(uncertainty_matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#0f172a", "#2563eb", "#0891b2", "#7c3aed"]
    for idx, label in enumerate(parameter_labels):
        ax.plot(
            x_values,
            uncertainty_matrix[:, idx],
            marker="o",
            linewidth=2.0,
            color=colors[idx % len(colors)],
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("std dev after calibration")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_eigenvalues_vs_scalar(
    x_values: Sequence[float],
    eigenvalue_matrix: np.ndarray,
    xlabel: str,
    title: str,
    save_path: str | None = None,
):
    x_values = np.asarray(x_values, dtype=float)
    eigenvalue_matrix = np.asarray(eigenvalue_matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#0f172a", "#2563eb", "#0891b2", "#7c3aed"]
    for idx in range(eigenvalue_matrix.shape[1]):
        ax.plot(
            x_values,
            eigenvalue_matrix[:, idx],
            marker="o",
            linewidth=2.0,
            color=colors[idx % len(colors)],
            label=f"eig {idx + 1}",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("eigenvalue after calibration")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig
