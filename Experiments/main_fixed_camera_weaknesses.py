import argparse
import json
import pathlib
import sys
from datetime import datetime
from typing import Callable, Dict, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from optimal_camera_placement.DataGenerator import sim_data_utils as sdu
from optimal_camera_placement.OASIS import FIM as fim
from optimal_camera_placement.OASIS import calibration_analysis as cal_analysis
from optimal_camera_placement.OASIS import calibration_visualize as cal_viz
from optimal_camera_placement.OASIS import methods


def create_run_output_dir(base_dir: pathlib.Path, run_name: str) -> pathlib.Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"{run_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def average_metric_reports(reports: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = reports[0].keys()
    return {key: float(np.mean([report[key] for report in reports])) for key in keys}


def average_before_after(problem_summaries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    labels = problem_summaries[0]["parameter_labels"]
    before_std = np.mean([summary["before"]["std_dev"] for summary in problem_summaries], axis=0)
    after_std = np.mean([summary["after"]["std_dev"] for summary in problem_summaries], axis=0)
    before_eigvals = np.mean([summary["before"]["eigvals"] for summary in problem_summaries], axis=0)
    after_eigvals = np.mean([summary["after"]["eigvals"] for summary in problem_summaries], axis=0)
    return {
        "parameter_labels": labels,
        "before": {
            "std_dev": np.asarray(before_std, dtype=float).tolist(),
            "eigvals": np.asarray(before_eigvals, dtype=float).tolist(),
            "min_eig": float(np.min(before_eigvals)),
        },
        "after": {
            "std_dev": np.asarray(after_std, dtype=float).tolist(),
            "eigvals": np.asarray(after_eigvals, dtype=float).tolist(),
            "min_eig": float(np.min(after_eigvals)),
        },
    }


def build_report_interpretation() -> dict:
    return {
        "primary_objective": {
            "name": "min_eig",
            "description": "Minimum eigenvalue of the calibration information matrix H_cal.",
            "desired_direction": "higher",
        },
        "metrics": {
            "min_eig": {"desired_direction": "higher"},
            "logdet": {"desired_direction": "higher"},
            "trace_cov": {"desired_direction": "lower"},
            "cond": {"desired_direction": "lower"},
            "visible_points": {"desired_direction": "higher_usually"},
        },
        "experiment_question": (
            "How common perception-side weaknesses affect fixed-camera calibration when the checkerboard moves in space."
        ),
    }


def mode_definitions() -> Dict[str, dict]:
    mean_radius = 5.0

    return {
        "limited_viewpoint_diversity": {
            "values": [1, 2, 4, 8, 16],
            "xlabel": "azimuth anchor count",
            "description": "Few unique viewing directions make the measurements redundant.",
            "builder": lambda value: {
                "azimuth_anchor_count": int(value),
                "azimuth_span_deg": 360.0,
                "radius_range": (4.0, 6.0),
                "offaxis_angle_range_deg": (4.0, 25.0),
                "tilt_span_deg": 30.0,
                "symmetry_strength": 0.0,
            },
        },
        "narrow_azimuth_coverage": {
            "values": [30.0, 60.0, 90.0, 120.0, 180.0, 360.0],
            "xlabel": "azimuth span (deg)",
            "description": "Seeing the board only from one side weakens some intrinsic modes.",
            "builder": lambda value: {
                "azimuth_span_deg": float(value),
                "azimuth_anchor_count": None,
                "radius_range": (4.0, 6.0),
                "offaxis_angle_range_deg": (4.0, 25.0),
                "tilt_span_deg": 30.0,
                "symmetry_strength": 0.0,
            },
        },
        "little_depth_variation": {
            "values": [0.0, 0.5, 1.0, 1.5, 2.0],
            "xlabel": "distance span (m)",
            "description": "When the board stays at almost one distance, scale sensitivity is weaker.",
            "builder": lambda value: {
                "azimuth_span_deg": 360.0,
                "radius_range": (mean_radius - 0.5 * float(value), mean_radius + 0.5 * float(value)),
                "offaxis_angle_range_deg": (4.0, 25.0),
                "tilt_span_deg": 30.0,
                "symmetry_strength": 0.0,
            },
        },
        "little_tilt_variation": {
            "values": [0.0, 5.0, 10.0, 20.0, 35.0, 50.0],
            "xlabel": "tilt span (deg)",
            "description": "Fronto-parallel board poses are less informative than slanted ones.",
            "builder": lambda value: {
                "azimuth_span_deg": 360.0,
                "radius_range": (4.0, 6.0),
                "offaxis_angle_range_deg": (4.0, 25.0),
                "tilt_span_deg": float(value),
                "symmetry_strength": 0.0,
            },
        },
        "poor_image_plane_coverage": {
            "values": [2.0, 5.0, 10.0, 15.0, 20.0, 30.0],
            "xlabel": "max off-axis angle (deg)",
            "description": "If corners stay near the image center, principal point and focal length separate poorly.",
            "builder": lambda value: {
                "azimuth_span_deg": 360.0,
                "radius_range": (4.0, 6.0),
                "offaxis_angle_range_deg": (0.0, float(value)),
                "tilt_span_deg": 30.0,
                "symmetry_strength": 0.0,
            },
        },
        "small_target_footprint": {
            "values": [0.05, 0.1, 0.2, 0.3, 0.4],
            "xlabel": "checkerboard square size",
            "description": "A physically smaller board creates weaker pixel motion.",
            "builder": lambda value: {
                "azimuth_span_deg": 360.0,
                "radius_range": (4.0, 6.0),
                "offaxis_angle_range_deg": (4.0, 25.0),
                "tilt_span_deg": 30.0,
                "board_square_size": float(value),
                "symmetry_strength": 0.0,
            },
        },
        "too_few_visible_points": {
            "values": [3, 5, 7, 9],
            "xlabel": "board size (rows = cols)",
            "description": "Fewer checkerboard corners mean fewer measurements.",
            "builder": lambda value: {
                "azimuth_span_deg": 360.0,
                "radius_range": (4.0, 6.0),
                "offaxis_angle_range_deg": (4.0, 25.0),
                "tilt_span_deg": 30.0,
                "board_rows": int(value),
                "board_cols": int(value),
                "symmetry_strength": 0.0,
            },
        },
        "symmetric_geometry": {
            "values": [1.0, 0.75, 0.5, 0.25, 0.0],
            "xlabel": "symmetry strength",
            "description": "Highly mirrored board motions can couple parameters and hide weak directions.",
            "builder": lambda value: {
                "azimuth_span_deg": 180.0,
                "radius_range": (4.5, 5.5),
                "offaxis_angle_range_deg": (2.0, 15.0),
                "tilt_span_deg": 15.0,
                "symmetry_strength": float(value),
                "azimuth_anchor_count": 8,
            },
        },
        "planar_pose_degeneracy": {
            "values": [0.0, 0.25, 0.5, 0.75, 1.0],
            "xlabel": "pose richness",
            "description": "A planar board becomes weak quickly when overall pose diversity collapses.",
            "builder": lambda value: {
                "azimuth_span_deg": 45.0 + 315.0 * float(value),
                "radius_range": (mean_radius - 0.5 * (0.2 + 1.8 * float(value)), mean_radius + 0.5 * (0.2 + 1.8 * float(value))),
                "offaxis_angle_range_deg": (0.0, 3.0 + 27.0 * float(value)),
                "tilt_span_deg": 2.0 + 43.0 * float(value),
                "azimuth_anchor_count": max(1, int(round(1 + 15 * float(value)))),
                "symmetry_strength": 0.0,
            },
        },
        "high_pixel_noise": {
            "values": [0.1, 0.5, 1.0, 2.0, 3.0, 5.0],
            "xlabel": "pixel noise sigma",
            "description": "Higher corner noise lowers Fisher information and increases uncertainty.",
            "builder": lambda value: {
                "azimuth_span_deg": 360.0,
                "radius_range": (4.0, 6.0),
                "offaxis_angle_range_deg": (4.0, 25.0),
                "tilt_span_deg": 30.0,
                "pixel_noise_sigma": float(value),
                "symmetry_strength": 0.0,
            },
        },
    }


def run_sweep(
    mode_name: str,
    mode_spec: dict,
    args: argparse.Namespace,
    out_dir: pathlib.Path,
) -> dict:
    values = mode_spec["values"]
    xlabel = mode_spec["xlabel"]
    builder: Callable[[float], dict] = mode_spec["builder"]
    mode_dir = out_dir / mode_name
    mode_dir.mkdir(parents=True, exist_ok=True)

    sweep_results = []
    parameter_labels = None
    uncertainty_curves = []
    eigenvalue_curves = []
    representative_artifacts = []
    representative_ids = {0, len(values) - 1}

    print(f"Running mode: {mode_name}")
    print(f"  {mode_spec['description']}")

    for idx, value in enumerate(values):
        print(f"  value {value} ({idx + 1}/{len(values)})")
        config = {
            "num_candidate_poses": args.num_candidate_poses,
            "radius_range": (args.min_radius, args.max_radius),
            "azimuth_span_deg": 360.0,
            "azimuth_center_deg": args.azimuth_center_deg,
            "offaxis_angle_range_deg": (args.min_offaxis_deg, args.max_offaxis_deg),
            "tilt_span_deg": args.tilt_span_deg,
            "azimuth_anchor_count": None,
            "symmetry_strength": 0.0,
            "board_rows": args.board_rows,
            "board_cols": args.board_cols,
            "board_square_size": args.board_square_size,
            "pixel_noise_sigma": args.pixel_noise,
        }
        config.update(builder(value))

        problems = sdu.generate_fixed_camera_problem_set(
            num_runs=args.num_runs,
            image_size=(args.width, args.height),
            pixel_noise_sigma=config["pixel_noise_sigma"],
            num_candidate_poses=config["num_candidate_poses"],
            radius_range=config["radius_range"],
            azimuth_span_deg=config["azimuth_span_deg"],
            azimuth_center_deg=config["azimuth_center_deg"],
            offaxis_angle_range_deg=config["offaxis_angle_range_deg"],
            tilt_span_deg=config["tilt_span_deg"],
            azimuth_anchor_count=config["azimuth_anchor_count"],
            symmetry_strength=config["symmetry_strength"],
            board_rows=config["board_rows"],
            board_cols=config["board_cols"],
            board_square_size=config["board_square_size"],
        )
        priors = [fim.build_prior_blocks(problem) for problem in problems]
        best_score, selected_poses, selected_indices, elapsed, selection = methods.run_single_experiment_exp(
            problems,
            select_k=args.select_k,
            priors=priors,
        )
        selected_reports = [
            cal_analysis.evaluate_selection(problem, selected_indices, prior=prior)
            for problem, prior in zip(problems, priors)
        ]
        random_reports = [
            cal_analysis.random_baseline_report(problem, args.select_k, prior=prior, num_trials=20, seed=11)["average"]
            for problem, prior in zip(problems, priors)
        ]
        selected_avg = average_metric_reports(selected_reports)
        random_avg = average_metric_reports(random_reports)
        before_after_per_problem = [
            cal_analysis.before_after_calibration_summary(problem, selected_indices, prior=prior)
            for problem, prior in zip(problems, priors)
        ]
        before_after = average_before_after(before_after_per_problem)
        if parameter_labels is None:
            parameter_labels = before_after["parameter_labels"]
        uncertainty_curves.append(before_after["after"]["std_dev"])
        eigenvalue_curves.append(before_after["after"]["eigvals"])

        candidate_scores = np.mean(
            [cal_analysis.candidate_min_eig_scores(problem, prior=prior) for problem, prior in zip(problems, priors)],
            axis=0,
        )
        report = cal_analysis.compare_selected_vs_random(
            problems[0],
            selected_indices,
            prior=priors[0],
            num_random_trials=20,
            seed=11,
        )
        if idx in representative_ids:
            tag = "weak" if idx == 0 else "strong"
            pose_path = mode_dir / f"{mode_name}_{tag}_pose_selection.png"
            eig3d_path = mode_dir / f"{mode_name}_{tag}_candidate_eigenvalues_3d.png"
            cal_viz.plot_pose_selection(problems[0], selected_indices, report["random_best_indices"], save_path=str(pose_path))
            cal_viz.plot_candidate_eigenvalues_3d(
                problems[0],
                candidate_scores,
                selected_indices,
                report["random_best_indices"],
                save_path=str(eig3d_path),
            )
            representative_artifacts.append(
                {
                    "value": value,
                    "tag": tag,
                    "pose_selection_plot": str(pose_path),
                    "candidate_eigenvalue_3d_plot": str(eig3d_path),
                }
            )
            plt.close("all")

        sweep_results.append(
            {
                "value": value,
                "config": {
                    key: (list(val) if isinstance(val, tuple) else val)
                    for key, val in config.items()
                },
                "selected_indices": selected_indices,
                "best_score": best_score,
                "runtime_seconds": elapsed,
                "selection_vector": selection.astype(int).tolist(),
                "selected_report": selected_avg,
                "random_average_report": random_avg,
                "random_best_indices": report["random_best_indices"],
                "before_after_calibration": before_after,
            }
        )

    x_values = np.asarray(values, dtype=float)
    uncertainty_curves = np.asarray(uncertainty_curves, dtype=float)
    eigenvalue_curves = np.asarray(eigenvalue_curves, dtype=float)
    selected_min_eig = [item["selected_report"]["min_eig"] for item in sweep_results]
    random_min_eig = [item["random_average_report"]["min_eig"] for item in sweep_results]
    selected_trace_cov = [item["selected_report"]["trace_cov"] for item in sweep_results]
    random_trace_cov = [item["random_average_report"]["trace_cov"] for item in sweep_results]
    selected_cond = [item["selected_report"]["cond"] for item in sweep_results]
    random_cond = [item["random_average_report"]["cond"] for item in sweep_results]
    selected_logdet = [item["selected_report"]["logdet"] for item in sweep_results]
    random_logdet = [item["random_average_report"]["logdet"] for item in sweep_results]
    selected_visible_points = [item["selected_report"]["visible_points"] for item in sweep_results]
    random_visible_points = [item["random_average_report"]["visible_points"] for item in sweep_results]

    plot_paths = {
        "min_eig_plot": mode_dir / f"{mode_name}_min_eig.png",
        "trace_cov_plot": mode_dir / f"{mode_name}_trace_cov.png",
        "condition_number_plot": mode_dir / f"{mode_name}_condition_number.png",
        "logdet_plot": mode_dir / f"{mode_name}_logdet.png",
        "visible_points_plot": mode_dir / f"{mode_name}_visible_points.png",
        "parameter_uncertainty_plot": mode_dir / f"{mode_name}_parameter_uncertainty.png",
        "eigenvalues_plot": mode_dir / f"{mode_name}_eigenvalues.png",
    }
    cal_viz.plot_scalar_sweep_metric(
        x_values,
        selected_min_eig,
        random_min_eig,
        xlabel=xlabel,
        ylabel="min eig(H_cal)",
        title=f"{mode_name.replace('_', ' ').title()} Vs Minimum Eigenvalue",
        save_path=str(plot_paths["min_eig_plot"]),
    )
    cal_viz.plot_scalar_sweep_metric(
        x_values,
        selected_trace_cov,
        random_trace_cov,
        xlabel=xlabel,
        ylabel="trace(covariance)",
        title=f"{mode_name.replace('_', ' ').title()} Vs Total Uncertainty",
        save_path=str(plot_paths["trace_cov_plot"]),
    )
    cal_viz.plot_scalar_sweep_metric(
        x_values,
        selected_cond,
        random_cond,
        xlabel=xlabel,
        ylabel="condition number",
        title=f"{mode_name.replace('_', ' ').title()} Vs Conditioning",
        save_path=str(plot_paths["condition_number_plot"]),
    )
    cal_viz.plot_scalar_sweep_metric(
        x_values,
        selected_logdet,
        random_logdet,
        xlabel=xlabel,
        ylabel="logdet(H_cal)",
        title=f"{mode_name.replace('_', ' ').title()} Vs Log Determinant",
        save_path=str(plot_paths["logdet_plot"]),
    )
    cal_viz.plot_scalar_sweep_metric(
        x_values,
        selected_visible_points,
        random_visible_points,
        xlabel=xlabel,
        ylabel="visible points",
        title=f"{mode_name.replace('_', ' ').title()} Vs Visible Points",
        save_path=str(plot_paths["visible_points_plot"]),
    )
    cal_viz.plot_parameter_uncertainty_vs_scalar(
        x_values,
        parameter_labels or ["fx", "fy", "cx", "cy"],
        uncertainty_curves,
        xlabel=xlabel,
        title=f"{mode_name.replace('_', ' ').title()} Vs Intrinsic Uncertainty",
        save_path=str(plot_paths["parameter_uncertainty_plot"]),
    )
    cal_viz.plot_eigenvalues_vs_scalar(
        x_values,
        eigenvalue_curves,
        xlabel=xlabel,
        title=f"{mode_name.replace('_', ' ').title()} Vs Calibration Eigenvalues",
        save_path=str(plot_paths["eigenvalues_plot"]),
    )
    plt.close("all")

    return {
        "mode_name": mode_name,
        "description": mode_spec["description"],
        "xlabel": xlabel,
        "values": values,
        "sweep_results": sweep_results,
        "parameter_labels": parameter_labels,
        "selected_parameter_uncertainty_curves": uncertainty_curves.tolist(),
        "selected_eigenvalue_curves": eigenvalue_curves.tolist(),
        "artifacts": {key: str(path) for key, path in plot_paths.items()},
        "representative_artifacts": representative_artifacts,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Fixed-camera checkerboard-motion experiments for common calibration weakness modes.',
    )
    parser.add_argument(
        '--modes',
        nargs="+",
        default=[
            "limited_viewpoint_diversity",
            "narrow_azimuth_coverage",
            "little_depth_variation",
            "little_tilt_variation",
            "poor_image_plane_coverage",
            "small_target_footprint",
            "too_few_visible_points",
            "symmetric_geometry",
            "planar_pose_degeneracy",
            "high_pixel_noise",
        ],
        help='subset of weakness modes to run',
    )
    parser.add_argument('-n', '--num_runs', type=int, default=3, help='number of synthetic calibration problems')
    parser.add_argument('-s', '--select_k', type=int, default=15, help='number of poses to select')
    parser.add_argument('--num-candidate-poses', type=int, default=300, help='number of sampled candidate board poses')
    parser.add_argument('--width', type=int, default=1280, help='image width in pixels')
    parser.add_argument('--height', type=int, default=960, help='image height in pixels')
    parser.add_argument('--pixel-noise', type=float, default=1.0, help='baseline pixel noise standard deviation')
    parser.add_argument('--min-radius', type=float, default=4.0, help='baseline minimum board distance')
    parser.add_argument('--max-radius', type=float, default=6.0, help='baseline maximum board distance')
    parser.add_argument('--azimuth-center-deg', type=float, default=0.0, help='azimuth center for board motion')
    parser.add_argument('--min-offaxis-deg', type=float, default=2.0, help='baseline minimum off-axis angle')
    parser.add_argument('--max-offaxis-deg', type=float, default=25.0, help='baseline maximum off-axis angle')
    parser.add_argument('--tilt-span-deg', type=float, default=30.0, help='baseline board tilt span')
    parser.add_argument('--board-rows', type=int, default=9, help='baseline checkerboard rows')
    parser.add_argument('--board-cols', type=int, default=9, help='baseline checkerboard columns')
    parser.add_argument('--board-square-size', type=float, default=0.125, help='baseline checkerboard square size')
    parser.add_argument(
        '-o',
        '--output-dir',
        default=str(pathlib.Path(__file__).resolve().parents[1] / "results"),
        help='directory for plots',
    )
    args = parser.parse_args()

    specs = mode_definitions()
    unknown_modes = sorted(set(args.modes) - set(specs.keys()))
    if unknown_modes:
        raise ValueError(f"Unknown modes requested: {unknown_modes}")

    out_dir = create_run_output_dir(pathlib.Path(args.output_dir), "fixed_camera_weaknesses")
    print("Fixed-camera checkerboard-motion calibration weakness experiments")
    print("Plots saved per mode:")
    print("  min eig, trace covariance, condition number, logdet, visible points")
    print("  intrinsic uncertainty per parameter")
    print("  all final eigenvalues")
    print("  representative weak/strong 3D pose-selection plots")
    print("  representative weak/strong 3D candidate eigenvalue maps")

    mode_results = {}
    for mode_name in args.modes:
        mode_results[mode_name] = run_sweep(mode_name, specs[mode_name], args, out_dir)

    summary_path = out_dir / "summary.json"
    metadata_path = out_dir / "metadata.json"
    summary = {
        "experiment_name": "fixed_camera_weaknesses",
        "output_dir": str(out_dir),
        "mode_results": mode_results,
        "report_interpretation": build_report_interpretation(),
    }
    metadata = {
        "experiment_name": "fixed_camera_weaknesses",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "parameters": {
            "modes": args.modes,
            "num_runs": args.num_runs,
            "select_k": args.select_k,
            "num_candidate_poses": args.num_candidate_poses,
            "image_size": [args.width, args.height],
            "pixel_noise": args.pixel_noise,
            "radius_range": [args.min_radius, args.max_radius],
            "azimuth_center_deg": args.azimuth_center_deg,
            "offaxis_angle_range_deg": [args.min_offaxis_deg, args.max_offaxis_deg],
            "tilt_span_deg": args.tilt_span_deg,
            "board_rows": args.board_rows,
            "board_cols": args.board_cols,
            "board_square_size": args.board_square_size,
        },
        "results": mode_results,
        "report_interpretation": build_report_interpretation(),
        "artifacts": {
            "summary_json": str(summary_path),
            "metadata_json": str(metadata_path),
        },
    }
    write_json(summary_path, summary)
    write_json(metadata_path, metadata)

    print(f"Results saved to: {out_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"Metadata saved to: {metadata_path}")
