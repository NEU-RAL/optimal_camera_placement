import argparse
import json
import pathlib
import sys
from datetime import datetime
from typing import Dict, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

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
        "experiment_question": "How changing checkerboard square spacing affects calibration quality while keeping the number of checkerboard points fixed.",
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Sweep checkerboard spacing while keeping the number of points fixed.',
    )
    parser.add_argument(
        '--square-sizes',
        type=float,
        nargs="+",
        default=[0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        help='checkerboard square sizes to evaluate',
    )
    parser.add_argument('-n', '--num_runs', type=int, default=5, help='number of synthetic calibration problems')
    parser.add_argument('-s', '--select_k', type=int, default=20, help='number of poses to select')
    parser.add_argument('--num-candidate-poses', type=int, default=1000, help='number of sampled candidate poses')
    parser.add_argument('--width', type=int, default=1280, help='image width in pixels')
    parser.add_argument('--height', type=int, default=960, help='image height in pixels')
    parser.add_argument('--pixel-noise', type=float, default=1.0, help='pixel noise standard deviation')
    parser.add_argument('--azimuth-samples', type=int, default=4, help='candidate azimuth samples')
    parser.add_argument('--elevation-samples', type=int, default=2, help='candidate elevation samples')
    parser.add_argument('--radius-samples', type=int, default=3, help='candidate radius samples')
    parser.add_argument('--min-radius', type=float, default=0.6, help='minimum candidate radius')
    parser.add_argument('--max-radius', type=float, default=1.2, help='maximum candidate radius')
    parser.add_argument('--board-rows', type=int, default=9, help='checkerboard rows')
    parser.add_argument('--board-cols', type=int, default=9, help='checkerboard columns')
    parser.add_argument(
        '-o',
        '--output-dir',
        default=str(pathlib.Path(__file__).resolve().parents[1] / "results"),
        help='directory for plots',
    )
    args = parser.parse_args()

    out_dir = create_run_output_dir(pathlib.Path(args.output_dir), "checkerboard_spacing")

    spacing_results = []
    parameter_labels = None
    selected_uncertainty_curves = []
    for square_size in args.square_sizes:
        problems = sdu.generate_problem_set(
            num_runs=args.num_runs,
            image_size=(args.width, args.height),
            pixel_noise_sigma=args.pixel_noise,
            num_candidate_poses=args.num_candidate_poses,
            azimuth_samples=args.azimuth_samples,
            elevation_samples=args.elevation_samples,
            radius_samples=args.radius_samples,
            radius_range=(args.min_radius, args.max_radius),
            board_rows=args.board_rows,
            board_cols=args.board_cols,
            board_square_size=square_size,
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
            cal_analysis.random_baseline_report(problem, args.select_k, prior=prior, num_trials=25, seed=11)["average"]
            for problem, prior in zip(problems, priors)
        ]
        selected_avg = average_metric_reports(selected_reports)
        random_avg = average_metric_reports(random_reports)
        before_after = cal_analysis.before_after_calibration_summary(problems[0], selected_indices, prior=priors[0])
        if parameter_labels is None:
            parameter_labels = before_after["parameter_labels"]
        selected_uncertainty_curves.append(before_after["after"]["std_dev"])

        spacing_results.append(
            {
                "square_size": square_size,
                "num_points": int(args.board_rows * args.board_cols),
                "selected_indices": selected_indices,
                "best_score": best_score,
                "runtime_seconds": elapsed,
                "selection_vector": selection.astype(int).tolist(),
                "selected_report": selected_avg,
                "random_average_report": random_avg,
                "before_after_calibration": before_after,
            }
        )

    spacings = [item["square_size"] for item in spacing_results]
    selected_min_eig = [item["selected_report"]["min_eig"] for item in spacing_results]
    random_min_eig = [item["random_average_report"]["min_eig"] for item in spacing_results]
    selected_trace_cov = [item["selected_report"]["trace_cov"] for item in spacing_results]
    random_trace_cov = [item["random_average_report"]["trace_cov"] for item in spacing_results]
    selected_cond = [item["selected_report"]["cond"] for item in spacing_results]
    random_cond = [item["random_average_report"]["cond"] for item in spacing_results]
    selected_logdet = [item["selected_report"]["logdet"] for item in spacing_results]
    random_logdet = [item["random_average_report"]["logdet"] for item in spacing_results]
    selected_visible_points = [item["selected_report"]["visible_points"] for item in spacing_results]
    random_visible_points = [item["random_average_report"]["visible_points"] for item in spacing_results]
    selected_uncertainty_curves = np.asarray(selected_uncertainty_curves, dtype=float)

    min_eig_plot_path = out_dir / "checkerboard_spacing_min_eig.png"
    trace_cov_plot_path = out_dir / "checkerboard_spacing_trace_cov.png"
    cond_plot_path = out_dir / "checkerboard_spacing_condition_number.png"
    logdet_plot_path = out_dir / "checkerboard_spacing_logdet.png"
    visible_points_plot_path = out_dir / "checkerboard_spacing_visible_points.png"
    uncertainty_plot_path = out_dir / "checkerboard_spacing_parameter_uncertainty.png"
    summary_path = out_dir / "summary.json"
    metadata_path = out_dir / "metadata.json"

    cal_viz.plot_spacing_sweep_metric(
        spacings,
        selected_min_eig,
        random_min_eig,
        ylabel="min eig(H_cal)",
        title="Checkerboard Spacing Vs Minimum Eigenvalue",
        save_path=str(min_eig_plot_path),
    )
    cal_viz.plot_spacing_sweep_metric(
        spacings,
        selected_trace_cov,
        random_trace_cov,
        ylabel="trace(covariance)",
        title="Checkerboard Spacing Vs Total Uncertainty",
        save_path=str(trace_cov_plot_path),
    )
    cal_viz.plot_spacing_sweep_metric(
        spacings,
        selected_cond,
        random_cond,
        ylabel="condition number",
        title="Checkerboard Spacing Vs Conditioning",
        save_path=str(cond_plot_path),
    )
    cal_viz.plot_spacing_sweep_metric(
        spacings,
        selected_logdet,
        random_logdet,
        ylabel="logdet(H_cal)",
        title="Checkerboard Spacing Vs Log Determinant",
        save_path=str(logdet_plot_path),
    )
    cal_viz.plot_spacing_sweep_metric(
        spacings,
        selected_visible_points,
        random_visible_points,
        ylabel="visible points",
        title="Checkerboard Spacing Vs Visible Points",
        save_path=str(visible_points_plot_path),
    )
    cal_viz.plot_parameter_uncertainty_vs_spacing(
        spacings,
        parameter_labels or ["fx", "fy", "cx", "cy"],
        selected_uncertainty_curves,
        save_path=str(uncertainty_plot_path),
    )

    summary = {
        "square_sizes": list(args.square_sizes),
        "num_points": int(args.board_rows * args.board_cols),
        "spacing_results": spacing_results,
        "parameter_labels": parameter_labels,
        "selected_parameter_uncertainty_curves": selected_uncertainty_curves.tolist(),
        "report_interpretation": build_report_interpretation(),
        "output_dir": str(out_dir),
    }
    metadata = {
        "experiment_name": "checkerboard_spacing",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "parameters": {
            "square_sizes": list(args.square_sizes),
            "num_runs": args.num_runs,
            "select_k": args.select_k,
            "num_candidate_poses": args.num_candidate_poses,
            "image_size": [args.width, args.height],
            "pixel_noise": args.pixel_noise,
            "azimuth_samples": args.azimuth_samples,
            "elevation_samples": args.elevation_samples,
            "radius_samples": args.radius_samples,
            "radius_range": [args.min_radius, args.max_radius],
            "board_rows": args.board_rows,
            "board_cols": args.board_cols,
            "num_points_fixed": int(args.board_rows * args.board_cols),
        },
        "results": {
            "spacing_results": spacing_results,
            "parameter_labels": parameter_labels,
            "selected_parameter_uncertainty_curves": selected_uncertainty_curves.tolist(),
        },
        "report_interpretation": build_report_interpretation(),
        "artifacts": {
            "min_eig_plot": str(min_eig_plot_path),
            "trace_cov_plot": str(trace_cov_plot_path),
            "condition_number_plot": str(cond_plot_path),
            "logdet_plot": str(logdet_plot_path),
            "visible_points_plot": str(visible_points_plot_path),
            "parameter_uncertainty_plot": str(uncertainty_plot_path),
            "summary_json": str(summary_path),
        },
    }
    write_json(summary_path, summary)
    write_json(metadata_path, metadata)

    print("Checkerboard spacing experiment")
    print(f"Square sizes: {args.square_sizes}")
    print(f"Fixed checkerboard points: {args.board_rows * args.board_cols}")
    print(f"Results saved to: {out_dir}")
    print(f"  min eig plot: {min_eig_plot_path}")
    print(f"  trace covariance plot: {trace_cov_plot_path}")
    print(f"  condition number plot: {cond_plot_path}")
    print(f"  logdet plot: {logdet_plot_path}")
    print(f"  visible points plot: {visible_points_plot_path}")
    print(f"  parameter uncertainty plot: {uncertainty_plot_path}")
    print(f"  summary saved to: {summary_path}")
    print(f"  metadata saved to: {metadata_path}")
