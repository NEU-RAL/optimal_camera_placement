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
            "How restricting candidate poses to a narrower azimuth sector changes calibration quality, "
            "intrinsic uncertainty, and eigenvalue structure."
        ),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Sweep candidate-pose azimuth span while keeping candidate count fixed.',
    )
    parser.add_argument(
        '--azimuth-spans-deg',
        type=float,
        nargs="+",
        default=[30.0, 60.0, 90.0, 120.0, 180.0, 360.0],
        help='azimuth span values in degrees for candidate-pose sampling',
    )
    parser.add_argument('--azimuth-center-deg', type=float, default=0.0, help='center azimuth angle in degrees')
    parser.add_argument('-n', '--num_runs', type=int, default=5, help='number of synthetic calibration problems')
    parser.add_argument('-s', '--select_k', type=int, default=20, help='number of poses to select')
    parser.add_argument('--num-candidate-poses', type=int, default=1000, help='number of sampled candidate poses')
    parser.add_argument('--width', type=int, default=1280, help='image width in pixels')
    parser.add_argument('--height', type=int, default=960, help='image height in pixels')
    parser.add_argument('--pixel-noise', type=float, default=1.0, help='pixel noise standard deviation')
    parser.add_argument('--azimuth-samples', type=int, default=4, help='candidate azimuth samples for grid mode')
    parser.add_argument('--elevation-samples', type=int, default=2, help='candidate elevation samples')
    parser.add_argument('--radius-samples', type=int, default=3, help='candidate radius samples')
    parser.add_argument('--min-radius', type=float, default=0.6, help='minimum candidate radius')
    parser.add_argument('--max-radius', type=float, default=1.2, help='maximum candidate radius')
    parser.add_argument('--board-rows', type=int, default=9, help='checkerboard rows')
    parser.add_argument('--board-cols', type=int, default=9, help='checkerboard columns')
    parser.add_argument('--board-square-size', type=float, default=0.125, help='checkerboard square size')
    parser.add_argument(
        '-o',
        '--output-dir',
        default=str(pathlib.Path(__file__).resolve().parents[1] / "results"),
        help='directory for plots',
    )
    args = parser.parse_args()

    out_dir = create_run_output_dir(pathlib.Path(args.output_dir), "azimuth_span")

    print("Azimuth span experiment")
    print("Plots that will be saved:")
    print("  azimuth_span_min_eig.png")
    print("  azimuth_span_trace_cov.png")
    print("  azimuth_span_condition_number.png")
    print("  azimuth_span_logdet.png")
    print("  azimuth_span_visible_points.png")
    print("  azimuth_span_parameter_uncertainty.png")
    print("  azimuth_span_eigenvalues.png")
    print("  azimuth_span_<span>deg_pose_selection.png")
    print("  azimuth_span_<span>deg_candidate_eigenvalues_3d.png")

    span_results = []
    parameter_labels = None
    selected_uncertainty_curves = []
    selected_eigenvalue_curves = []
    per_span_artifacts = []

    for azimuth_span_deg in args.azimuth_spans_deg:
        problems = sdu.generate_problem_set(
            num_runs=args.num_runs,
            image_size=(args.width, args.height),
            pixel_noise_sigma=args.pixel_noise,
            num_candidate_poses=args.num_candidate_poses,
            azimuth_samples=args.azimuth_samples,
            elevation_samples=args.elevation_samples,
            radius_samples=args.radius_samples,
            radius_range=(args.min_radius, args.max_radius),
            azimuth_span_deg=azimuth_span_deg,
            azimuth_center_deg=args.azimuth_center_deg,
            board_rows=args.board_rows,
            board_cols=args.board_cols,
            board_square_size=args.board_square_size,
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
        report = cal_analysis.compare_selected_vs_random(
            problems[0],
            selected_indices,
            prior=priors[0],
            num_random_trials=25,
            seed=11,
        )
        candidate_scores = np.mean(
            [cal_analysis.candidate_min_eig_scores(problem, prior=prior) for problem, prior in zip(problems, priors)],
            axis=0,
        )

        before_after_per_problem = [
            cal_analysis.before_after_calibration_summary(problem, selected_indices, prior=prior)
            for problem, prior in zip(problems, priors)
        ]
        before_after = average_before_after(before_after_per_problem)
        if parameter_labels is None:
            parameter_labels = before_after["parameter_labels"]
        selected_uncertainty_curves.append(before_after["after"]["std_dev"])
        selected_eigenvalue_curves.append(before_after["after"]["eigvals"])

        span_tag = f"{int(round(azimuth_span_deg))}deg"
        pose_plot_path = out_dir / f"azimuth_span_{span_tag}_pose_selection.png"
        eig_plot_3d_path = out_dir / f"azimuth_span_{span_tag}_candidate_eigenvalues_3d.png"
        cal_viz.plot_pose_selection(
            problems[0],
            selected_indices,
            report["random_best_indices"],
            save_path=str(pose_plot_path),
        )
        cal_viz.plot_candidate_eigenvalues_3d(
            problems[0],
            candidate_scores,
            selected_indices,
            report["random_best_indices"],
            save_path=str(eig_plot_3d_path),
        )
        per_span_artifacts.append(
            {
                "azimuth_span_deg": azimuth_span_deg,
                "pose_selection_plot": str(pose_plot_path),
                "candidate_eigenvalue_3d_plot": str(eig_plot_3d_path),
            }
        )

        span_results.append(
            {
                "azimuth_span_deg": azimuth_span_deg,
                "azimuth_center_deg": args.azimuth_center_deg,
                "selected_indices": selected_indices,
                "best_score": best_score,
                "runtime_seconds": elapsed,
                "selection_vector": selection.astype(int).tolist(),
                "selected_report": selected_avg,
                "random_average_report": random_avg,
                "random_best_indices": report["random_best_indices"],
                "candidate_scores": candidate_scores.tolist(),
                "before_after_calibration": before_after,
            }
        )

    azimuth_spans_deg = [item["azimuth_span_deg"] for item in span_results]
    selected_min_eig = [item["selected_report"]["min_eig"] for item in span_results]
    random_min_eig = [item["random_average_report"]["min_eig"] for item in span_results]
    selected_trace_cov = [item["selected_report"]["trace_cov"] for item in span_results]
    random_trace_cov = [item["random_average_report"]["trace_cov"] for item in span_results]
    selected_cond = [item["selected_report"]["cond"] for item in span_results]
    random_cond = [item["random_average_report"]["cond"] for item in span_results]
    selected_logdet = [item["selected_report"]["logdet"] for item in span_results]
    random_logdet = [item["random_average_report"]["logdet"] for item in span_results]
    selected_visible_points = [item["selected_report"]["visible_points"] for item in span_results]
    random_visible_points = [item["random_average_report"]["visible_points"] for item in span_results]
    selected_uncertainty_curves = np.asarray(selected_uncertainty_curves, dtype=float)
    selected_eigenvalue_curves = np.asarray(selected_eigenvalue_curves, dtype=float)

    min_eig_plot_path = out_dir / "azimuth_span_min_eig.png"
    trace_cov_plot_path = out_dir / "azimuth_span_trace_cov.png"
    cond_plot_path = out_dir / "azimuth_span_condition_number.png"
    logdet_plot_path = out_dir / "azimuth_span_logdet.png"
    visible_points_plot_path = out_dir / "azimuth_span_visible_points.png"
    uncertainty_plot_path = out_dir / "azimuth_span_parameter_uncertainty.png"
    eigvals_plot_path = out_dir / "azimuth_span_eigenvalues.png"
    summary_path = out_dir / "summary.json"
    metadata_path = out_dir / "metadata.json"

    cal_viz.plot_azimuth_span_metric(
        azimuth_spans_deg,
        selected_min_eig,
        random_min_eig,
        ylabel="min eig(H_cal)",
        title="Azimuth Span Vs Minimum Eigenvalue",
        save_path=str(min_eig_plot_path),
    )
    cal_viz.plot_azimuth_span_metric(
        azimuth_spans_deg,
        selected_trace_cov,
        random_trace_cov,
        ylabel="trace(covariance)",
        title="Azimuth Span Vs Total Uncertainty",
        save_path=str(trace_cov_plot_path),
    )
    cal_viz.plot_azimuth_span_metric(
        azimuth_spans_deg,
        selected_cond,
        random_cond,
        ylabel="condition number",
        title="Azimuth Span Vs Conditioning",
        save_path=str(cond_plot_path),
    )
    cal_viz.plot_azimuth_span_metric(
        azimuth_spans_deg,
        selected_logdet,
        random_logdet,
        ylabel="logdet(H_cal)",
        title="Azimuth Span Vs Log Determinant",
        save_path=str(logdet_plot_path),
    )
    cal_viz.plot_azimuth_span_metric(
        azimuth_spans_deg,
        selected_visible_points,
        random_visible_points,
        ylabel="visible points",
        title="Azimuth Span Vs Visible Points",
        save_path=str(visible_points_plot_path),
    )
    cal_viz.plot_parameter_uncertainty_vs_azimuth_span(
        azimuth_spans_deg,
        parameter_labels or ["fx", "fy", "cx", "cy"],
        selected_uncertainty_curves,
        save_path=str(uncertainty_plot_path),
    )
    cal_viz.plot_eigenvalues_vs_azimuth_span(
        azimuth_spans_deg,
        selected_eigenvalue_curves,
        save_path=str(eigvals_plot_path),
    )

    summary = {
        "azimuth_spans_deg": list(args.azimuth_spans_deg),
        "azimuth_center_deg": args.azimuth_center_deg,
        "span_results": span_results,
        "parameter_labels": parameter_labels,
        "selected_parameter_uncertainty_curves": selected_uncertainty_curves.tolist(),
        "selected_eigenvalue_curves": selected_eigenvalue_curves.tolist(),
        "per_span_artifacts": per_span_artifacts,
        "report_interpretation": build_report_interpretation(),
        "output_dir": str(out_dir),
    }
    metadata = {
        "experiment_name": "azimuth_span",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "parameters": {
            "azimuth_spans_deg": list(args.azimuth_spans_deg),
            "azimuth_center_deg": args.azimuth_center_deg,
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
            "board_square_size": args.board_square_size,
        },
        "results": {
            "span_results": span_results,
            "parameter_labels": parameter_labels,
            "selected_parameter_uncertainty_curves": selected_uncertainty_curves.tolist(),
            "selected_eigenvalue_curves": selected_eigenvalue_curves.tolist(),
            "per_span_artifacts": per_span_artifacts,
        },
        "report_interpretation": build_report_interpretation(),
        "artifacts": {
            "min_eig_plot": str(min_eig_plot_path),
            "trace_cov_plot": str(trace_cov_plot_path),
            "condition_number_plot": str(cond_plot_path),
            "logdet_plot": str(logdet_plot_path),
            "visible_points_plot": str(visible_points_plot_path),
            "parameter_uncertainty_plot": str(uncertainty_plot_path),
            "eigenvalues_plot": str(eigvals_plot_path),
            "per_span_artifacts": per_span_artifacts,
            "summary_json": str(summary_path),
            "metadata_json": str(metadata_path),
        },
    }
    write_json(summary_path, summary)
    write_json(metadata_path, metadata)

    print(f"Azimuth spans: {args.azimuth_spans_deg}")
    print(f"Azimuth center: {args.azimuth_center_deg} deg")
    print(f"Results saved to: {out_dir}")
    print(f"  min eig plot: {min_eig_plot_path}")
    print(f"  trace covariance plot: {trace_cov_plot_path}")
    print(f"  condition number plot: {cond_plot_path}")
    print(f"  logdet plot: {logdet_plot_path}")
    print(f"  visible points plot: {visible_points_plot_path}")
    print(f"  parameter uncertainty plot: {uncertainty_plot_path}")
    print(f"  eigenvalues plot: {eigvals_plot_path}")
    for artifact in per_span_artifacts:
        print(
            f"  span {artifact['azimuth_span_deg']} deg pose plot: "
            f"{artifact['pose_selection_plot']}"
        )
        print(
            f"  span {artifact['azimuth_span_deg']} deg 3D eigenvalue plot: "
            f"{artifact['candidate_eigenvalue_3d_plot']}"
        )
    print(f"  summary saved to: {summary_path}")
    print(f"  metadata saved to: {metadata_path}")
