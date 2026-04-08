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
            "runtime_seconds": {"desired_direction": "lower"},
        },
        "plots_generated": [
            "Heatmaps over candidate count N and selected pose count k for min_eig, trace_cov, cond, logdet, visible_points, and runtime.",
            "Metric-vs-candidate-count curves with one line per selected k, including visible points.",
            "Metric-vs-selected-k curves with one line per candidate count N, including visible points.",
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Sweep candidate-pool size N and selected pose count k.',
    )
    parser.add_argument(
        '--candidate-counts',
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 200, 400, 700, 1000],
        help='candidate pool sizes N to evaluate',
    )
    parser.add_argument(
        '--select-ks',
        type=int,
        nargs="+",
        default=[10, 20, 40, 60, 80, 100],
        help='numbers of poses k to select',
    )
    parser.add_argument('-n', '--num_runs', type=int, default=3, help='number of synthetic calibration problems')
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
    parser.add_argument('--board-square-size', type=float, default=0.125, help='checkerboard square size')
    parser.add_argument(
        '-o',
        '--output-dir',
        default=str(pathlib.Path(__file__).resolve().parents[1] / "results"),
        help='directory for plots',
    )
    args = parser.parse_args()

    candidate_counts = sorted(set(args.candidate_counts))
    select_ks = sorted(set(args.select_ks))
    out_dir = create_run_output_dir(pathlib.Path(args.output_dir), "candidate_k_sweep")

    grid_results = []
    min_eig_matrix = np.full((len(select_ks), len(candidate_counts)), np.nan, dtype=float)
    trace_cov_matrix = np.full((len(select_ks), len(candidate_counts)), np.nan, dtype=float)
    cond_matrix = np.full((len(select_ks), len(candidate_counts)), np.nan, dtype=float)
    logdet_matrix = np.full((len(select_ks), len(candidate_counts)), np.nan, dtype=float)
    visible_points_matrix = np.full((len(select_ks), len(candidate_counts)), np.nan, dtype=float)
    runtime_matrix = np.full((len(select_ks), len(candidate_counts)), np.nan, dtype=float)

    for cand_col, num_candidates in enumerate(candidate_counts):
        problems = sdu.generate_problem_set(
            num_runs=args.num_runs,
            image_size=(args.width, args.height),
            pixel_noise_sigma=args.pixel_noise,
            num_candidate_poses=num_candidates,
            azimuth_samples=args.azimuth_samples,
            elevation_samples=args.elevation_samples,
            radius_samples=args.radius_samples,
            radius_range=(args.min_radius, args.max_radius),
            board_rows=args.board_rows,
            board_cols=args.board_cols,
            board_square_size=args.board_square_size,
        )
        priors = [fim.build_prior_blocks(problem) for problem in problems]

        for k_row, select_k in enumerate(select_ks):
            if select_k > num_candidates:
                continue

            best_score, selected_poses, selected_indices, elapsed, selection = methods.run_single_experiment_exp(
                problems,
                select_k=select_k,
                priors=priors,
            )
            selected_reports = [
                cal_analysis.evaluate_selection(problem, selected_indices, prior=prior)
                for problem, prior in zip(problems, priors)
            ]
            selected_avg = average_metric_reports(selected_reports)

            result = {
                "num_candidate_poses": num_candidates,
                "select_k": select_k,
                "selected_indices": selected_indices,
                "best_score": best_score,
                "runtime_seconds": elapsed,
                "selection_vector": selection.astype(int).tolist(),
                "selected_report": selected_avg,
            }
            grid_results.append(result)

            min_eig_matrix[k_row, cand_col] = selected_avg["min_eig"]
            trace_cov_matrix[k_row, cand_col] = selected_avg["trace_cov"]
            cond_matrix[k_row, cand_col] = selected_avg["cond"]
            logdet_matrix[k_row, cand_col] = selected_avg["logdet"]
            visible_points_matrix[k_row, cand_col] = selected_avg["visible_points"]
            runtime_matrix[k_row, cand_col] = elapsed

    plot_paths = {
        "min_eig_heatmap": out_dir / "candidate_k_min_eig_heatmap.png",
        "trace_cov_heatmap": out_dir / "candidate_k_trace_cov_heatmap.png",
        "cond_heatmap": out_dir / "candidate_k_condition_number_heatmap.png",
        "logdet_heatmap": out_dir / "candidate_k_logdet_heatmap.png",
        "visible_points_heatmap": out_dir / "candidate_k_visible_points_heatmap.png",
        "runtime_heatmap": out_dir / "candidate_k_runtime_heatmap.png",
        "min_eig_vs_candidates": out_dir / "candidate_k_min_eig_vs_candidates.png",
        "trace_cov_vs_candidates": out_dir / "candidate_k_trace_cov_vs_candidates.png",
        "cond_vs_candidates": out_dir / "candidate_k_condition_number_vs_candidates.png",
        "logdet_vs_candidates": out_dir / "candidate_k_logdet_vs_candidates.png",
        "visible_points_vs_candidates": out_dir / "candidate_k_visible_points_vs_candidates.png",
        "runtime_vs_candidates": out_dir / "candidate_k_runtime_vs_candidates.png",
        "min_eig_by_candidate_curve_vs_k": out_dir / "candidate_k_min_eig_by_candidate_curve_vs_k.png",
        "trace_cov_by_candidate_curve_vs_k": out_dir / "candidate_k_trace_cov_by_candidate_curve_vs_k.png",
        "cond_by_candidate_curve_vs_k": out_dir / "candidate_k_condition_number_by_candidate_curve_vs_k.png",
        "logdet_by_candidate_curve_vs_k": out_dir / "candidate_k_logdet_by_candidate_curve_vs_k.png",
        "visible_points_by_candidate_curve_vs_k": out_dir / "candidate_k_visible_points_by_candidate_curve_vs_k.png",
        "runtime_by_candidate_curve_vs_k": out_dir / "candidate_k_runtime_by_candidate_curve_vs_k.png",
    }
    summary_path = out_dir / "summary.json"
    metadata_path = out_dir / "metadata.json"

    cal_viz.plot_metric_heatmap(
        candidate_counts,
        select_ks,
        min_eig_matrix,
        xlabel="num candidate poses",
        ylabel="selected k",
        title="Minimum Eigenvalue Over Candidate Count And Selected k",
        colorbar_label="min eig(H_cal)",
        save_path=str(plot_paths["min_eig_heatmap"]),
    )
    cal_viz.plot_metric_heatmap(
        candidate_counts,
        select_ks,
        trace_cov_matrix,
        xlabel="num candidate poses",
        ylabel="selected k",
        title="Trace Covariance Over Candidate Count And Selected k",
        colorbar_label="trace(covariance)",
        save_path=str(plot_paths["trace_cov_heatmap"]),
    )
    cal_viz.plot_metric_heatmap(
        candidate_counts,
        select_ks,
        cond_matrix,
        xlabel="num candidate poses",
        ylabel="selected k",
        title="Condition Number Over Candidate Count And Selected k",
        colorbar_label="condition number",
        save_path=str(plot_paths["cond_heatmap"]),
    )
    cal_viz.plot_metric_heatmap(
        candidate_counts,
        select_ks,
        logdet_matrix,
        xlabel="num candidate poses",
        ylabel="selected k",
        title="Log Determinant Over Candidate Count And Selected k",
        colorbar_label="logdet(H_cal)",
        save_path=str(plot_paths["logdet_heatmap"]),
    )
    cal_viz.plot_metric_heatmap(
        candidate_counts,
        select_ks,
        visible_points_matrix,
        xlabel="num candidate poses",
        ylabel="selected k",
        title="Visible Points Over Candidate Count And Selected k",
        colorbar_label="visible points",
        save_path=str(plot_paths["visible_points_heatmap"]),
    )
    cal_viz.plot_metric_heatmap(
        candidate_counts,
        select_ks,
        runtime_matrix,
        xlabel="num candidate poses",
        ylabel="selected k",
        title="Runtime Over Candidate Count And Selected k",
        colorbar_label="runtime (s)",
        save_path=str(plot_paths["runtime_heatmap"]),
    )

    by_k_series = {
        f"k={select_k}": min_eig_matrix[k_idx, :]
        for k_idx, select_k in enumerate(select_ks)
    }
    cal_viz.plot_metric_curves_by_series(
        candidate_counts,
        by_k_series,
        xlabel="num candidate poses",
        ylabel="min eig(H_cal)",
        title="Minimum Eigenvalue Vs Candidate Count",
        save_path=str(plot_paths["min_eig_vs_candidates"]),
    )
    by_k_series = {f"k={select_k}": trace_cov_matrix[k_idx, :] for k_idx, select_k in enumerate(select_ks)}
    cal_viz.plot_metric_curves_by_series(
        candidate_counts,
        by_k_series,
        xlabel="num candidate poses",
        ylabel="trace(covariance)",
        title="Trace Covariance Vs Candidate Count",
        save_path=str(plot_paths["trace_cov_vs_candidates"]),
    )
    by_k_series = {f"k={select_k}": cond_matrix[k_idx, :] for k_idx, select_k in enumerate(select_ks)}
    cal_viz.plot_metric_curves_by_series(
        candidate_counts,
        by_k_series,
        xlabel="num candidate poses",
        ylabel="condition number",
        title="Condition Number Vs Candidate Count",
        save_path=str(plot_paths["cond_vs_candidates"]),
    )
    by_k_series = {f"k={select_k}": logdet_matrix[k_idx, :] for k_idx, select_k in enumerate(select_ks)}
    cal_viz.plot_metric_curves_by_series(
        candidate_counts,
        by_k_series,
        xlabel="num candidate poses",
        ylabel="logdet(H_cal)",
        title="Log Determinant Vs Candidate Count",
        save_path=str(plot_paths["logdet_vs_candidates"]),
    )
    by_k_series = {f"k={select_k}": visible_points_matrix[k_idx, :] for k_idx, select_k in enumerate(select_ks)}
    cal_viz.plot_metric_curves_by_series(
        candidate_counts,
        by_k_series,
        xlabel="num candidate poses",
        ylabel="visible points",
        title="Visible Points Vs Candidate Count",
        save_path=str(plot_paths["visible_points_vs_candidates"]),
    )
    by_k_series = {f"k={select_k}": runtime_matrix[k_idx, :] for k_idx, select_k in enumerate(select_ks)}
    cal_viz.plot_metric_curves_by_series(
        candidate_counts,
        by_k_series,
        xlabel="num candidate poses",
        ylabel="runtime (s)",
        title="Runtime Vs Candidate Count",
        save_path=str(plot_paths["runtime_vs_candidates"]),
    )

    by_candidate_series = {
        f"N={num_candidates}": min_eig_matrix[:, n_idx]
        for n_idx, num_candidates in enumerate(candidate_counts)
    }
    cal_viz.plot_metric_curves_by_series(
        select_ks,
        by_candidate_series,
        xlabel="selected k",
        ylabel="min eig(H_cal)",
        title="Minimum Eigenvalue Vs Selected k (One Curve Per Candidate Count)",
        save_path=str(plot_paths["min_eig_by_candidate_curve_vs_k"]),
    )
    by_candidate_series = {f"N={num_candidates}": trace_cov_matrix[:, n_idx] for n_idx, num_candidates in enumerate(candidate_counts)}
    cal_viz.plot_metric_curves_by_series(
        select_ks,
        by_candidate_series,
        xlabel="selected k",
        ylabel="trace(covariance)",
        title="Trace Covariance Vs Selected k (One Curve Per Candidate Count)",
        save_path=str(plot_paths["trace_cov_by_candidate_curve_vs_k"]),
    )
    by_candidate_series = {f"N={num_candidates}": cond_matrix[:, n_idx] for n_idx, num_candidates in enumerate(candidate_counts)}
    cal_viz.plot_metric_curves_by_series(
        select_ks,
        by_candidate_series,
        xlabel="selected k",
        ylabel="condition number",
        title="Condition Number Vs Selected k (One Curve Per Candidate Count)",
        save_path=str(plot_paths["cond_by_candidate_curve_vs_k"]),
    )
    by_candidate_series = {f"N={num_candidates}": logdet_matrix[:, n_idx] for n_idx, num_candidates in enumerate(candidate_counts)}
    cal_viz.plot_metric_curves_by_series(
        select_ks,
        by_candidate_series,
        xlabel="selected k",
        ylabel="logdet(H_cal)",
        title="Log Determinant Vs Selected k (One Curve Per Candidate Count)",
        save_path=str(plot_paths["logdet_by_candidate_curve_vs_k"]),
    )
    by_candidate_series = {f"N={num_candidates}": visible_points_matrix[:, n_idx] for n_idx, num_candidates in enumerate(candidate_counts)}
    cal_viz.plot_metric_curves_by_series(
        select_ks,
        by_candidate_series,
        xlabel="selected k",
        ylabel="visible points",
        title="Visible Points Vs Selected k (One Curve Per Candidate Count)",
        save_path=str(plot_paths["visible_points_by_candidate_curve_vs_k"]),
    )
    by_candidate_series = {f"N={num_candidates}": runtime_matrix[:, n_idx] for n_idx, num_candidates in enumerate(candidate_counts)}
    cal_viz.plot_metric_curves_by_series(
        select_ks,
        by_candidate_series,
        xlabel="selected k",
        ylabel="runtime (s)",
        title="Runtime Vs Selected k (One Curve Per Candidate Count)",
        save_path=str(plot_paths["runtime_by_candidate_curve_vs_k"]),
    )

    summary = {
        "candidate_counts": candidate_counts,
        "select_ks": select_ks,
        "grid_results": grid_results,
        "min_eig_matrix": min_eig_matrix.tolist(),
        "trace_cov_matrix": trace_cov_matrix.tolist(),
        "cond_matrix": cond_matrix.tolist(),
        "logdet_matrix": logdet_matrix.tolist(),
        "visible_points_matrix": visible_points_matrix.tolist(),
        "runtime_matrix": runtime_matrix.tolist(),
        "report_interpretation": build_report_interpretation(),
        "output_dir": str(out_dir),
    }
    metadata = {
        "experiment_name": "candidate_k_sweep",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "parameters": {
            "candidate_counts": candidate_counts,
            "select_ks": select_ks,
            "num_runs": args.num_runs,
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
            "grid_results": grid_results,
            "min_eig_matrix": min_eig_matrix.tolist(),
            "trace_cov_matrix": trace_cov_matrix.tolist(),
            "cond_matrix": cond_matrix.tolist(),
            "logdet_matrix": logdet_matrix.tolist(),
            "visible_points_matrix": visible_points_matrix.tolist(),
            "runtime_matrix": runtime_matrix.tolist(),
        },
        "report_interpretation": build_report_interpretation(),
        "artifacts": {name: str(path) for name, path in plot_paths.items()} | {"summary_json": str(summary_path)},
    }
    write_json(summary_path, summary)
    write_json(metadata_path, metadata)

    print("Candidate-count and selected-k sweep experiment")
    print(f"Candidate counts: {candidate_counts}")
    print(f"Selected k values: {select_ks}")
    print("Plots generated:")
    print("  heatmaps for min eig, trace covariance, condition number, logdet, visible points, and runtime")
    print("  metric-vs-candidate-count curves for each k, including visible points")
    print("  metric-vs-selected-k curves with one unique curve per candidate count, including visible points")
    print(f"Results saved to: {out_dir}")
    print(f"  summary saved to: {summary_path}")
    print(f"  metadata saved to: {metadata_path}")
