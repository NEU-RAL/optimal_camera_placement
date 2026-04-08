import pathlib
import sys
from datetime import datetime
import json

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


def build_report_interpretation() -> dict:
    return {
        "primary_objective": {
            "name": "min_eig",
            "description": "Minimum eigenvalue of the calibration information matrix H_cal.",
            "desired_direction": "higher",
            "why": "Higher values mean the weakest constrained calibration direction is better observed.",
        },
        "metrics": {
            "min_eig": {
                "description": "Worst-direction information strength after marginalizing nuisance pose variables.",
                "desired_direction": "higher",
            },
            "max_eig": {
                "description": "Strongest constrained calibration direction.",
                "desired_direction": "context_dependent",
            },
            "logdet": {
                "description": "Overall information volume of the calibration estimate.",
                "desired_direction": "higher",
            },
            "trace_cov": {
                "description": "Sum of parameter variances from the approximate covariance; total uncertainty.",
                "desired_direction": "lower",
            },
            "cond": {
                "description": "Condition number of H_cal; lower means better numerical conditioning and more balanced constraints.",
                "desired_direction": "lower",
            },
            "visible_points": {
                "description": "Total number of visible checkerboard points across the selected poses.",
                "desired_direction": "higher_usually",
            },
        },
        "selection_goal": "Prefer pose sets with high min_eig, low trace_cov, and low cond.",
        "comparison_note": "selected_report should ideally beat random_average_report on min_eig and usually also improve trace_cov and cond.",
    }


if __name__ == '__main__':
    num_candidate_poses = 1000
    select_k = 20
    radius_range = (0.6, 1.2)
    radius_samples = 3
    problem = sdu.generate_calibration_problem(
        num_candidate_poses=num_candidate_poses,
        radius_samples=radius_samples,
        radius_range=radius_range,
    )
    prior = fim.build_prior_blocks(problem)
    selected_poses, selected_indices, best_score, selection = methods.greedy_selection(problem, select_k=select_k, prior=prior)
    report = cal_analysis.compare_selected_vs_random(problem, selected_indices, prior=prior, num_random_trials=25, seed=7)
    candidate_scores = cal_analysis.candidate_min_eig_scores(problem, prior=prior)
    candidate_spectra = cal_analysis.candidate_eigenvalue_spectra(problem, prior=prior)
    before_after = cal_analysis.before_after_calibration_summary(problem, selected_indices, prior=prior)
    results_root = pathlib.Path(__file__).resolve().parents[1] / "results"
    out_dir = create_run_output_dir(results_root, "single_problem")
    pose_plot_path = out_dir / "single_problem_pose_selection.png"
    report_plot_path = out_dir / "single_problem_selection_report.png"
    eig_plot_path = out_dir / "single_problem_candidate_eigenvalues.png"
    eig_plot_3d_path = out_dir / "single_problem_candidate_eigenvalues_3d.png"
    eig_spectra_plot_path = out_dir / "single_problem_candidate_eigenvalue_spectra.png"
    uncertainty_plot_path = out_dir / "single_problem_parameter_uncertainty_before_after.png"
    min_eig_compare_plot_path = out_dir / "single_problem_min_eig_before_after.png"
    eig_compare_plot_path = out_dir / "single_problem_eigenvalues_before_after.png"
    summary_path = out_dir / "summary.json"
    metadata_path = out_dir / "metadata.json"
    cal_viz.plot_pose_selection(problem, selected_indices, report["random_best_indices"], save_path=str(pose_plot_path))
    cal_viz.plot_selection_report(report, save_path=str(report_plot_path))
    cal_viz.plot_candidate_eigenvalues(candidate_scores, selected_indices, report["random_best_indices"], save_path=str(eig_plot_path))
    cal_viz.plot_candidate_eigenvalues_3d(
        problem,
        candidate_scores,
        selected_indices,
        report["random_best_indices"],
        save_path=str(eig_plot_3d_path),
    )
    cal_viz.plot_candidate_eigenvalue_spectra(
        candidate_spectra,
        selected_indices,
        report["random_best_indices"],
        save_path=str(eig_spectra_plot_path),
    )
    cal_viz.plot_parameter_uncertainty_before_after(
        before_after["parameter_labels"],
        before_after["before"]["std_dev"],
        before_after["after"]["std_dev"],
        save_path=str(uncertainty_plot_path),
    )
    cal_viz.plot_min_eigenvalue_before_after(
        before_after["before"]["min_eig"],
        before_after["after"]["min_eig"],
        save_path=str(min_eig_compare_plot_path),
    )
    cal_viz.plot_eigenvalues_before_after(
        before_after["before"]["eigvals"],
        before_after["after"]["eigvals"],
        save_path=str(eig_compare_plot_path),
    )

    summary = {
        "num_candidate_poses": num_candidate_poses,
        "select_k": select_k,
        "radius_samples": radius_samples,
        "radius_range": list(radius_range),
        "selected_indices": selected_indices,
        "best_score": best_score,
        "selection_vector": selection.astype(int).tolist(),
        "selected_report": report["selected"],
        "random_average_report": report["random_average"],
        "random_best_report": report["random_best"],
        "random_best_indices": report["random_best_indices"],
        "candidate_scores": candidate_scores.tolist(),
        "candidate_eigenvalue_spectra": candidate_spectra.tolist(),
        "before_after_calibration": before_after,
        "report_interpretation": build_report_interpretation(),
        "output_dir": str(out_dir),
    }
    metadata = {
        "experiment_name": "single_problem",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "parameters": {
            "num_candidate_poses": num_candidate_poses,
            "select_k": select_k,
            "radius_samples": radius_samples,
            "radius_range": list(radius_range),
        },
        "problem": {
            "image_size": list(problem.image_size),
            "pixel_noise_sigma": problem.pixel_noise_sigma,
            "intrinsics_gt": problem.intrinsics_gt.tolist(),
            "intrinsics_init": problem.intrinsics_init.tolist(),
            "num_candidates": problem.num_candidates,
            "intrinsics_dim": problem.intrinsics_dim,
            "sampled_radii": sorted(np.unique(np.linalg.norm(problem.candidate_translations, axis=1)).round(6).tolist()),
        },
        "results": {
            "selected_indices": selected_indices,
            "best_score": best_score,
            "selection_vector": selection.astype(int).tolist(),
            "selected_report": report["selected"],
            "random_average_report": report["random_average"],
            "random_best_report": report["random_best"],
            "random_best_indices": report["random_best_indices"],
            "before_after_calibration": before_after,
        },
        "report_interpretation": build_report_interpretation(),
        "artifacts": {
            "pose_plot": str(pose_plot_path),
            "report_plot": str(report_plot_path),
            "candidate_eigenvalue_plot": str(eig_plot_path),
            "candidate_eigenvalue_3d_plot": str(eig_plot_3d_path),
            "candidate_eigenvalue_spectra_plot": str(eig_spectra_plot_path),
            "parameter_uncertainty_before_after_plot": str(uncertainty_plot_path),
            "min_eig_before_after_plot": str(min_eig_compare_plot_path),
            "eigenvalues_before_after_plot": str(eig_compare_plot_path),
            "summary_json": str(summary_path),
        },
    }
    write_json(summary_path, summary)
    write_json(metadata_path, metadata)

    print('Optimal pose selection for camera intrinsics calibration')
    print('Objective: maximize lambda_min(H_cal)')
    print(f'Number of sampled candidate poses: {num_candidate_poses}')
    print(f'Radius discretization: {radius_samples} shells over [{radius_range[0]:.2f}, {radius_range[1]:.2f}]')
    print(f'Number of selected poses: {select_k}')
    print(f'Selected pose indices: {selected_indices}')
    print(f'Best score: {best_score:.6f}')
    for idx, pose in zip(selected_indices, selected_poses):
        print(f'Pose {idx}: t_wc = {pose["translation_wc"]}, visible_points = {pose["visible_points"]}')
    print(f'Selection vector: {selection.astype(int)}')
    print('Selected-vs-random report:')
    print(f'  selected min eig: {report["selected"]["min_eig"]:.6f}')
    print(f'  random average min eig: {report["random_average"]["min_eig"]:.6f}')
    print(f'  random best min eig: {report["random_best"]["min_eig"]:.6f}')
    print(f'  random best indices: {report["random_best_indices"]}')
    print(f'  candidate eigenvalue plot saved to: {eig_plot_path}')
    print(f'  3D eigenvalue plot saved to: {eig_plot_3d_path}')
    print(f'  eigenvalue spectra plot saved to: {eig_spectra_plot_path}')
    print(f'  parameter uncertainty plot saved to: {uncertainty_plot_path}')
    print(f'  min eigenvalue comparison plot saved to: {min_eig_compare_plot_path}')
    print(f'  eigenvalue comparison plot saved to: {eig_compare_plot_path}')
    print(f'  pose plot saved to: {pose_plot_path}')
    print(f'  report plot saved to: {report_plot_path}')
    print(f'  summary saved to: {summary_path}')
    print(f'  metadata saved to: {metadata_path}')
