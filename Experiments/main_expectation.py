import argparse
import json
import pathlib
import sys
from datetime import datetime

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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Optimal pose selection for camera intrinsics calibration',
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
    parser.add_argument(
        '-o',
        '--output-dir',
        default=str(pathlib.Path(__file__).resolve().parents[1] / "results"),
        help='directory for plots',
    )
    args = parser.parse_args()

    problems = sdu.generate_problem_set(
        num_runs=args.num_runs,
        image_size=(args.width, args.height),
        pixel_noise_sigma=args.pixel_noise,
        num_candidate_poses=args.num_candidate_poses,
        azimuth_samples=args.azimuth_samples,
        elevation_samples=args.elevation_samples,
        radius_samples=args.radius_samples,
        radius_range=(args.min_radius, args.max_radius),
    )
    priors = [fim.build_prior_blocks(problem) for problem in problems]

    best_score, selected_poses, selected_indices, elapsed, selection = methods.run_single_experiment_exp(
        problems,
        select_k=args.select_k,
        priors=priors,
    )
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
    candidate_spectra = np.mean(
        [cal_analysis.candidate_eigenvalue_spectra(problem, prior=prior) for problem, prior in zip(problems, priors)],
        axis=0,
    )
    before_after = cal_analysis.before_after_calibration_summary(problems[0], selected_indices, prior=priors[0])
    out_dir = create_run_output_dir(pathlib.Path(args.output_dir), "expectation")
    pose_plot_path = out_dir / "expectation_pose_selection.png"
    report_plot_path = out_dir / "expectation_selection_report.png"
    eig_plot_path = out_dir / "expectation_candidate_eigenvalues.png"
    eig_plot_3d_path = out_dir / "expectation_candidate_eigenvalues_3d.png"
    eig_spectra_plot_path = out_dir / "expectation_candidate_eigenvalue_spectra.png"
    uncertainty_plot_path = out_dir / "expectation_parameter_uncertainty_before_after.png"
    min_eig_compare_plot_path = out_dir / "expectation_min_eig_before_after.png"
    eig_compare_plot_path = out_dir / "expectation_eigenvalues_before_after.png"
    summary_path = out_dir / "summary.json"
    metadata_path = out_dir / "metadata.json"
    cal_viz.plot_pose_selection(problems[0], selected_indices, report["random_best_indices"], save_path=str(pose_plot_path))
    cal_viz.plot_selection_report(report, save_path=str(report_plot_path))
    cal_viz.plot_candidate_eigenvalues(candidate_scores, selected_indices, report["random_best_indices"], save_path=str(eig_plot_path))
    cal_viz.plot_candidate_eigenvalues_3d(
        problems[0],
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
        "num_runs": args.num_runs,
        "num_candidate_poses": problems[0].num_candidates,
        "select_k": args.select_k,
        "radius_samples": args.radius_samples,
        "radius_range": [args.min_radius, args.max_radius],
        "selected_indices": selected_indices,
        "best_score": best_score,
        "runtime_seconds": elapsed,
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
        "experiment_name": "expectation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "parameters": {
            "num_runs": args.num_runs,
            "select_k": args.select_k,
            "num_candidate_poses": args.num_candidate_poses,
            "width": args.width,
            "height": args.height,
            "pixel_noise": args.pixel_noise,
            "azimuth_samples": args.azimuth_samples,
            "elevation_samples": args.elevation_samples,
            "radius_samples": args.radius_samples,
            "min_radius": args.min_radius,
            "max_radius": args.max_radius,
        },
        "problem": {
            "image_size": list(problems[0].image_size),
            "pixel_noise_sigma": problems[0].pixel_noise_sigma,
            "intrinsics_gt": problems[0].intrinsics_gt.tolist(),
            "intrinsics_init": problems[0].intrinsics_init.tolist(),
            "num_candidates": problems[0].num_candidates,
            "intrinsics_dim": problems[0].intrinsics_dim,
            "sampled_radii": sorted(np.unique(np.linalg.norm(problems[0].candidate_translations, axis=1)).round(6).tolist()),
        },
        "results": {
            "selected_indices": selected_indices,
            "best_score": best_score,
            "runtime_seconds": elapsed,
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
    print(f'Objective: maximize lambda_min(H_cal)')
    print(f'Number of candidate poses: {problems[0].num_candidates}')
    print(f'Radius discretization: {args.radius_samples} shells over [{args.min_radius:.2f}, {args.max_radius:.2f}]')
    print(f'Number of selected poses: {args.select_k}')
    print(f'Selected pose indices: {selected_indices}')
    print(f'Best summed minimum-eigenvalue score: {best_score:.6f}')
    print(f'Runtime: {elapsed:.3f} seconds')

    for idx, pose in zip(selected_indices, selected_poses):
        print(f'Pose {idx}: t_wc = {pose["translation_wc"]}, visible_points = {pose.get("visible_points", "n/a")}')

    print(f'Ground-truth intrinsics: {problems[0].intrinsics_gt}')
    print(f'Initial intrinsics estimate: {problems[0].intrinsics_init}')
    print(f'Selection vector: {selection.astype(int)}')
    print('Selected-vs-random report on the first problem instance:')
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
