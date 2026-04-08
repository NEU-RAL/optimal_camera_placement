import argparse
import json
import pathlib
import sys
from datetime import datetime

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from optimal_camera_placement.DataGenerator import sim_data_utils as sdu
from optimal_camera_placement.OASIS.FIM import CalibrationProblem
from optimal_camera_placement.OASIS import calibration_visualize as cal_viz


def create_run_output_dir(base_dir: pathlib.Path, run_name: str) -> pathlib.Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"{run_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def compact_diagnostics(
    diagnostics: list[dict],
    limit: int | None = None,
) -> list[dict]:
    if limit is None:
        selected = diagnostics
    else:
        selected = diagnostics[: max(int(limit), 0)]
    compact = []
    for item in selected:
        compact.append(
            {
                "radius_m": item["radius_m"],
                "azimuth_deg": item["azimuth_deg"],
                "elevation_deg": item["elevation_deg"],
                "roll_deg": item["roll_deg"],
                "yaw_perturb_deg": item["yaw_perturb_deg"],
                "pitch_perturb_deg": item["pitch_perturb_deg"],
                "aim_point_label": item.get("aim_point_label", "center"),
                "all_corners_visible": item["all_corners_visible"],
                "target_area_fraction": item["target_area_fraction"],
                "slant_angle_deg": item["slant_angle_deg"],
                "min_projected_corner_spread_px": item["min_projected_corner_spread_px"],
                "valid": item["valid"],
            }
        )
    return compact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and visualize a structured moving-camera candidate bank."
    )
    parser.add_argument(
        "--sampler-mode",
        choices=["structured", "random"],
        default="structured",
        help="candidate pose generation strategy",
    )
    parser.add_argument("--width", type=int, default=1280, help="image width in pixels")
    parser.add_argument("--height", type=int, default=960, help="image height in pixels")
    parser.add_argument("--num-random-candidates", type=int, default=2000, help="number of random candidates when sampler-mode=random")
    parser.add_argument(
        "--radius-levels-m",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 5.0, 10.0],
        help="camera radius levels in meters",
    )
    parser.add_argument(
        "--azimuth-levels-deg",
        type=float,
        nargs="+",
        default=[-30.0, -15.0, 0.0, 15.0, 30.0],
        help="azimuth levels in degrees",
    )
    parser.add_argument(
        "--elevation-levels-deg",
        type=float,
        nargs="+",
        default=[-25.0, 0.0, 25.0],
        help="elevation levels in degrees",
    )
    parser.add_argument(
        "--roll-levels-deg",
        type=float,
        nargs="+",
        default=[0.0, 90.0],
        help="roll levels in degrees",
    )
    parser.add_argument(
        "--yaw-perturb-levels-deg",
        type=float,
        nargs="+",
        default=[-5.0, 0.0, 5.0],
        help="small yaw perturbation levels in degrees",
    )
    parser.add_argument(
        "--pitch-perturb-levels-deg",
        type=float,
        nargs="+",
        default=[-5.0, 0.0, 5.0],
        help="small pitch perturbation levels in degrees",
    )
    parser.add_argument("--random-seed", type=int, default=0, help="seed for random pose sampling")
    parser.add_argument(
        "--aim-point-mode",
        choices=["center", "board_anchors"],
        default="board_anchors",
        help="where cameras are aimed on the checkerboard",
    )
    parser.add_argument("--max-valid-axes", type=int, default=250, help="maximum kept camera axes to draw in 3D plots")
    parser.add_argument("--max-invalid-axes", type=int, default=100, help="maximum filtered camera axes to draw in 3D plots")
    parser.add_argument("--board-rows", type=int, default=9, help="checkerboard rows")
    parser.add_argument("--board-cols", type=int, default=9, help="checkerboard cols")
    parser.add_argument("--board-square-size", type=float, default=0.125, help="checkerboard square size in meters")
    parser.add_argument("--min-target-area-fraction", type=float, default=0.02, help="minimum projected target bbox area fraction")
    parser.add_argument("--max-target-area-fraction", type=float, default=0.55, help="maximum projected target bbox area fraction")
    parser.add_argument("--max-slant-deg", type=float, default=65.0, help="maximum allowed slant angle")
    parser.add_argument("--min-corner-spread-px", type=float, default=8.0, help="minimum nearest-neighbor projected corner spread")
    parser.add_argument("--html-only", action="store_true", help="skip PNG generation and save only interactive HTML plus summary")
    parser.add_argument("--save-full-diagnostics", action="store_true", help="store every per-candidate diagnostic entry in summary.json")
    parser.add_argument("--max-diagnostic-entries", type=int, default=500, help="maximum number of compact diagnostics to store when full diagnostics are disabled")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "results"),
        help="directory for plots",
    )
    args = parser.parse_args()

    results_root = pathlib.Path(args.output_dir)
    out_dir = create_run_output_dir(results_root, "camera_candidate_bank")

    image_size = (args.width, args.height)
    intrinsics = np.array([820.0, 815.0, image_size[0] / 2.0, image_size[1] / 2.0], dtype=float)
    if args.sampler_mode == "structured":
        bank = sdu.generate_structured_candidate_camera_bank(
            image_size=image_size,
            intrinsics=intrinsics,
            radius_levels_m=args.radius_levels_m,
            azimuth_levels_deg=args.azimuth_levels_deg,
            elevation_levels_deg=args.elevation_levels_deg,
            roll_levels_deg=args.roll_levels_deg,
            yaw_perturb_levels_deg=args.yaw_perturb_levels_deg,
            pitch_perturb_levels_deg=args.pitch_perturb_levels_deg,
            board_rows=args.board_rows,
            board_cols=args.board_cols,
            board_square_size=args.board_square_size,
            aim_point_mode=args.aim_point_mode,
            min_target_area_fraction=args.min_target_area_fraction,
            max_target_area_fraction=args.max_target_area_fraction,
            max_slant_deg=args.max_slant_deg,
            min_projected_corner_spread_px=args.min_corner_spread_px,
        )
    else:
        bank = sdu.generate_random_candidate_camera_bank(
            num_candidates=args.num_random_candidates,
            image_size=image_size,
            intrinsics=intrinsics,
            radius_range_m=(min(args.radius_levels_m), max(args.radius_levels_m)),
            azimuth_range_deg=(min(args.azimuth_levels_deg), max(args.azimuth_levels_deg)),
            elevation_range_deg=(min(args.elevation_levels_deg), max(args.elevation_levels_deg)),
            roll_range_deg=(min(args.roll_levels_deg), max(args.roll_levels_deg)),
            yaw_perturb_range_deg=(min(args.yaw_perturb_levels_deg), max(args.yaw_perturb_levels_deg)),
            pitch_perturb_range_deg=(min(args.pitch_perturb_levels_deg), max(args.pitch_perturb_levels_deg)),
            board_rows=args.board_rows,
            board_cols=args.board_cols,
            board_square_size=args.board_square_size,
            aim_point_mode=args.aim_point_mode,
            min_target_area_fraction=args.min_target_area_fraction,
            max_target_area_fraction=args.max_target_area_fraction,
            max_slant_deg=args.max_slant_deg,
            min_projected_corner_spread_px=args.min_corner_spread_px,
            seed=args.random_seed,
        )
    problem_unfiltered = CalibrationProblem(
        target_points=np.asarray(bank["target_points"], dtype=float),
        candidate_rotations=np.asarray(bank["candidate_rotations"], dtype=float),
        candidate_translations=np.asarray(bank["candidate_translations"], dtype=float),
        measurements=np.asarray(bank["measurements"], dtype=float),
        intrinsics_gt=np.asarray(bank["intrinsics"], dtype=float),
        intrinsics_init=np.asarray(bank["intrinsics"], dtype=float),
        image_size=image_size,
        pixel_noise_sigma=0.0,
        camera_is_fixed=False,
    )
    valid_mask = np.asarray(bank["valid_mask"], dtype=bool)

    unfiltered_3d_path = out_dir / "camera_bank_unfiltered_3d.png"
    filtered_3d_path = out_dir / "camera_bank_filtered_3d.png"
    unfiltered_3d_html_path = out_dir / "camera_bank_unfiltered_3d.html"
    filtered_3d_html_path = out_dir / "camera_bank_filtered_3d.html"
    unfiltered_corners_path = out_dir / "camera_bank_unfiltered_corner_projections.png"
    filtered_corners_path = out_dir / "camera_bank_filtered_corner_projections.png"
    summary_path = out_dir / "summary.json"

    if not args.html_only:
        cal_viz.plot_camera_candidate_bank_3d(
            problem_unfiltered,
            np.ones_like(valid_mask, dtype=bool),
            title="Structured Camera Candidate Bank Before Filtering",
            max_valid_axes=args.max_valid_axes,
            max_invalid_axes=0,
            save_path=str(unfiltered_3d_path),
        )
        cal_viz.plot_camera_candidate_bank_3d(
            problem_unfiltered,
            valid_mask,
            title="Structured Camera Candidate Bank After Filtering",
            max_valid_axes=args.max_valid_axes,
            max_invalid_axes=args.max_invalid_axes,
            save_path=str(filtered_3d_path),
        )
    cal_viz.plot_camera_candidate_bank_3d_plotly(
        problem_unfiltered,
        np.ones_like(valid_mask, dtype=bool),
        title="Structured Camera Candidate Bank Before Filtering",
        max_valid_axes=args.max_valid_axes,
        max_invalid_axes=0,
        save_path=str(unfiltered_3d_html_path),
    )
    cal_viz.plot_camera_candidate_bank_3d_plotly(
        problem_unfiltered,
        valid_mask,
        title="Structured Camera Candidate Bank After Filtering",
        max_valid_axes=args.max_valid_axes,
        max_invalid_axes=args.max_invalid_axes,
        save_path=str(filtered_3d_html_path),
    )
    if not args.html_only:
        cal_viz.plot_fixed_camera_projected_corner_bank(
            bank["measurements"],
            np.ones_like(valid_mask, dtype=bool),
            image_size,
            title="Projected Checkerboard Corners Before Filtering",
            save_path=str(unfiltered_corners_path),
        )
        cal_viz.plot_fixed_camera_projected_corner_bank(
            bank["measurements"],
            valid_mask,
            image_size,
            title="Projected Checkerboard Corners After Filtering",
            save_path=str(filtered_corners_path),
        )
    plt.close("all")

    diagnostics_payload = (
        bank["diagnostics"]
        if args.save_full_diagnostics
        else compact_diagnostics(bank["diagnostics"], limit=args.max_diagnostic_entries)
    )

    summary = {
        "candidate_bank_size": int(valid_mask.size),
        "valid_candidate_count": int(np.count_nonzero(valid_mask)),
        "filtered_out_count": int(np.count_nonzero(~valid_mask)),
        "valid_fraction": float(np.mean(valid_mask)),
        "radius_levels_m": list(args.radius_levels_m),
        "azimuth_levels_deg": list(args.azimuth_levels_deg),
        "elevation_levels_deg": list(args.elevation_levels_deg),
        "roll_levels_deg": list(args.roll_levels_deg),
        "yaw_perturb_levels_deg": list(args.yaw_perturb_levels_deg),
        "pitch_perturb_levels_deg": list(args.pitch_perturb_levels_deg),
        "sampler_mode": args.sampler_mode,
        "aim_point_mode": args.aim_point_mode,
        "board_rows": args.board_rows,
        "board_cols": args.board_cols,
        "board_square_size": args.board_square_size,
        "filter_parameters": bank["filter_parameters"],
        "plot_limits": {
            "max_valid_axes": args.max_valid_axes,
            "max_invalid_axes": args.max_invalid_axes,
        },
        "output_mode": {
            "html_only": bool(args.html_only),
            "save_full_diagnostics": bool(args.save_full_diagnostics),
            "max_diagnostic_entries": None if args.save_full_diagnostics else int(args.max_diagnostic_entries),
        },
        "reason_counts": bank["reason_counts"],
        "artifacts": {
            "unfiltered_3d_html": str(unfiltered_3d_html_path),
            "filtered_3d_html": str(filtered_3d_html_path),
        },
        "diagnostics_entry_count": len(diagnostics_payload),
        "diagnostics_total_count": len(bank["diagnostics"]),
        "diagnostics": diagnostics_payload,
    }
    if "random_parameters" in bank:
        summary["random_parameters"] = bank["random_parameters"]
    if not args.html_only:
        summary["artifacts"]["unfiltered_3d_plot"] = str(unfiltered_3d_path)
        summary["artifacts"]["filtered_3d_plot"] = str(filtered_3d_path)
        summary["artifacts"]["unfiltered_corner_projection_plot"] = str(unfiltered_corners_path)
        summary["artifacts"]["filtered_corner_projection_plot"] = str(filtered_corners_path)
    write_json(summary_path, summary)

    print(f"{args.sampler_mode.capitalize()} moving-camera candidate bank")
    print(f"Total candidates: {valid_mask.size}")
    print(f"Valid candidates after filtering: {np.count_nonzero(valid_mask)}")
    print(f"Filtered out: {np.count_nonzero(~valid_mask)}")
    print(f"Results saved to: {out_dir}")
    print(f"  unfiltered 3D html: {unfiltered_3d_html_path}")
    print(f"  filtered 3D html: {filtered_3d_html_path}")
    if not args.html_only:
        print(f"  unfiltered 3D plot: {unfiltered_3d_path}")
        print(f"  filtered 3D plot: {filtered_3d_path}")
        print(f"  unfiltered corner projection plot: {unfiltered_corners_path}")
        print(f"  filtered corner projection plot: {filtered_corners_path}")
    print(f"  summary saved to: {summary_path}")
