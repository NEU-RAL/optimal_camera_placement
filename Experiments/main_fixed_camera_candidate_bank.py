import json
import argparse
import pathlib
import sys
from datetime import datetime

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from optimal_camera_placement.DataGenerator import sim_data_utils as sdu
from optimal_camera_placement.OASIS import calibration_visualize as cal_viz


def create_run_output_dir(base_dir: pathlib.Path, run_name: str) -> pathlib.Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"{run_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and visualize a discrete fixed-camera checkerboard candidate bank."
    )
    parser.add_argument("--width", type=int, default=1280, help="image width in pixels")
    parser.add_argument("--height", type=int, default=960, help="image height in pixels")
    parser.add_argument(
        "--depths-m",
        type=float,
        nargs="+",
        default=[2.0, 2.5, 3.0, 3.5],
        help="candidate checkerboard depths in meters",
    )
    parser.add_argument("--grid-rows", type=int, default=3, help="image-region grid rows")
    parser.add_argument("--grid-cols", type=int, default=3, help="image-region grid cols")
    parser.add_argument(
        "--tilts-deg",
        type=float,
        nargs="+",
        default=[0.0, -15.0, 15.0, -30.0, 30.0],
        help="board tilt values in degrees",
    )
    parser.add_argument(
        "--rolls-deg",
        type=float,
        nargs="+",
        default=[0.0, 45.0, 90.0, 135.0],
        help="board roll values in degrees",
    )
    parser.add_argument("--board-rows", type=int, default=9, help="checkerboard rows")
    parser.add_argument("--board-cols", type=int, default=9, help="checkerboard cols")
    parser.add_argument("--board-square-size", type=float, default=0.125, help="checkerboard square size in meters")
    parser.add_argument("--min-target-area-fraction", type=float, default=0.02, help="minimum projected target bbox area fraction")
    parser.add_argument("--max-target-area-fraction", type=float, default=0.45, help="maximum projected target bbox area fraction")
    parser.add_argument("--max-slant-deg", type=float, default=55.0, help="maximum allowed board slant angle")
    parser.add_argument("--min-corner-spread-px", type=float, default=12.0, help="minimum nearest-neighbor projected corner spread")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "results"),
        help="directory for plots",
    )
    args = parser.parse_args()

    results_root = pathlib.Path(args.output_dir)
    out_dir = create_run_output_dir(results_root, "fixed_camera_candidate_bank")

    image_size = (args.width, args.height)
    intrinsics = np.array([820.0, 815.0, image_size[0] / 2.0, image_size[1] / 2.0], dtype=float)
    bank = sdu.generate_discrete_candidate_checkerboard_bank_fixed_camera(
        image_size=image_size,
        intrinsics=intrinsics,
        depths_m=args.depths_m,
        region_grid_shape=(args.grid_rows, args.grid_cols),
        tilts_deg=args.tilts_deg,
        rolls_deg=args.rolls_deg,
        board_rows=args.board_rows,
        board_cols=args.board_cols,
        board_square_size=args.board_square_size,
        min_target_area_fraction=args.min_target_area_fraction,
        max_target_area_fraction=args.max_target_area_fraction,
        max_slant_deg=args.max_slant_deg,
        min_projected_corner_spread_px=args.min_corner_spread_px,
    )
    problem_unfiltered = sdu.build_fixed_camera_problem_from_candidate_bank(
        bank,
        image_size=image_size,
        pixel_noise_sigma=0.0,
        seed=0,
        filtered_only=False,
    )

    unfiltered_3d_path = out_dir / "candidate_bank_unfiltered_3d.png"
    filtered_3d_path = out_dir / "candidate_bank_filtered_3d.png"
    unfiltered_image_path = out_dir / "candidate_bank_unfiltered_image_plane.png"
    filtered_image_path = out_dir / "candidate_bank_filtered_image_plane.png"
    unfiltered_corners_path = out_dir / "candidate_bank_unfiltered_corner_projections.png"
    filtered_corners_path = out_dir / "candidate_bank_filtered_corner_projections.png"
    summary_path = out_dir / "summary.json"

    valid_mask = np.asarray(bank["valid_mask"], dtype=bool)
    cal_viz.plot_fixed_camera_candidate_bank_3d(
        problem_unfiltered,
        np.ones_like(valid_mask, dtype=bool),
        title="Fixed-Camera Candidate Bank Before Filtering",
        save_path=str(unfiltered_3d_path),
    )
    cal_viz.plot_fixed_camera_candidate_bank_3d(
        problem_unfiltered,
        valid_mask,
        title="Fixed-Camera Candidate Bank After Filtering",
        save_path=str(filtered_3d_path),
    )
    cal_viz.plot_fixed_camera_candidate_bank_image_plane(
        bank["diagnostics"],
        image_size,
        np.ones_like(valid_mask, dtype=bool),
        title="Projected Candidate Centers Before Filtering",
        save_path=str(unfiltered_image_path),
    )
    cal_viz.plot_fixed_camera_candidate_bank_image_plane(
        bank["diagnostics"],
        image_size,
        valid_mask,
        title="Projected Candidate Centers After Filtering",
        save_path=str(filtered_image_path),
    )
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

    summary = {
        "candidate_bank_size": int(valid_mask.size),
        "valid_candidate_count": int(np.count_nonzero(valid_mask)),
        "filtered_out_count": int(np.count_nonzero(~valid_mask)),
        "valid_fraction": float(np.mean(valid_mask)),
        "depths_m": list(args.depths_m),
        "image_regions": f"{args.grid_rows}x{args.grid_cols} grid",
        "tilts_deg": list(args.tilts_deg),
        "rolls_deg": list(args.rolls_deg),
        "board_rows": args.board_rows,
        "board_cols": args.board_cols,
        "board_square_size": args.board_square_size,
        "filter_parameters": bank["filter_parameters"],
        "reason_counts": bank["reason_counts"],
        "artifacts": {
            "unfiltered_3d_plot": str(unfiltered_3d_path),
            "filtered_3d_plot": str(filtered_3d_path),
            "unfiltered_image_plane_plot": str(unfiltered_image_path),
            "filtered_image_plane_plot": str(filtered_image_path),
            "unfiltered_corner_projection_plot": str(unfiltered_corners_path),
            "filtered_corner_projection_plot": str(filtered_corners_path),
        },
        "diagnostics": bank["diagnostics"],
    }
    write_json(summary_path, summary)

    print("Fixed-camera discrete candidate bank sampler")
    print("Bank definition:")
    print(f"  depths: {list(args.depths_m)} m")
    print(f"  image regions: {args.grid_rows}x{args.grid_cols} grid")
    print(f"  tilts: {list(args.tilts_deg)} deg")
    print(f"  rolls: {list(args.rolls_deg)} deg")
    print(f"  board: {args.board_rows}x{args.board_cols}, square size {args.board_square_size} m")
    print(f"Total candidates: {valid_mask.size}")
    print(f"Valid candidates after filtering: {np.count_nonzero(valid_mask)}")
    print(f"Filtered out: {np.count_nonzero(~valid_mask)}")
    print(f"Results saved to: {out_dir}")
    print(f"  unfiltered 3D plot: {unfiltered_3d_path}")
    print(f"  filtered 3D plot: {filtered_3d_path}")
    print(f"  unfiltered image-plane plot: {unfiltered_image_path}")
    print(f"  filtered image-plane plot: {filtered_image_path}")
    print(f"  unfiltered corner projection plot: {unfiltered_corners_path}")
    print(f"  filtered corner projection plot: {filtered_corners_path}")
    print(f"  summary saved to: {summary_path}")
