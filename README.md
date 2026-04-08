# Optimal Pose Selection For Camera Intrinsics Calibration

This repository now focuses on selecting a subset of calibration views that best constrain camera intrinsics.

## Objective
For a set of candidate camera poses, we maximize the worst-direction calibration robustness:

```math
\max_{s \in \{0,1\}^N,\ \mathbf{1}^\top s = k} \lambda_{\min}(H_{cal}(s))
```

where the retained information matrix is the intrinsics Schur complement

```math
H_{cal} = H_{\theta\theta} - H_{\theta\xi} H_{\xi\xi}^{-1} H_{\xi\theta}
```

with `theta` the intrinsic parameters and `xi` the nuisance per-view pose perturbations.

## Synthetic Problem
The default experiment generates:
- a planar checkerboard target
- a bank of 1000 candidate camera poses around the target
- noisy image measurements from the ground-truth intrinsics
- an initial intrinsics estimate used for FIM linearization
- a selection of 20 poses optimized with the `lambda_min(H_cal)` objective

The intrinsic parameter vector is currently
`[fx, fy, cx, cy]`.

The default checkerboard setup is now a `9 x 9` board with `0.125` square spacing, which gives a `1.0 m x 1.0 m` board span.

For large candidate sets, the sampler uses an exact-count mode rather than a rectangular azimuth/elevation grid. The 1000 poses are distributed quasi-uniformly over the allowed viewing shell and each pose is oriented to look at the checkerboard center.
The experiments now discretize radius explicitly as multiple shells, so candidate poses can be spread across near, mid, and far viewing distances instead of a single radius.

## Run
Single synthetic problem:

```bash
python3 Experiments/main.py
```

Average across multiple synthetic problems:

```bash
python3 Experiments/main_expectation.py --num_runs 5 --num-candidate-poses 1000 --select_k 20 --radius-samples 3 --min-radius 0.6 --max-radius 1.2
```

Checkerboard spacing sweep with the same number of checkerboard points:

```bash
python3 Experiments/main_checkerboard_spacing.py --square-sizes 0.03 0.1 0.2 0.4 0.6 0.8 1.0 --num_runs 5 --num-candidate-poses 1000 --select_k 20
```

Pixel noise sweep with intrinsic uncertainty and eigenvalue curves:

```bash
python3 Experiments/main_pixel_noise.py --pixel-noises 0.1 0.5 1.0 2.0 3.0 5.0 --num_runs 5 --num-candidate-poses 1000 --select_k 20
```

Candidate azimuth-span sweep with intrinsic uncertainty and eigenvalue curves:

```bash
python3 Experiments/main_azimuth_span.py --azimuth-spans-deg 30 60 90 120 180 360 --num_runs 5 --num-candidate-poses 1000 --select_k 20
```

Fixed-camera checkerboard-motion weakness suite:

```bash
python3 Experiments/main_fixed_camera_weaknesses.py --modes limited_viewpoint_diversity narrow_azimuth_coverage little_depth_variation little_tilt_variation poor_image_plane_coverage small_target_footprint too_few_visible_points symmetric_geometry planar_pose_degeneracy high_pixel_noise
```

Candidate-count and selected-k sweep:

```bash
python3 Experiments/main_candidate_k_sweep.py --candidate-counts 10 25 50 100 200 400 700 1000 --select-ks 10 20 40 60 80 100 --num_runs 3
```

Both entry points now:
- print a selected-vs-random calibration report
- save a 3D pose-selection plot
- save a metric comparison plot
- save a per-candidate eigenvalue plot
- save visible-points plots in the sweep experiments
- create a new timestamped subfolder for each run under `results`
