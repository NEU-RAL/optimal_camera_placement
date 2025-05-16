#!/usr/bin/env python3
import os
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import scipy.sparse as sp

from fim_utils import combined_fim, precompute_fim_stack
from graph_generation import (
    generate_test_matrices,
    generate_test_problem_constraints,
    quadruplets_to_sparse
)
from optimization_methods import (
    frank_wolfe_optimization,
    greedy_algorithm_2,
    randomized_rounding,
    round_solution_with_cr,
    branch_and_cut_gurobi,
    compute_suboptimality_bound
)
from objectives import (
    reduced_laplacian,
    logdet_objective,
    logdet_gradient
)

# JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        return super().default(obj)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_d_optimality")


def run_experiments(
    n: int = 100,
    m: int = 100,
    k: int = 10,
    remove_index: int = 0,
    algorithms: List[str] = ["fw","greedy","rounding","cr","branch_and_cut"],
    time_limit: int = 300,
    seed: int = 42,
    output_dir: str = None
) -> Dict[str,Any]:
    all_quads, prior_quads, weights = generate_test_matrices(n=n, m=m, seed=seed)

    all_quads_reduced   = reduced_laplacian(all_quads, remove_index)
    prior_quads_reduced = reduced_laplacian(prior_quads, remove_index)

    dim_reduced = m - 1
    inf_mats = quadruplets_to_sparse(all_quads_reduced, n, dim_reduced)
    H0       = quadruplets_to_sparse(prior_quads_reduced, 1, dim_reduced)[0]

    cache = precompute_fim_stack(inf_mats, H0)
    A, b = generate_test_problem_constraints(n, weights, k)

    results = {
        "problem": dict(n=n, m=m, k=k, remove_index=remove_index, seed=seed),
        "algorithms": {}
    }

    def obj(x, *_) : return logdet_objective(x, cache)
    def grad(x, *_) : return logdet_gradient(x, cache)

    if "fw" in algorithms:
        t0 = time.time()
        sol_fw, obj_fw, its, log = frank_wolfe_optimization(
            obj, grad, np.zeros(n), A, b,
            max_iterations=500, convergence_tol=1e-3, verbose=True
        )
        dt = time.time() - t0
        results["algorithms"]["frank_wolfe"] = {
            "solution": sol_fw.tolist(),
            "objective": float(obj_fw),
            "iterations": its,
            "runtime": dt,
            "log": log
        }
        logger.info(f"[FW]  obj={obj_fw:.6f}  time={dt:.2f}s")

    # ==================================================================
    # Greedy
    # ==================================================================
    if "greedy" in algorithms:
        t0 = time.time()
        sol_g, obj_g, stats = greedy_algorithm_2(
            obj, A, b, n=n, verbose=True, timeout=time_limit
        )
        dt = time.time() - t0
        results["algorithms"]["greedy"] = {
            "solution": sol_g.tolist(),
            "objective": float(obj_g),
            "runtime": dt,
            "stats": stats
        }
        logger.info(f"[Greedy]  obj={obj_g:.6f}  time={dt:.2f}s")

    # ==================================================================
    # Randomized Rounding
    # ==================================================================
    if "rounding" in algorithms and "frank_wolfe" in results["algorithms"]:
        fw_sol = np.array(results["algorithms"]["frank_wolfe"]["solution"])
        t0 = time.time()
        sol_rr, obj_rr, stats = randomized_rounding(
            cont_sol=fw_sol, obj_func=obj, A=A, b=b,
            num_samples=100, verbose=True
        )
        dt = time.time() - t0
        ag, rg = compute_suboptimality_bound(
            obj_rr, results["algorithms"]["frank_wolfe"]["objective"]
        )
        results["algorithms"]["randomized_rounding"] = {
            "solution": sol_rr.tolist(),
            "objective": float(obj_rr),
            "runtime": dt,
            "abs_gap": ag,
            "rel_gap": rg,
            "stats": stats
        }
        logger.info(f"[RandRound]  obj={obj_rr:.6f}  time={dt:.2f}s")

    # ==================================================================
    # Contention Resolution
    # ==================================================================
    if "cr" in algorithms and "frank_wolfe" in results["algorithms"]:
        fw_sol = np.array(results["algorithms"]["frank_wolfe"]["solution"])
        t0 = time.time()
        sol_cr, obj_cr, stats = round_solution_with_cr(
            cont_sol=fw_sol, obj_func=obj, A=A, b=b,
            num_samples=10, verbose=True
        )
        dt = time.time() - t0
        ag, rg = compute_suboptimality_bound(
            obj_cr, results["algorithms"]["frank_wolfe"]["objective"]
        )
        results["algorithms"]["contention_resolution"] = {
            "solution": sol_cr.tolist(),
            "objective": float(obj_cr),
            "runtime": dt,
            "abs_gap": ag,
            "rel_gap": rg,
            "stats": stats
        }
        logger.info(f"[CR]  obj={obj_cr:.6f}  time={dt:.2f}s")

    # ==================================================================
    # Branch & Cut
    # ==================================================================
    if "branch_and_cut" in algorithms:
        try:
            t0 = time.time()
            bc_sol, bc_obj, bc_stats = branch_and_cut_gurobi(
                obj, grad, A, b, n=n,
                time_limit=time_limit,
                verbose=True,
                cont_solution=results["algorithms"]["frank_wolfe"]["solution"]
            )
            dt = time.time() - t0
            results["algorithms"]["branch_and_cut"] = {
                "solution": bc_sol.tolist(),
                "objective": float(bc_obj),
                "runtime": dt,
                "stats": bc_stats
            }
            logger.info(f"[B&C]  obj={bc_obj:.6f}  time={dt:.2f}s")
        except ImportError:
            logger.warning("Gurobi not available – skipping Branch & Cut")

    # summary
    logger.info("="*60)
    for name, info in results["algorithms"].items():
        logger.info(f"{name:20s}  obj={info['objective']:.6f}  time={info['runtime']:.2f}s")
    logger.info("="*60)

    # optional JSON dump
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fn = os.path.join(output_dir, f"dopt_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(fn, "w") as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)
        logger.info(f"→ saved: {fn}")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run D‑optimal reduced‑Laplacian experiments")
    p.add_argument("--n",          type=int, default=100)
    p.add_argument("--m",          type=int, default=100)
    p.add_argument("--k",          type=int, default=10)
    p.add_argument("--remove_idx", type=int, default=0)
    p.add_argument("--time_limit", type=int, default=300)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output",     type=str, default=None)
    p.add_argument("--algorithms", nargs="+",
                   default=["fw","greedy","rounding","cr","branch_and_cut"])
    args = p.parse_args()

    run_experiments(
        n=args.n,
        m=args.m,
        k=args.k,
        remove_index=args.remove_idx,
        algorithms=args.algorithms,
        time_limit=args.time_limit,
        seed=args.seed,
        output_dir=args.output
    )
