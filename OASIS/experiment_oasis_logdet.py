#!/usr/bin/env python3
import numpy as np
import os, time, json, logging
from datetime import datetime
from typing import Any, Dict, List

# ------------------------------------------------------------------ #
#  Local imports
# ------------------------------------------------------------------ #
from fim_utils   import precompute_fim_stack
from objectives  import logdet_objective, logdet_gradient
from optimization_methods import (
    frank_wolfe_optimization,
    greedy_algorithm_2,
    randomized_rounding,
    round_solution_with_cr,
    branch_and_cut_gurobi,
    compute_suboptimality_bound,
)
from graph_generation import generate_test_matrices, generate_test_problem_constraints

logger = logging.getLogger("oasis_logdet")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create a NumPy-compatible JSON encoder
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# ------------------------------------------------------------------ #
#  Main experiment function
# ------------------------------------------------------------------ #
def run_experiments(n: int = 200, m: int = 600, k: int = 40,
                    algorithms: List[str] = ("fw", "greedy", "rounding", "cr"),
                    time_limit: int = 300, seed: int = 42,
                    output_dir: str | None = None) -> Dict[str, Any]:
    logger.info("Generating synthetic instance  n=%d  m=%d  k=%d", n, m, k)

    # ----- problem generation ------------------------------------------------
    inf_mats, H0, weights = generate_test_matrices(n=n, m=m, seed=seed)
    A_cons, b_cons         = generate_test_problem_constraints(n, weights, k)

    # ----- pre‑compute stack for fast F(x) -----------------------------------
    fim_cache = precompute_fim_stack(inf_mats, H0)

    results: Dict[str, Any] = {
        "problem": dict(n=n, m=m, k=k, seed=seed),
        "algorithms": {},
    }
    x0 = np.zeros(n)

    # ================= Frank–Wolfe ==========================================
    if "fw" in algorithms:
        logger.info("▶ Frank‑Wolfe")
        t0 = time.time()
        x_fw, obj_fw, it_fw, log_fw = frank_wolfe_optimization(
            obj_func=logdet_objective,
            obj_grad=logdet_gradient,
            selection_init=x0,
            A=A_cons, b=b_cons,
            max_iterations=800,
            convergence_tol=1e-3,
            # step_size_="diminishing",
            verbose=True,
            args=(fim_cache,),                
        )
        results["algorithms"]["frank_wolfe"] = dict(
            solution=x_fw.tolist(),
            objective=float(obj_fw),
            iterations=it_fw,
            runtime=time.time() - t0,
            log=log_fw,
        )

    # ================= Greedy ===============================================
    if "greedy" in algorithms:
        logger.info("▶ Greedy")
        t0 = time.time()
        x_g, obj_g, stats_g = greedy_algorithm_2(
            obj_func=logdet_objective,
            A=A_cons, b=b_cons, n=n,
            verbose=True, timeout=time_limit,
            args=(fim_cache,),
        )
        results["algorithms"]["greedy"] = dict(
            solution=x_g.tolist(),
            objective=float(obj_g),
            runtime=time.time() - t0,
            stats=stats_g,
        )

    # ============ Randomised / CR rounding  =================================
    if "rounding" in algorithms and "frank_wolfe" in results["algorithms"]:
        logger.info("▶ Randomised rounding")
        t0 = time.time()
        x_rr, obj_rr, stats_rr = randomized_rounding(
            cont_sol=np.asarray(results["algorithms"]["frank_wolfe"]["solution"]),
            obj_func=logdet_objective,
            A=A_cons, b=b_cons,
            num_samples=120,
            verbose=True,
            args=(fim_cache,),
        )
        abs_gap, rel_gap = compute_suboptimality_bound(obj_rr, obj_fw)
        results["algorithms"]["randomized_rounding"] = dict(
            solution=x_rr.tolist(),
            objective=float(obj_rr),
            runtime=time.time() - t0,
            abs_gap=float(abs_gap),
            rel_gap=float(rel_gap),
            stats=stats_rr,
        )

    if "cr" in algorithms and "frank_wolfe" in results["algorithms"]:
        logger.info("▶ Contention‑resolution")
        t0 = time.time()
        x_cr, obj_cr, stats_cr = round_solution_with_cr(
            cont_sol=np.asarray(results["algorithms"]["frank_wolfe"]["solution"]),
            obj_func=logdet_objective,
            A=A_cons, b=b_cons,
            num_samples=15,
            verbose=True,
            args=(fim_cache,),
        )
        abs_gap, rel_gap = compute_suboptimality_bound(obj_cr, obj_fw)
        results["algorithms"]["contention_resolution"] = dict(
            solution=x_cr.tolist(),
            objective=float(obj_cr),
            runtime=time.time() - t0,
            abs_gap=float(abs_gap),
            rel_gap=float(rel_gap),
            stats=stats_cr,
        )

    # ============ (Optional) branch‑and‑cut via Gurobi =======================
    if "branch_and_cut" in algorithms:
        try:
            import gurobipy
            logger.info("▶ Branch‑and‑cut (Gurobi)")
            t0 = time.time()
            x_bc, obj_bc, stats_bc = branch_and_cut_gurobi(
                obj_func=logdet_objective,
                obj_grad=logdet_gradient,
                A=A_cons, b=b_cons, n=n,
                time_limit=time_limit, verbose=True,
                cont_solution=np.asarray(results["algorithms"]["frank_wolfe"]["solution"]),
                args=(fim_cache,),
            )
            results["algorithms"]["branch_and_cut"] = dict(
                solution=x_bc.tolist() if x_bc is not None else None,
                objective=float(obj_bc),
                runtime=time.time() - t0,
                stats=stats_bc,
            )
        except ImportError:
            logger.warning("Gurobi unavailable → skip")

    # ----- summary / persistence -------------------------------------------
    _print_summary(results)
    if output_dir:
        _save_results(results, output_dir)
    return results


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def _print_summary(res: Dict[str, Any]):
    logger.info("\nResults summary")
    logger.info("%-25s %-15s %-15s",
                "Algorithm", "Objective", "Runtime (s)")
    logger.info("-" * 57)
    for alg, r in sorted(res["algorithms"].items(),
                         key=lambda x: x[1]["objective"], reverse=True):
        logger.info("%-25s %-15.6f %-15.2f",
                    alg, r["objective"], r["runtime"])


def _save_results(res: Dict[str, Any], out_dir: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    d  = os.path.join(out_dir, f"experiment_{ts}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as fp:
        json.dump(res, fp, indent=2, cls=NumpyEncoder)
    logger.info("Saved → %s", d)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("OASIS log‑det experiment")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--m", type=int, default=600)
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-limit", type=int, default=300)
    p.add_argument("--algorithms", nargs="+",
                   choices=["fw", "greedy", "rounding", "cr",
                            "branch_and_cut", "all"],
                   default=["fw", "greedy", "rounding", "cr"])
    p.add_argument("--output", default="./results")
    args = p.parse_args()
    algs = (["fw", "greedy", "rounding", "cr", "branch_and_cut"]
            if "all" in args.algorithms else args.algorithms)
    os.makedirs(args.output, exist_ok=True)
    run_experiments(n=args.n, m=args.m, k=args.k,
                    algorithms=algs, time_limit=args.time_limit,
                    seed=args.seed, output_dir=args.output)
