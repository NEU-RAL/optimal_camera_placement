#!/usr/bin/env python3
"""
Test script for algebraic connectivity maximization.

This script defines the objective function and gradient for maximizing the
algebraic connectivity (second–smallest eigenvalue of the Laplacian) of a graph
with edge selection, and compares several optimisation algorithms. 3‑D and 2‑D
results are visualised with Plotly.

Author: Immanuel Ampomah Mensah
"""

# ───────────────────────────────────────────────────────────────────────────────
# Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
import json
import time
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import networkx as nx
import networkx.linalg as la
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pose_graph_utils import read_g2o_file

# Local modules
from optimization_methods import (
    frank_wolfe_optimization,
    branch_and_cut_gurobi,
    greedy_algorithm_2,
    randomized_rounding,
    round_solution_with_cr,
    compute_suboptimality_bound,
)
from graph_generation import (
    generate_pose_graph,
    weight_graph_lap_from_edge_list,
    generate_3d_grid_graph,
    generate_3d_sphere_pose_graph,
)

# ───────────────────────────────────────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_algebraic_connectivity")

# ───────────────────────────────────────────────────────────────────────────────
# Data structures
# ───────────────────────────────────────────────────────────────────────────────


class Edge:
    """Simple weighted edge container."""

    def __init__(self, i: int, j: int, weight: float = 1.0) -> None:
        self.i, self.j, self.weight = i, j, weight

    def __repr__(self) -> str:  # pragma: no cover
        return f"Edge({self.i}, {self.j}, {self.weight})"


# ───────────────────────────────────────────────────────────────────────────────
# Linear‑algebra utilities
# ───────────────────────────────────────────────────────────────────────────────
# REPLACE THIS
def combined_laplacian(
    w: np.ndarray,
    L_base: sp.spmatrix,
    laplacian_e_list: List[sp.spmatrix],
    tol: float = 1e-10,
) -> sp.spmatrix:
    """Return L_base + Σ_i w_i * L_i."""
    idx = np.where(w > tol)[0]
    if len(idx) == 0:
        return L_base.copy()

    C = L_base.copy()
    for i in idx:
        C += w[i] * laplacian_e_list[i]
    return C


def find_fiedler_pair(L: sp.spmatrix) -> Tuple[float, np.ndarray]:
    """Second smallest eigenpair of Laplacian."""
    try:
        vals, vecs = eigsh(L.tocsc(), k=2, which="SM", tol=1e-2)
        order = np.argsort(vals)
        return float(vals[order[1]]), vecs[:, order[1]]
    except Exception as exc:
        logger.error("eigsh failed: %s", exc)
        return 0.0, np.zeros(L.shape[0])

# ───────────────────────────────────────────────────────────────────────────────
# Objective & gradient
# ───────────────────────────────────────────────────────────────────────────────
# REPLACE THIS


def algebraic_connectivity_objective(
    w: np.ndarray,
    L_base: sp.spmatrix,
    laplacian_e_list: List[sp.spmatrix],
) -> float:
    L = combined_laplacian(w, L_base, laplacian_e_list)
    fiedler_val, _ = find_fiedler_pair(L)
    return fiedler_val


def algebraic_connectivity_gradient(
    w: np.ndarray,
    L_base: sp.spmatrix,
    laplacian_e_list: List[sp.spmatrix],
    edge_list: List[Tuple[int, int]],
    weights: np.ndarray,
) -> np.ndarray:
    L = combined_laplacian(w, L_base, laplacian_e_list)
    _, fiedler_vec = find_fiedler_pair(L)

    grad = np.zeros_like(weights, dtype=float)
    for k, (i, j) in enumerate(edge_list):
        grad[k] = weights[k] * (fiedler_vec[i] - fiedler_vec[j]) ** 2
    return grad


# ───────────────────────────────────────────────────────────────────────────────
# Problem generation
# ───────────────────────────────────────────────────────────────────────────────
def setup_graph_problem(
    graph_type: str = "pose_graph",
    graph_import: Optional[str] = None,
    num_poses: int = 100,
    grid_size: Tuple[int, int, int] = (5, 5, 5),
    loop_closure_ratio: float = 0.2,
    selection_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Return all data needed to run an experiment."""
    if graph_import:
        odom_meas, num_poses, graph = read_g2o_file(graph_import)
        mst = nx.minimum_spanning_tree(graph)
        lc_meas = [
            Edge(u, v, d.get("weight", 1.0))
            for u, v, d in graph.edges(data=True)
            if not mst.has_edge(u, v)
        ]
    elif graph_type == "pose_graph":
        odom_meas, lc_meas, _ = generate_pose_graph(
            num_poses=num_poses, loop_closure_ratio=loop_closure_ratio, seed=seed
        )

    elif graph_type == "grid_3d":
        n_x, n_y, n_z = grid_size
        graph, _ = generate_3d_grid_graph(n_x, n_y, n_z, seed=seed)

        mst = nx.minimum_spanning_tree(graph)
        odom_meas = [Edge(u, v, d.get("weight", 1.0))
                     for u, v, d in mst.edges(data=True)]
        lc_meas = [
            Edge(u, v, d.get("weight", 1.0))
            for u, v, d in graph.edges(data=True)
            if not mst.has_edge(u, v)
        ]
        num_poses = graph.number_of_nodes()

    elif graph_type == "sphere_3d":
        odom_meas, lc_meas, _ = generate_3d_sphere_pose_graph(
            n_poses=num_poses,
            radius=10.0,
            loop_closure_ratio=loop_closure_ratio,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type}")

    # Laplacians
    L_base = weight_graph_lap_from_edge_list(odom_meas, num_poses)
    lap_e_list, weights, edge_list = [], [], []
    for m in lc_meas:
        lap_e_list.append(weight_graph_lap_from_edge_list([m], num_poses))
        weights.append(m.weight)
        edge_list.append((m.i, m.j))
    weights = np.asarray(weights, dtype=float)

    n = len(lc_meas)
    k = max(1, int(np.ceil(selection_ratio * n)))  # ensure at least one edge
    A = np.ones((1, n))
    b = np.array([[k]])

    logger.info("Problem: %d poses, %d LC, select k=%d", num_poses, n, k)

    return dict(
        graph_type=graph_type,
        num_poses=num_poses,
        num_loop_closures=n,
        k=k,
        L_base=L_base,
        laplacian_e_list=lap_e_list,
        weights=weights,
        edge_list=edge_list,
        A=A,
        b=b,
        odom_measurements=odom_meas,
        lc_measurements=lc_meas,
        grid_size=grid_size,  # for plotting
    )


# ───────────────────────────────────────────────────────────────────────────────
# JSON encoder for NumPy
# ───────────────────────────────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):  # noqa: D401
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ───────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ───────────────────────────────────────────────────────────────────────────────
def run_experiments(
    problem_setup: Dict[str, Any],
    algorithms: List[str],
    time_limit: int,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    """Benchmark all requested algorithms."""
    n = problem_setup["num_loop_closures"]
    selection_init = np.zeros(n, dtype=float)

    # Objective & gradient wrappers
    L_base = problem_setup["L_base"]
    lap_e_list = problem_setup["laplacian_e_list"]
    edge_list = problem_setup["edge_list"]
    weights = problem_setup["weights"]

    def obj_func(w): return algebraic_connectivity_objective(
        w, L_base, lap_e_list)

    def grad_func(w): return algebraic_connectivity_gradient(
        w, L_base, lap_e_list, edge_list, weights
    )

    A, b = problem_setup["A"], problem_setup["b"]

    results: Dict[str, Any] = dict(
        problem=dict(
            graph_type=problem_setup["graph_type"],
            num_poses=problem_setup["num_poses"],
            num_loop_closures=n,
            k=problem_setup["k"],
        ),
        algorithms={},
    )

    # ── Frank–Wolfe ───────────────────────────────────────────────────────────
    if "fw" in algorithms:
        logger.info("▶ Frank‑Wolfe")
        t0 = time.time()
        fw_sol, fw_obj, fw_iters, fw_log = frank_wolfe_optimization(
            obj_func=obj_func,
            obj_grad=grad_func,
            selection_init=selection_init,
            A=A,
            b=b,
            max_iterations=1000,
            convergence_tol=2e-2,
            step_size_strategy="diminishing",
            verbose=True,
        )
        results["algorithms"]["frank_wolfe"] = dict(
            solution=fw_sol.tolist(),
            objective=float(fw_obj),
            iterations=fw_iters,
            runtime=time.time() - t0,
            log=fw_log,
        )

    # ── Greedy ────────────────────────────────────────────────────────────────
    if "greedy" in algorithms:
        logger.info("▶ Greedy")
        t0 = time.time()
        g_sol, g_obj, g_stats = greedy_algorithm_2(
            obj_func=obj_func, A=A, b=b, n=n, verbose=True, timeout=time_limit
        )
        results["algorithms"]["greedy"] = dict(
            solution=g_sol.tolist(),
            objective=float(g_obj),
            runtime=time.time() - t0,
            stats=g_stats,
        )

    # ── Randomised rounding ───────────────────────────────────────────────────
    if "rounding" in algorithms and "frank_wolfe" in results["algorithms"]:
        logger.info("▶ Randomised rounding")
        t0 = time.time()
        rr_sol, rr_obj, rr_stats = randomized_rounding(
            cont_sol=np.asarray(
                results["algorithms"]["frank_wolfe"]["solution"]),
            obj_func=obj_func,
            A=A,
            b=b,
            num_samples=100,
            verbose=True,
        )
        abs_gap, rel_gap = compute_suboptimality_bound(
            rr_obj, results["algorithms"]["frank_wolfe"]["objective"]
        )
        results["algorithms"]["randomized_rounding"] = dict(
            solution=rr_sol.tolist(),
            objective=float(rr_obj),
            runtime=time.time() - t0,
            abs_gap=float(abs_gap),
            rel_gap=float(rel_gap),
            stats=rr_stats,
        )

    # ── Contention‑resolution rounding ────────────────────────────────────────
    if "cr" in algorithms and "frank_wolfe" in results["algorithms"]:
        logger.info("▶ Contention‑resolution rounding")
        t0 = time.time()
        cr_sol, cr_obj, cr_stats = round_solution_with_cr(
            cont_sol=np.asarray(
                results["algorithms"]["frank_wolfe"]["solution"]),
            obj_func=obj_func,
            A=A,
            b=b,
            num_samples=10,
            verbose=True,
        )
        abs_gap, rel_gap = compute_suboptimality_bound(
            cr_obj, results["algorithms"]["frank_wolfe"]["objective"]
        )
        results["algorithms"]["contention_resolution"] = dict(
            solution=cr_sol.tolist(),
            objective=float(cr_obj),
            runtime=time.time() - t0,
            abs_gap=float(abs_gap),
            rel_gap=float(rel_gap),
            stats=cr_stats,
        )

    # ── Branch‑and‑cut (Gurobi) ───────────────────────────────────────────────
    if "branch_and_cut" in algorithms:
        try:
            import gurobipy  # noqa: F401

            logger.info("▶ Branch‑and‑cut (Gurobi)")
            t0 = time.time()
            bc_sol, bc_obj, bc_stats = branch_and_cut_gurobi(
                obj_func=obj_func,
                obj_grad=grad_func,
                A=A,
                b=b,
                n=n,
                time_limit=time_limit,
                verbose=True,
                cont_solution=np.asarray(
                    results["algorithms"].get(
                        "frank_wolfe", {}).get("solution")
                ),
            )
            results["algorithms"]["branch_and_cut"] = dict(
                solution=bc_sol.tolist() if bc_sol is not None else None,
                objective=float(bc_obj),
                runtime=time.time() - t0,
                stats=bc_stats,
            )
        except ImportError:
            logger.warning("Gurobi unavailable → skip branch‑and‑cut")

    # ── Summary table ─────────────────────────────────────────────────────────
    logger.info("\nResults summary")
    logger.info("%-25s %-15s %-15s", "Algorithm", "Objective", "Runtime (s)")
    logger.info("-" * 57)
    for alg, res in sorted(
        results["algorithms"].items(), key=lambda x: x[1]["objective"], reverse=True
    ):
        logger.info("%-25s %-15.8f %-15.2f", alg,
                    res["objective"], res["runtime"])

    # ── Persist results ───────────────────────────────────────────────────────
    if output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        res_dir = os.path.join(output_dir, f"experiment_{ts}")
        os.makedirs(res_dir, exist_ok=True)
        with open(os.path.join(res_dir, "results.json"), "w") as fp:
            json.dump(results, fp, indent=2, cls=NumpyEncoder)
        logger.info("Saved → %s", res_dir)

    return results


# ───────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ───────────────────────────────────────────────────────────────────────────────
def plot_results(results: Dict[str, Any], out_html: str | None = None) -> None:
    """Bar charts + FW convergence."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Objective comparison",
            "Runtime comparison",
            "Frank‑Wolfe objective",
            "Frank‑Wolfe duality gap (log)",
        ],
    )

    algs = list(results["algorithms"].keys())
    objs = [results["algorithms"][a]["objective"] for a in algs]
    times = [results["algorithms"][a]["runtime"] for a in algs]

    fig.add_trace(go.Bar(x=algs, y=objs), row=1, col=1)
    fig.add_trace(go.Bar(x=algs, y=times), row=1, col=2)

    if "frank_wolfe" in results["algorithms"]:
        fwlog = results["algorithms"]["frank_wolfe"]["log"]
        fig.add_trace(go.Scatter(
            x=fwlog["iter"], y=fwlog["obj_val"]), row=2, col=1)
        fig.add_trace(
            go.Scatter(x=fwlog["iter"], y=fwlog["duality_gap"]), row=2, col=2
        )
        fig.update_yaxes(type="log", row=2, col=2)

    fig.update_layout(
        width=1200, height=800, title="Optimization results", showlegend=False
    )
    if out_html:
        pio.write_html(fig, out_html)
    else:
        fig.show()


# ───────────────────────────────────────────────────────────────────────────────
# Visualisation
# ───────────────────────────────────────────────────────────────────────────────
def visualize_graph(
    problem_setup: Dict[str, Any],
    solution: np.ndarray,
    title: str,
    out_html: str | None = None,
) -> None:
    """
    Plot the graph showing
      • odometry edges         ─ black
      • ALL loop closures      ─ light‑grey
      • selected loop closures ─ red (highlighted)

    Works for 2‑D (pose_graph) and 3‑D (grid_3d / sphere_3d) problems.
    """
    G = nx.Graph()
    G.add_nodes_from(range(problem_setup["num_poses"]))

    # Edge groups
    odom = [(e.i, e.j) for e in problem_setup["odom_measurements"]]
    all_lc = [(e.i, e.j) for e in problem_setup["lc_measurements"]]
    sel_lc = [
        (e.i, e.j)
        for s, e in zip(solution, problem_setup["lc_measurements"])
        if s > 0.5
    ]

    G.add_edges_from(odom, et="odom")
    G.add_edges_from(all_lc, et="lc")

    graph_type = problem_setup["graph_type"]
    is_3d = graph_type != "pose_graph"

    # ---- node positions -----------------------------------------------------
    if not is_3d:
        pos = nx.circular_layout(G)
    elif graph_type == "grid_3d":
        n_x, n_y, n_z = problem_setup["grid_size"]
        pos = {
            n: ((n % n_x), ((n // n_x) % n_y), n // (n_x * n_y)) for n in G.nodes
        }
    else:  # sphere_3d
        rng = np.random.default_rng(42)
        radius = 10.0
        pos = {
            n: (
                radius
                * np.sin(phi := rng.uniform(0, np.pi))
                * np.cos(theta := rng.uniform(0, 2 * np.pi)),
                radius * np.sin(phi) * np.sin(theta),
                radius * np.cos(phi),
            )
            for n in G.nodes
        }

    # ---- helper to add edges -------------------------------------------------
    def add_edges(edges, colour, width, legend_name, show_legend=True):
        if is_3d:
            xs, ys, zs = [], [], []
            for u, v in edges:
                xs += [pos[u][0], pos[v][0], np.nan]
                ys += [pos[u][1], pos[v][1], np.nan]
                zs += [pos[u][2], pos[v][2], np.nan]
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color=colour, width=width),
                    name=legend_name,
                    showlegend=show_legend,
                )
            )
        else:
            for u, v in edges:
                fig.add_trace(
                    go.Scatter(
                        x=[pos[u][0], pos[v][0]],
                        y=[pos[u][1], pos[v][1]],
                        mode="lines",
                        line=dict(color=colour, width=width),
                        name=legend_name,
                        showlegend=show_legend,
                    )
                )
                show_legend = False  # only first line gets legend entry

    # ---- Plotly figure ------------------------------------------------------
    fig = go.Figure()

    # nodes
    if is_3d:
        xs, ys, zs = zip(*[pos[n] for n in G.nodes])
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(size=6, color="lightblue", line=dict(width=0.5)),
                name="nodes",
            )
        )
    else:
        xs, ys = zip(*[pos[n] for n in G.nodes])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=10, color="lightblue", line=dict(width=1)),
                name="nodes",
            )
        )

    # edges
    add_edges(odom,   "black",   2, "odometry")
    add_edges(all_lc, "lightgrey", 1, "loop closures (all)")
    add_edges(sel_lc, "red",     3, "selected loop closures")

    # layout
    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    if is_3d:
        fig.update_layout(scene=dict(aspectmode="cube"))

    # save / show
    pio.write_html(fig, out_html) if out_html else fig.show()

# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser("Algebraic connectivity experiment")
    parser.add_argument(
        "--graph-type", choices=["pose_graph", "grid_3d", "sphere_3d"], default="pose_graph")
    parser.add_argument("--graph-import", type=str)
    parser.add_argument("--num-poses", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, nargs=3, default=[5, 5, 5])
    parser.add_argument("--loop-ratio", type=float, default=0.2)
    parser.add_argument("--selection-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["fw", "greedy", "rounding", "cr", "branch_and_cut", "all"],
        default=["fw", "greedy", "rounding", "cr", "branch_and_cut"],
    )
    parser.add_argument("--output", default="./results")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    algs = (
        ["fw", "greedy", "rounding", "cr", "branch_and_cut"]
        if "all" in args.algorithms
        else args.algorithms
    )

    os.makedirs(args.output, exist_ok=True)

    problem = setup_graph_problem(
        graph_type=args.graph_type,
        graph_import=args.graph_import,
        num_poses=args.num_poses,
        grid_size=tuple(args.grid_size),
        loop_closure_ratio=args.loop_ratio,
        selection_ratio=args.selection_ratio,
        seed=args.seed,
    )

    results = run_experiments(
        problem, algs, args.time_limit, output_dir=args.output)
    plot_results(results, out_html=os.path.join(args.output, "summary.html"))

    if args.visualize:
        best_alg = max(
            results["algorithms"], key=lambda a: results["algorithms"][a]["objective"])
        best_sol = np.asarray(results["algorithms"][best_alg]["solution"])
        viz_dir = os.path.join(args.output, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        visualize_graph(
            problem,
            best_sol,
            f"Selected loop closures – {best_alg}",
            out_html=os.path.join(viz_dir, "graph.html"),
        )
        logger.info("Graph saved to %s", viz_dir)


if __name__ == "__main__":
    main()
