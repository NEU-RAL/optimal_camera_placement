#!/usr/bin/env python
"""
Standalone script for solving a relaxed sensor selection problem
using an outer approximation method with Gurobi.

The original problem is to maximize:
    f(x) = λ_min(H0 + Σ_j x_j M_j)
subject to:
    Ax ≤ b,  x ∈ {0,1}^n.

We use a continuous variable t and add cuts of the form:
    t ≤ f(x^k) + ⟨g^k, x - x^k⟩,
where g^k is a subgradient at x^k. To ensure the master problem is bounded,
an initial feasible point must be provided (here we use x₀ = (1,1,0) in our toy example).
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import gurobipy as gp
from gurobipy import GRB
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("outer_approximation")

# --- Objective and Subgradient Functions ---

def min_eigenvalue_objective(x: np.ndarray, inf_mats: List[sp.spmatrix],
                             H0: sp.spmatrix) -> float:
    """
    Compute f(x) = λ_min(H0 + Σ_j x_j M_j).
    """
    F = H0.copy()
    for xi, M in zip(x, inf_mats):
        F += xi * M
    # Add tiny regularization for stability
    F += 1e-10 * sp.eye(F.shape[0])
    try:
        eig_vals, _ = eigsh(F.tocsc(), k=1, which='SA', tol=1e-4)
        return eig_vals[0]
    except Exception as e:
        logger.error(f"Error computing eigenvalues: {str(e)}")
        return -1e10

def min_eigenvalue_gradient(x: np.ndarray, inf_mats: List[sp.spmatrix],
                            H0: sp.spmatrix) -> np.ndarray:
    """
    Compute a subgradient of f(x) = λ_min(H0 + Σ_j x_j M_j) at x.
    The subgradient is given (almost everywhere) by:
      g_j = -v^T M_j v,
    where v is the eigenvector associated with the minimum eigenvalue.
    """
    F = H0.copy()
    for xi, M in zip(x, inf_mats):
        F += xi * M
    F += 1e-10 * sp.eye(F.shape[0])
    n = F.shape[0]
    try:
        eig_vals, eig_vecs = eigsh(F.tocsc(), k=1, which='SA', tol=1e-4)
        v = eig_vecs.flatten()
    except Exception as e:
        from scipy.sparse.linalg import lobpcg
        v0 = np.random.rand(n)
        v0 = v0 / np.linalg.norm(v0)
        X = np.zeros((n,1))
        X[:,0] = v0
        eig_vals, eig_vecs = lobpcg(F, X, largest=False, maxiter=500, tol=1e-6)
        v = eig_vecs[:,0]
    g = np.empty(len(inf_mats))
    for i, M in enumerate(inf_mats):
        g[i] = -float(v.T @ (M.dot(v)))
    return g

# --- Outer Approximation Method ---
def outer_approximation_sensor_selection(
    inf_mats: List[sp.spmatrix],
    H0: sp.spmatrix,
    A: np.ndarray,
    b: np.ndarray,
    max_iter: int = 20,
    tol: float = 1e-4,
    verbose: bool = False,
    initial_x: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Solve the sensor selection problem via outer approximation.
    
    We wish to solve:
       max { f(x) = λ_min(H0 + Σ_j x_j M_j) : Ax ≤ b, x ∈ {0,1}^n }.
       
    We introduce a continuous variable t and iteratively add cuts:
       t ≤ f(x^k) + ⟨g^k, x - x^k⟩,
    where g^k is a subgradient at x^k.
    
    An initial feasible solution must be provided so that the master problem is bounded.
    """
    n = len(inf_mats)
    if initial_x is None:
        # For our toy example, we choose x0 = (1,1,0)
        initial_x = np.zeros(n)
        # Assume n >= 2; here we set first two sensors to 1.
        initial_x[0] = 1
        initial_x[1] = 1
    
    # Prepare storage for cuts: each cut is (x^k, f(x^k), g^k)
    cuts = []
    best_lower_bound = -np.inf
    best_upper_bound = np.inf
    iter_stats = {"iterations": 0, "master_obj_history": []}
    
    for it in range(max_iter):
        iter_stats["iterations"] = it + 1
        # Build master MILP
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 1 if verbose else 0)
            env.start()
            with gp.Model("OuterApproximation", env=env) as model:
                # Decision variables: x in {0,1}^n and continuous t
                x_vars = model.addVars(n, vtype=GRB.BINARY, name="x")
                t_var = model.addVar(lb=-GRB.INFINITY, name="t")
                # Original constraints: A x ≤ b
                for i in range(A.shape[0]):
                    expr = gp.LinExpr()
                    for j in range(n):
                        expr.add(x_vars[j], A[i, j])
                    model.addConstr(expr <= b[i][0], name=f"orig_constr_{i}")
                # Add all outer approximation cuts
                for idx, (xk, fk, gk) in enumerate(cuts):
                    expr = fk
                    for j in range(n):
                        expr += gk[j] * (x_vars[j] - xk[j])
                    model.addConstr(t_var <= expr, name=f"cut_{idx}")
                # Set objective: maximize t
                model.setObjective(t_var, GRB.MAXIMIZE)
                model.optimize()
                if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
                    logger.error("Master problem not solved optimally.")
                    return None, float('-inf'), {"status": model.status}
                master_obj = model.ObjVal
                iter_stats["master_obj_history"].append(master_obj)
                candidate_x = np.array([round(x_vars[j].X) for j in range(n)])
                candidate_t = t_var.X
        # Evaluate candidate solution
        f_candidate = min_eigenvalue_objective(candidate_x, inf_mats, H0)
        g_candidate = min_eigenvalue_gradient(candidate_x, inf_mats, H0)
        best_lower_bound = max(best_lower_bound, f_candidate)
        best_upper_bound = min(best_upper_bound, master_obj)
        gap = best_upper_bound - best_lower_bound
        if verbose:
            logger.info(f"Iteration {it+1}: Master obj = {master_obj:.6f}, "
                        f"f(candidate) = {f_candidate:.6f}, gap = {gap:.6f}")
        if gap <= tol:
            logger.info("Convergence achieved (gap <= tol).")
            return candidate_x, f_candidate, {
                "iterations": it+1,
                "best_lower_bound": best_lower_bound,
                "best_upper_bound": best_upper_bound,
                "gap": gap,
                "iter_stats": iter_stats
            }
        # Add new cut
        cuts.append((candidate_x, f_candidate, g_candidate))
    
    logger.info("Maximum iterations reached in outer approximation.")
    return candidate_x, f_candidate, {
        "iterations": max_iter,
        "best_lower_bound": best_lower_bound,
        "best_upper_bound": best_upper_bound,
        "gap": gap,
        "iter_stats": iter_stats
    }

# --- Main Section ---
def main():
    # For our toy example: m = 2 (each matrix is 2x2), n = 3 candidate matrices.
    m = 2
    n = 3
    # Prior matrix H0: 2x2 identity.
    H0 = sp.csc_matrix(np.eye(m))
    # Candidate matrices (symmetric, PSD)
    M1 = sp.csc_matrix(np.array([[2.0, 0.0],
                                 [0.0, 1.0]]))
    M2 = sp.csc_matrix(np.array([[1.0, 0.5],
                                 [0.5, 1.5]]))
    M3 = sp.csc_matrix(np.array([[1.2, 0.1],
                                 [0.1, 1.0]]))
    inf_mats = [M1, M2, M3]
    
    # Constraint: sum(x) ≤ 2 (i.e. at most two matrices may be chosen)
    A = np.array([[1, 1, 1]])
    b = np.array([[2]])
    
    # Use an initial point x0 = (1,1,0) to generate a proper first cut.
    initial_x = np.array([1, 1, 0])
    
    x_sol, f_val, stats = outer_approximation_sensor_selection(
        inf_mats, H0, A, b, max_iter=20, tol=1e-4, verbose=True, initial_x=initial_x
    )
    
    print("\n--- Outer Approximation Results ---")
    print("Selected vector (binary):", x_sol)
    print("Objective value (min eigenvalue):", f_val)
    print("Stats:", stats)

if __name__ == "__main__":
    main()
