from __future__ import annotations

from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple, Any, Union

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.sparse.linalg import eigsh, lobpcg
import logging, time, warnings
import warnings

# suppress LOBPCG iteration‐limit warnings
warnings.filterwarnings(
    "ignore",
    message="Exited at iteration.*not reaching the requested tolerance.*",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="Exited postprocessing with accuracies.*",
    category=UserWarning
)

# ---------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("opt_methods")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------
#  Enumerations
# ---------------------------------------------------------------------
class Metric(Enum):
    LOGDET   = auto()
    MIN_EIG  = auto()
    MSE      = auto()


class StepSize(Enum):
    FIXED        = auto()
    DIMINISHING  = auto()
    BACKTRACKING = auto()

# ---------------------------------------------------------------------
#  Step‑size utilities
# ---------------------------------------------------------------------
def _diminishing(k: int, gamma: float = 2.0) -> float:
    """α_k = γ / (k+2)"""
    return gamma / (k + 2)


def _backtracking(
        obj: Callable, grad: Callable,
        x: np.ndarray, d: np.ndarray,
        c: float = 1e-4, beta: float = .5,
        args: tuple = (), max_iter: int = 30) -> float:
    """Armijo rule for *maximisation* (note ≥ inequality)."""
    f_x   = obj(x, *args)
    g_dot = grad(x, *args) @ d
    α     = 1.0
    for _ in range(max_iter):
        if obj(x + α*d, *args) >= f_x + c*α*g_dot:
            return α
        α *= beta
    return α


# ---------------------------------------------------------------------
#  Frank–Wolfe optimiser
# ---------------------------------------------------------------------
def frank_wolfe_optimization(
        obj_func: Callable,
        obj_grad: Callable,
        selection_init: np.ndarray,
        A: np.ndarray, b: np.ndarray,
        max_iterations: int = 1000,
        convergence_tol: float = 2e-2,
        step_rule: StepSize = StepSize.DIMINISHING,
        verbose: bool = True,
        args: tuple = ()) -> Tuple[np.ndarray, float, int, Dict[str, List]]:

    log: Dict[str, List] = {k: [] for k in
        ("iter", "obj_val", "step_size", "duality_gap", "grad_norm", "time_per_iter")}

    x = selection_init.astype(float)
    overall_start = time.time()  

    for k in range(max_iterations):
        t0 = time.time()
        f  = obj_func(x, *args)
        g  = obj_grad(x, *args)

        # LMO
        s = _solve_lmo(-g, A, b)
        if s is None:
            logger.error("LMO failed – abort.")
            break

        d   = s - x
        gap = g @ d
        g2  = np.linalg.norm(g)

        # check convergence
        if gap < convergence_tol * max(1, abs(f)):
            if verbose:
                logger.info(f"Converged at iteration {k} (duality gap {gap:.2e}).")
            break

        # step size
        if step_rule is StepSize.FIXED:
            α = .1
        elif step_rule is StepSize.DIMINISHING:
            α = _diminishing(k)
        else:
            α = _backtracking(obj_func, obj_grad, x, d, args=args)

        x += α * d
        x = np.clip(x, 0, 1)

        iter_time = time.time() - t0
        for key, val in zip(
            ("iter","obj_val","step_size","duality_gap","grad_norm","time_per_iter"),
            (k, f, α, gap, g2, iter_time)
        ):
            log[key].append(val)

        if verbose:
            logger.info(f"it={k:4d}  f={f:9.3e}  α={α:.3f}  gap={gap:5.2e}  t_iter={iter_time:.3f}s")

    total_time = time.time() - overall_start  # ← compute total
    if verbose:
        logger.info(f"Frank–Wolfe total runtime: {total_time:.3f} s over {len(log['iter'])} iterations")

    # final objective
    final_obj = obj_func(x, *args)
    return x, final_obj, len(log["iter"]), log

def _solve_lmo(grad: np.ndarray,
               A: Optional[Union[np.ndarray, sp.spmatrix]],
               b: np.ndarray) -> Optional[np.ndarray]:
    n = len(grad)
    bounds = [(0,1)]*n
    if A is not None and sp.issparse(A):
        A = A.tocsr()

    res = linprog(c=grad, A_ub=A, b_ub=b, bounds=bounds, method="highs", options={"presolve": True})
    if res.success:
        return res.x
    logger.warning("LMO linprog failed: %s", res.message)
    return None

# ======================================================================
# Gurobi Branch and Cut Implementation
# ======================================================================

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    warnings.warn("Gurobi not available, branch and cut optimization will be disabled")

def branch_and_cut_gurobi(
    obj_func: Callable,
    obj_grad: Callable,
    A: np.ndarray,
    b: np.ndarray,
    n: int,
    time_limit: int = 600,
    mip_gap: float = 0.0,
    verbose: bool = False,
    cont_solution: np.ndarray = None,
    args: tuple = ()
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Solves matrix selection problem using branch and cut with Gurobi.
    
    Args:
        obj_func: Objective function to maximize
        obj_grad: Gradient function to compute subgradients/cuts
        A: Inequality constraint matrix
        b: Inequality constraint bounds
        n: Dimension of the solution vector
        time_limit: Time limit in seconds (default: 600)
        mip_gap: MIP gap for termination (default: 0.01)
        verbose: Whether to print detailed output
        cont_solution: Continuous solution to use for initialization (optional)
        args: Additional arguments to pass to objective and gradient functions
    
    Returns:
        Tuple containing:
        - Selected binary vector
        - Objective value
        - Solution statistics
    """
    if not GUROBI_AVAILABLE:
        logger.error("Gurobi is not available. Cannot perform branch and cut optimization.")
        return None, float('-inf'), {"status": "Gurobi not available"}
    
    # Create a Gurobi model
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 1 if verbose else 0)
        env.start()
        
        with gp.Model("Branch_And_Cut", env=env) as model:
            # Set parameters
            model.setParam('TimeLimit', time_limit)
            model.setParam('MIPGap', mip_gap)
            model.setParam('LazyConstraints', 1)
            model.setParam('Cuts', 2)  # Aggressive cut generation
            
            # Create binary decision variables
            x = model.addVars(n, vtype=GRB.BINARY, name="x")
            
            # Add constraints
            for i in range(A.shape[0]):
                expr = gp.LinExpr()
                for j in range(n):
                    expr.add(x[j], A[i, j])
                model.addConstr(expr <= b[i], f"constraint_{i}")

            # Add auxiliary variable for objective
            t = model.addVar(lb=-GRB.INFINITY, name="t")
            
            # Set the objective to maximize t
            model.setObjective(t, GRB.MAXIMIZE)
            
            # Add initial cut from the zero solution
            if verbose:
                logger.info("Adding initial cut from zero vector")
                
            # Compute objective value and gradient for the zero vector
            zero_vec = np.zeros(n)
            obj_val_0 = obj_func(zero_vec, *args)
            grad_0 = obj_grad(zero_vec, *args)
            
            # Add the cut: t <= obj_val_0 + sum_j grad_0[j] * x[j]
            cut_expr0 = gp.LinExpr()
            for j in range(n):
                cut_expr0.add(x[j], grad_0[j])
            model.addConstr(t <= obj_val_0 + cut_expr0, "initial_cut_zero")
            
            # If we have a continuous solution, use it to generate an upper bound
            if cont_solution is not None:
                if verbose:
                    logger.info("Using continuous solution for initial bound")
                
                # Compute objective value and gradient for continuous solution
                obj_val_cont = obj_func(cont_solution, *args)
                grad_cont = obj_grad(cont_solution, *args)
                
                # Add cut: t <= obj_val_cont + sum_j grad_cont[j] * (x[j] - cont_solution[j])
                rhs_cont = obj_val_cont - np.dot(grad_cont, cont_solution)
                cut_expr_cont = gp.LinExpr()
                for j in range(n):
                    cut_expr_cont.add(x[j], grad_cont[j])
                
                model.addConstr(t <= rhs_cont + cut_expr_cont, "initial_cut_cont")
                
                # Set an upper bound for t based on continuous solution
                t.ub = obj_val_cont
            
            def eigenvalue_callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    # Get current integer solution
                    x_vals = model.cbGetSolution([x[j] for j in range(n)])
                    
                    # Convert to NumPy array
                    x_vals = np.array(x_vals)
                    
                    t_val = model.cbGetSolution(t)
                    
                    # Compute the actual objective value and gradient at this solution
                    actual_obj = obj_func(x_vals, *args)
                    grad = obj_grad(x_vals, *args)
                    
                    # If the current t is greater than the actual objective (with some tolerance),
                    # add a cutting plane
                    if t_val > actual_obj + 1e-6:
                        # Add cut: t <= actual_obj + sum_j grad[j] * (x[j] - x_vals[j])
                        rhs = actual_obj - np.dot(grad, x_vals)
                        
                        cut_expr = gp.LinExpr()
                        for j in range(n):
                            cut_expr.add(x[j], grad[j])
                        
                        model.cbLazy(t <= rhs + cut_expr)
                        
                        if verbose:
                            violation = t_val - actual_obj
                            logger.debug(f"Added cut at integer solution with violation {violation:.6f}")
                
                elif where == GRB.Callback.MIPNODE:
                    # Only add cuts at nodes where we have an optimal relaxation
                    if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                        return
                    
                    # Get the relaxation solution at this node
                    x_vals = model.cbGetNodeRel([x[j] for j in range(n)])
                    
                    # Convert to NumPy array
                    x_vals = np.array(x_vals)
                    
                    t_val = model.cbGetNodeRel(t)
                    
                    # Skip if the solution is nearly binary (let MIPSOL handle it)
                    if all(xi < 0.1 or xi > 0.9 for xi in x_vals):
                        return
                    
                    # Compute the objective value and gradient at this relaxed point
                    actual_obj = obj_func(x_vals, *args)
                    grad = obj_grad(x_vals, *args)
                    
                    # Check if we need to add a cut
                    if t_val > actual_obj + 1e-6:
                        # Add the cut
                        rhs = actual_obj - np.dot(grad, x_vals)
                        
                        cut_expr = gp.LinExpr()
                        for j in range(n):
                            cut_expr.add(x[j], grad[j])
                        
                        model.cbCut(t <= rhs + cut_expr)
                        
                        if verbose:
                            violation = t_val - actual_obj
                            logger.debug(f"Added cut at node relaxation with violation {violation:.6f}")
            
            # Optimize with callback
            if verbose:
                logger.info("Starting branch and cut optimization")
            
            model.optimize(eigenvalue_callback)
            
            # Get solution
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                # Extract solution
                x_solution = np.zeros(n)
                for j in range(n):
                    x_solution[j] = round(x[j].X)  # Round to handle potential numerical issues
                
                # Compute actual objective with the final solution
                obj_val = obj_func(x_solution, *args)
                
                # Collect solution statistics
                stats = {
                    "status": model.status,
                    "runtime": model.Runtime,
                    "mip_gap": model.MIPGap if hasattr(model, 'MIPGap') else None,
                    "obj_bound": model.ObjBound if hasattr(model, 'ObjBound') else None,
                    "num_nodes": model.NodeCount,
                    "num_cuts": model.NumVars - n - 1  # Approximate measure of cuts added
                }
                
                if verbose:
                    logger.info(f"Branch and cut completed with status {model.status}")
                    logger.info(f"Objective value: {obj_val:.6f}")
                    logger.info(f"Runtime: {model.Runtime:.2f} seconds")
                    
                return x_solution, obj_val, stats
            else:
                logger.error(f"Gurobi optimization failed with status {model.status}")
                return None, float('-inf'), {"status": model.status}

def greedy_algorithm_2(
    obj_func: Callable,
    A: np.ndarray,
    b: np.ndarray,
    n: int,
    verbose: bool = True,
    timeout: Optional[float] = None,
    args: tuple = ()
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Implementation of Algorithm 2 for maximizing functions under multiple constraints.
    
    This algorithm greedily maximizes a function subject to constraints Ax ≤ b
    by selecting elements based on their marginal gain to cost ratio.
    
    Args:
        obj_func: Objective function to maximize (takes numpy array and returns float)
        A: Constraint matrix
        b: Constraint bounds vector
        n: Problem dimension (number of elements)
        verbose: Whether to print progress information
        timeout: Maximum execution time in seconds (None means no limit)
        args: Additional arguments to pass to objective function
        
    Returns:
        Tuple containing:
        - Selected binary vector
        - Objective value
        - Dictionary with statistics and status information
    """
    start_time = time.time()
    m = A.shape[0]  # Number of constraints
    
    # Statistics collection
    stats = {
        "iterations": 0,
        "obj_evaluations": 0,
        "timed_out": False,
        "runtime": 0.0,
        "obj_history": []
    }
    
    # Initialize with empty selection
    current_selection = np.zeros(n, dtype=int)
    
    # Precompute constraint values for empty selection
    constraint_values = np.zeros(A.shape[0])
    
    # Compute initial objective
    current_obj = obj_func(current_selection, *args)
    stats["obj_evaluations"] += 1
    stats["obj_history"].append(current_obj)
    
    if verbose:
        logger.info(f"Starting greedy selection with initial objective: {current_obj:.6f}")
        if timeout is not None:
            logger.info(f"Timeout set to {timeout:.1f} seconds")
    
    # Set of candidate indices (elements that can potentially be added)
    W = set(range(n))
    
    # Main greedy selection loop
    while W:
        # Check timeout
        current_time = time.time()
        if timeout is not None and (current_time - start_time) > timeout:
            if verbose:
                logger.warning(f"Greedy selection timed out after {current_time - start_time:.1f} seconds")
            stats["timed_out"] = True
            break
            
        stats["iterations"] += 1
        
        # Find the best element and constraint pair
        best_ratio = -float('inf')
        best_element = None
        best_i = None
        
        for v in W:
            # Calculate marginal gain in objective function
            test_selection = current_selection.copy()
            test_selection[v] = 1
            
            new_obj = obj_func(test_selection, *args)
            stats["obj_evaluations"] += 1
            
            delta_f = new_obj - current_obj
            
            # Skip if no improvement in objective
            if delta_f <= 0:
                continue
                
            # Find the best ratio across all constraints where delta_h_i > 0
            for i in range(m):
                delta_h_i = A[i, v]
                
                # Skip if element has no impact on this constraint or impact is negative
                if delta_h_i <= 0:
                    continue
                
                # Calculate ratio: marginal gain / marginal cost
                ratio = delta_f / delta_h_i
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_element = v
                    best_i = i
                    
        # If no improvement found, exit
        if best_element is None:
            break
        
        # Check if adding best_element violates any constraint
        will_violate = False
        for i in range(m):
            b_value = float(b[i]) if isinstance(b[i], np.ndarray) else float(b[i])
            if constraint_values[i] + A[i, best_element] > b_value:
                will_violate = True
                break
                
        if not will_violate:
            # Add element to solution
            current_selection[best_element] = 1
            W.remove(best_element)
            
            # Update objective
            current_obj = obj_func(current_selection, *args)
            stats["obj_evaluations"] += 1
            stats["obj_history"].append(current_obj)
            
            # Update constraint values
            for i in range(m):
                constraint_values[i] += A[i, best_element]
            
            if verbose and stats["iterations"] % 10 == 0:
                logger.info(f"Iteration {stats['iterations']}: Added element {best_element}")
                logger.info(f"New objective: {current_obj:.6f}")
        else:
            # Element violates constraints - remove from consideration
            W.remove(best_element)
            if verbose and stats["iterations"] % 10 == 0:
                logger.info(f"Element {best_element} violates constraints, removed from consideration")
    
    # Compute final statistics
    stats["runtime"] = time.time() - start_time
    stats["selected_count"] = np.sum(current_selection)
    stats["final_constraints"] = constraint_values.tolist()
    
    if verbose:
        logger.info(f"Greedy selection complete:")
        logger.info(f"Final objective value: {current_obj:.6f}")
        logger.info(f"Selected {stats['selected_count']} elements")
        logger.info(f"Runtime: {stats['runtime']:.2f} seconds")
        logger.info(f"Objective evaluations: {stats['obj_evaluations']}")
        
        # Report constraint satisfaction
        for i in range(m):
            b_value = float(b[i]) if isinstance(b[i], np.ndarray) else float(b[i])
            utilization = constraint_values[i] / b_value * 100 if b_value != 0 else 0
            logger.info(f"Constraint {i}: {constraint_values[i]:.4f} / {b_value:.4f} ({utilization:.1f}%)")
    
    return current_selection, current_obj, stats


# ======================================================================
# Variance Information Calculation
# ======================================================================

def compute_variance_info(A: np.ndarray, x_star: np.ndarray):
    """
    For each row i of A:
      - Print:
          E[s_i], Var(s_i), ratio = sqrt(Var) / E
      - Collect the same info into a list of dicts for JSON.

    Parameters
    ----------
    A : np.ndarray, shape (p, m)
        The constraint matrix (each row = one constraint).
    x_star : np.ndarray, shape (m,)
        The solution vector (entries in [0,1]).

    Returns
    -------
    A list of dictionaries, each like {"E": ..., "Var": ..., "ratio": ...},
    which is suitable for JSON serialization.
    """
    p, m = A.shape
    variance_info_list = []

    for i in range(p):
        a_i = A[i, :]
        E = np.dot(a_i, x_star)
        Var = np.sum((a_i**2) * x_star * (1 - x_star))
        
        if E > 1e-12:
            ratio = np.sqrt(Var) / E
        else:
            ratio = np.inf
        
        # -- Print to console (same as original) --
        print(f"Constraint row {i}:")
        print(f"  Expectation E[s_{i}] = {E:.4f}")
        print(f"  Variance Var(s_{i})  = {Var:.4f}")
        print(f"  Ratio sqrt(Var)/E    = {ratio:.4f}")
        
        # -- Also store in JSON-friendly dict --
        variance_info_list.append({
            "E": float(E),
            "Var": float(Var),
            "ratio": float(ratio)
        })

    return variance_info_list


# ======================================================================
# Rounding Function for Converting Continuous to Binary
# ======================================================================

def randomized_rounding(
    cont_sol: np.ndarray,
    obj_func: Callable,
    A: np.ndarray,
    b: np.ndarray,
    num_samples: int = 100,
    verbose: bool = False,
    args: tuple = ()
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Randomized rounding to convert a continuous solution to a feasible binary solution.
    
    Args:
        cont_sol: Continuous solution vector (in [0,1]^n)
        obj_func: Objective function to evaluate solutions
        A: Inequality constraint matrix
        b: Inequality constraint bounds
        num_samples: Number of randomized rounding samples to try
        verbose: Whether to print detailed output
        args: Additional arguments to pass to objective function
    
    Returns:
        Tuple containing:
        - Best binary solution found
        - Corresponding objective value
        - Statistics dictionary
    """
    n = len(cont_sol)
    
    # Initialize with no solution
    best_binary = None
    best_obj = float('-inf')
    
    # Statistics tracking
    stats = {
        "num_samples": num_samples,
        "num_feasible": 0,
        "feasibility_rate": 0.0,
        "obj_values": [],
        "runtime": 0.0
    }
    
    start_time = time.time()
    
    if verbose:
        logger.info(f"Starting randomized rounding with {num_samples} samples...")
        # Compute variance information for constraints
        if verbose:
            for i in range(A.shape[0]):
                a_i = A[i, :]
                E = np.dot(a_i, cont_sol)
                Var = np.sum((a_i**2) * cont_sol * (1 - cont_sol))
                
                if E > 1e-12:
                    ratio = np.sqrt(Var) / E
                else:
                    ratio = float('inf')
                
                logger.info(f"Constraint row {i}:")
                logger.info(f"  Expectation E[s_{i}] = {E:.4f}")
                logger.info(f"  Variance Var(s_{i})  = {Var:.4f}")
                logger.info(f"  Ratio sqrt(Var)/E    = {ratio:.4f}")
    
    # Try randomized rounding using continuous solution values as probabilities
    for i in range(num_samples):
        binary_sol = np.zeros(n, dtype=int)
        for j in range(n):
            # Each variable has probability cont_sol[j] of being 1
            binary_sol[j] = 1 if np.random.random() < cont_sol[j] else 0
        
        # Check constraints
        feasible = True
        for k in range(A.shape[0]):
            if np.dot(A[k], binary_sol) > b[k] + 1e-9:
                feasible = False
                break
        
        if feasible:
            stats["num_feasible"] += 1
            obj_val = obj_func(binary_sol, *args)
            stats["obj_values"].append(obj_val)
            
            if obj_val > best_obj:
                best_obj = obj_val
                best_binary = binary_sol.copy()
                
                if verbose and (i+1) % (max(1, num_samples // 10)) == 0:
                    logger.info(f"Sample {i+1}: Found better solution with objective {obj_val:.6f}")
    
    stats["runtime"] = time.time() - start_time
    stats["feasibility_rate"] = stats["num_feasible"] / num_samples if num_samples > 0 else 0
    
    # Report statistics if in verbose mode
    if verbose:
        logger.info(f"Rounding complete: {stats['num_feasible']} feasible solutions found ({stats['feasibility_rate']:.2%})")
        if best_binary is not None:
            logger.info(f"Best objective value: {best_obj:.6f}")
    
    # If no feasible solution was found, try a fallback approach
    if best_binary is None:
        if verbose:
            logger.warning("No feasible solution found with randomized rounding. Trying greedy fallback approach.")
        
        # Sort by value (descending to prioritize variables with high probability)
        sorted_indices = np.argsort(-cont_sol)
        
        # Try starting with empty solution and greedily adding variables
        binary_sol = np.zeros(n, dtype=int)
        
        for idx in sorted_indices:
            # Try adding this variable
            binary_sol[idx] = 1
            
            # Check if still feasible
            feasible = True
            for i in range(A.shape[0]):
                if np.dot(A[i], binary_sol) > b[i] + 1e-9:
                    feasible = False
                    break
            
            # If not feasible, revert
            if not feasible:
                binary_sol[idx] = 0
        
        # Check if this solution is valid
        obj_val = obj_func(binary_sol, *args)
        if best_binary is None or obj_val > best_obj:
            best_obj = obj_val
            best_binary = binary_sol.copy()
            
            if verbose:
                logger.info(f"Fallback approach found solution with objective {obj_val:.6f}")
    
    if best_binary is None:
        # If still no solution, return an empty selection as last resort
        logger.warning("Failed to find any feasible solution, returning empty selection")
        return np.zeros(n, dtype=int), obj_func(np.zeros(n, dtype=int), *args), stats
    
    return best_binary, best_obj, stats

def knapsack_cr(
    a_j: np.ndarray,
    b_j: np.ndarray,
    y: np.ndarray,
    verbose: bool = False
) -> np.ndarray:
    """
    KNAPSACK-CR procedure for a single constraint.
    
    Args:
        a_j: Vector of resource requirements for constraint j
        b_j: Capacity bound for constraint j (could be scalar or array)
        y: Binary vector from Bernoulli sampling
        verbose: Whether to print progress
    
    Returns:
        tau_j: Binary vector indicating which elements to keep
    """
    n = len(a_j)
    assert n == len(y), "Dimension mismatch between a_j and y"
    
    # Initialize empty solution
    tau = np.zeros(n, dtype=bool)
    
    # Extract scalar value from b_j if it's an array
    if isinstance(b_j, np.ndarray):
        capacity = float(b_j.item()) if b_j.size == 1 else float(b_j[0])
    else:
        capacity = float(b_j)
    
    # We only consider elements where y[i] = 1
    candidate_indices = np.where(y)[0]
    
    if len(candidate_indices) == 0:
        return tau
    
    # Get resource requirements for candidate elements
    resource_requirements = a_j[candidate_indices]
    
    # Sort candidates by increasing resource requirement
    sorted_idx = np.argsort(resource_requirements)
    sorted_candidates = candidate_indices[sorted_idx]
    
    # Track remaining capacity
    remaining_capacity = capacity
    
    # Greedily add elements in order of increasing resource requirement
    for i in sorted_candidates:
        # Convert resource requirement to float before comparison
        req = float(a_j[i])
        
        # If including this element doesn't violate the constraint
        if req <= remaining_capacity + 1e-9:
            tau[i] = True
            remaining_capacity -= req
    
    if verbose:
        usage = float(a_j @ tau)
        logger.info(f"KNAPSACK-CR for constraint used {usage:.4f}/{capacity:.4f} capacity")
    
    return tau

def algorithm_3_contention_resolution_rounding(
    x_star: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    seed: Optional[int] = None,
    num_samples: int = 1,
    verbose: bool = True
) -> np.ndarray:
    """
    Implementation of Algorithm 3: Contention Resolution Rounding.
    
    Given:
      - x_star: Continuous solution in [0,1]^n to be rounded
      - A: Matrix of shape (m, n) for m constraints over n items
      - b: Vector of shape (m,) with constraint bounds
      - seed: Random seed for reproducibility
      - num_samples: Number of rounding attempts to perform (more attempts can improve quality)
      - verbose: Whether to print progress information
    
    Returns:
      - omega: A binary solution in {0,1}^n that satisfies A·omega ≤ b
    """
    if seed is not None:
        np.random.seed(seed)
    
    m, n = A.shape
    assert n == len(x_star), f"Dimension mismatch: A is {m}x{n}, but x_star has length {len(x_star)}"
    assert m == len(b), f"Dimension mismatch: A is {m}x{n}, but b has length {len(b)}"
    
    best_omega = None
    best_objective = float('-inf')
    
    # Try multiple samples to get the best result
    for sample in range(num_samples):
        if verbose and num_samples > 1:
            logger.info(f"Attempt {sample+1}/{num_samples}")
        
        # Step 1: Independent randomized rounding (Bernoulli trials)
        y = np.random.random(n) <= x_star
        
        # Step 2: Apply KNAPSACK-CR for each constraint
        tau_results = []
        
        for j in range(m):
            # Perform knapsack contention resolution for constraint j
            tau_j = knapsack_cr(A[j], b[j], y, verbose=verbose if sample == 0 else False)
            tau_results.append(tau_j)
        
        # Step 3: Compute intersection of all solutions
        omega = np.ones(n, dtype=bool)
        for tau in tau_results:
            omega = np.logical_and(omega, tau)
        
        # Convert boolean array to int array (0s and 1s)
        omega = omega.astype(int)
        
        # Calculate objective value (here we use sum as a simple measure)
        obj_value = omega.sum()
        
        # Keep track of the best solution found
        if obj_value > best_objective:
            best_objective = obj_value
            best_omega = omega.copy()
        
        if verbose and num_samples > 1:
            logger.info(f"  Solution quality: {obj_value}")
    
    # Final solution
    omega = best_omega
    
    if verbose:
        # Check final feasibility
        violations = []
        for j in range(m):
            usage = A[j] @ omega
            if usage > b[j] + 1e-9:
                violations.append((j, usage, b[j]))
        
        logger.info(f"Final solution selects {omega.sum()} elements")
        
        if violations:
            logger.warning("Solution has constraint violations:")
            for j, usage, capacity in violations:
                logger.warning(f"  Constraint {j}: {usage:.4f} > {capacity:.4f}")
        else:
            logger.info("Solution is feasible for all constraints")
    
    return omega

def round_solution_with_cr(
    cont_sol: np.ndarray,
    obj_func: Callable,
    A: np.ndarray,
    b: np.ndarray,
    num_samples: int = 10,
    verbose: bool = True,
    args: tuple = ()
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Round a continuous solution using Contention Resolution (CR).
    
    Args:
        cont_sol: Continuous solution in [0,1]^n
        obj_func: Objective function to evaluate solutions
        A: Constraint matrix
        b: Constraint bounds
        num_samples: Number of contention resolution rounding attempts
        verbose: Whether to print detailed output
        args: Additional arguments to pass to objective function
    
    Returns:
        Tuple containing:
        - Best binary solution found
        - Corresponding objective value
        - Statistics dictionary
    """
    # Statistics tracking
    stats = {
        "num_samples": num_samples,
        "runtime": 0.0,
        "obj_values": []
    }
    
    start_time = time.time()
    
    if verbose:
        logger.info(f"Starting contention resolution rounding with {num_samples} samples...")
    
    # Perform contention resolution rounding
    binary_sol = algorithm_3_contention_resolution_rounding(
        x_star=cont_sol,
        A=A,
        b=b,
        num_samples=num_samples,
        verbose=verbose
    )
    
    # Calculate objective value of rounded solution
    obj_val = obj_func(binary_sol, *args)
    stats["obj_values"].append(obj_val)
    
    # Final solution
    stats["runtime"] = time.time() - start_time
    
    if verbose:
        logger.info(f"Rounded solution objective: {obj_val:.6f}")
        logger.info(f"Selected {np.sum(binary_sol)} elements")
    
    return binary_sol, obj_val, stats

def compute_variance_info(A: np.ndarray, x_star: np.ndarray):
    """
    For each row i of A:
      - Compute expectation, variance, and ratio of std to expectation
      - Useful for analyzing randomized rounding behavior
    
    Args:
        A: Constraint matrix (each row = one constraint)
        x_star: Solution vector (entries in [0,1])

    Returns:
        List of dictionaries with statistics for each constraint
    """
    p, m = A.shape
    variance_info_list = []

    for i in range(p):
        a_i = A[i, :]
        E = np.dot(a_i, x_star)
        Var = np.sum((a_i**2) * x_star * (1 - x_star))
        
        if E > 1e-12:
            ratio = np.sqrt(Var) / E
        else:
            ratio = float('inf')
        
        logger.info(f"Constraint row {i}:")
        logger.info(f"  Expectation E[s_{i}] = {E:.4f}")
        logger.info(f"  Variance Var(s_{i})  = {Var:.4f}")
        logger.info(f"  Ratio sqrt(Var)/E    = {ratio:.4f}")
        
        variance_info_list.append({
            "E": float(E),
            "Var": float(Var),
            "ratio": float(ratio)
        })

    return variance_info_list

def compute_suboptimality_bound(
    rounded_obj: float,
    relaxed_obj: float
) -> Tuple[float, float]:
    """
    Compute suboptimality bound using the relaxed solution.
    
    For a maximization problem, we know that:
    OPT <= relaxed_obj
    
    So the relative suboptimality is at most:
    (relaxed_obj - rounded_obj) / relaxed_obj
    
    Args:
        rounded_obj: Objective value of the rounded solution
        relaxed_obj: Objective value of the relaxed solution
        
    Returns:
        Tuple of (absolute gap, relative gap)
    """
    absolute_gap = relaxed_obj - rounded_obj
    relative_gap = absolute_gap / relaxed_obj if relaxed_obj > 0 else float('inf')
    
    return absolute_gap, relative_gap

def compute_comparative_metrics(
    results: Dict[str, Dict[str, Any]],
    baseline_algorithm: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute comparative metrics for algorithms.
    
    Args:
        results: Dictionary of algorithm results
        baseline_algorithm: Name of the algorithm to use as baseline
        
    Returns:
        Dictionary of comparative metrics
    """
    metrics = {}
    
    # If no baseline is specified, use the best algorithm by objective
    if baseline_algorithm is None or baseline_algorithm not in results:
        baseline_algorithm = max(
            results.items(),
            key=lambda x: x[1]["objective"]
        )[0]
    
    baseline_obj = results[baseline_algorithm]["objective"]
    baseline_time = results[baseline_algorithm]["runtime"]
    
    for alg_name, alg_results in results.items():
        obj = alg_results["objective"]
        time = alg_results["runtime"]
        
        # Compute metrics
        obj_ratio = obj / baseline_obj if baseline_obj != 0 else float('inf')
        time_ratio = time / baseline_time if baseline_time != 0 else float('inf')
        efficiency = obj_ratio / time_ratio if time_ratio != 0 else float('inf')
        
        metrics[alg_name] = {
            "obj_ratio": obj_ratio,
            "time_ratio": time_ratio,
            "efficiency": efficiency
        }
    
    return metrics