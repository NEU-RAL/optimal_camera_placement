import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.optimize import linprog
import gurobipy as gp
from gurobipy import GRB
import time
import matplotlib.pyplot as plt

# Problem setup
n_sensors = 100
M = 50  # size of each information matrix
k_selected = 10

# Generate random information matrices
def generate_random_inf_mats(n, M, seed=42):
    np.random.seed(seed)
    inf_mats = []
    for _ in range(n):
        A = np.random.randn(M, M)
        mat = A.T @ A + 0.1 * np.eye(M)
        inf_mats.append(sp.csr_matrix(mat))
    H0 = sp.csr_matrix(np.eye(M) * 0.01)
    return inf_mats, H0

# Objective function
def obj_func(x, inf_mats, H0):
    combined = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined += xi * Hi
    min_eigval, _ = eigsh(combined, k=1, which='SA')
    return min_eigval[0]

# Gradient of objective function
def obj_grad(x, inf_mats, H0):
    combined = H0.copy()
    for xi, Hi in zip(x, inf_mats):
        combined += xi * Hi
    _, v = eigsh(combined, k=1, which='SA')
    grad = np.array([v[:,0].T @ Hi @ v[:,0] for Hi in inf_mats])
    return grad

# LP-based LMO
def scipy_lmo(grad):
    c = -grad
    A_eq = np.ones((1, n_sensors))
    b_eq = np.array([k_selected])
    bounds = [(0,1)]*n_sensors
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return res.x if res.success else None

# Gurobi-based LMO
def gurobi_lmo(grad):
    model = gp.Model()
    model.Params.OutputFlag = 0
    x = model.addVars(n_sensors, vtype=GRB.BINARY)
    model.addConstr(x.sum() == k_selected)
    model.setObjective(gp.quicksum(grad[i]*x[i] for i in range(n_sensors)), GRB.MAXIMIZE)
    model.optimize()
    return np.array([x[i].X for i in range(n_sensors)])

# Backtracking line search
def backtracking(x, d, grad, inf_mats, H0, alpha=1.0, beta=0.5, c=1e-4):
    obj_current = obj_func(x, inf_mats, H0)
    while True:
        x_new = x + alpha * d
        if obj_func(x_new, inf_mats, H0) >= obj_current + c * alpha * np.dot(grad, d):
            break
        alpha *= beta
    return alpha

# Frank-Wolfe optimizer
def frank_wolfe(lmo, inf_mats, H0, max_iters=1000, tol=1e-5):
    x = np.zeros(n_sensors)
    x[:k_selected] = 1  # initial selection
    history = []
    termination_reason = 'Reached maximum iterations'
    for it in range(max_iters):
        grad = obj_grad(x, inf_mats, H0)
        s = lmo(grad)
        if s is None:
            termination_reason = 'LMO failed'
            break
        d = s - x
        step_size = backtracking(x, d, grad, inf_mats, H0)
        if np.linalg.norm(step_size * d) < tol:
            termination_reason = f'Converged: step size below tolerance at iteration {it+1}'
            break
        x += step_size * d
        x = np.clip(x, 0, 1)
        val = obj_func(x, inf_mats, H0)
        history.append(val)
    print(f"Termination reason: {termination_reason}")
    return x, history

# Comparison
inf_mats, H0 = generate_random_inf_mats(n_sensors, M)

# Run Original
start = time.time()
x_scipy, hist_scipy = frank_wolfe(scipy_lmo, inf_mats, H0)
time_scipy = time.time() - start

# Run Gurobi
start = time.time()
x_gurobi, hist_gurobi = frank_wolfe(gurobi_lmo, inf_mats, H0)
time_gurobi = time.time() - start

# Results
print(f"SciPy Time: {time_scipy:.2f}s")
print(f"Gurobi Time: {time_gurobi:.2f}s")

# Plot
plt.plot(hist_scipy, label='SciPy LP')
plt.plot(hist_gurobi, label='Gurobi')
plt.xlabel('Iteration')
plt.ylabel('Min Eigenvalue')
plt.title('Frank-Wolfe Optimization Comparison (Backtracking)')
plt.legend()
plt.grid()
plt.show()
