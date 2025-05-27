# ----------------------------------------------------------------------
#  Log‑det objective + gradient using one sparse Cholesky solve
#  plus “reduced” wrappers that work on cache['_F_matrix']
# ----------------------------------------------------------------------
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from typing import Dict, Any
from fim_utils import combined_fim, precompute_fim_stack
from scipy.sparse.linalg import lobpcg, spilu
from scipy.sparse import spmatrix
from scipy.linalg import eigh
from typing import List
from collections import namedtuple
from scipy.sparse import eye

Quadruplet = namedtuple('Quadruplet', ['matrix_idx','row','col','value'])

# ------------------------------------------------------------------ #
#  Backend for sparse Cholesky factorisation
# ------------------------------------------------------------------ #
try:
    from sksparse.cholmod import cholesky
    _FACTOR_BACKEND = "cholmod"
except ModuleNotFoundError:
    import scipy.sparse.linalg as spla
    _FACTOR_BACKEND = "splu"

def _factor(F_csc: sp.csc_matrix):
    if _FACTOR_BACKEND == "cholmod":
        return cholesky(F_csc)
    else:
        return spla.splu(F_csc)

# ------------------------------------------------------------------ #
#  Original logdet on F(x)
# ------------------------------------------------------------------ #
def logdet_objective(x: np.ndarray, cache: Dict[str, Any]) -> float:
    F_csc = combined_fim(x, cache).tocsc()
    F_csc += 1e-8 * sp.eye(F_csc.shape[0], format="csc")
    fac   = _factor(F_csc)
    return fac.logdet()

def logdet_gradient(x: np.ndarray, cache: Dict[str, Any]) -> np.ndarray:
    F_csc = combined_fim(x, cache).tocsc()
    F_csc += 1e-8 * eye(F_csc.shape[0], format="csc")
    fac   = _factor(F_csc)

    cols_unique = np.unique(cache["cols"])
    col_ptr = {c:i for i,c in enumerate(cols_unique)}

    d = cache["shape"][0]
    C = np.zeros((d, len(cols_unique)), dtype=cache["data0"].dtype)
    for r, c, v in zip(cache["rows"], cache["cols"], cache["data0"]):
        C[r, col_ptr[c]] += v

    if _FACTOR_BACKEND=="cholmod":
        Y = fac.solve_A(C)
    else:
        Y = fac.solve(C)

    grad = np.zeros_like(x)
    ptrs = [col_ptr[c] for c in cache["cols"]]
    yvals = Y[cache["rows"], ptrs]
    np.add.at(grad, cache["owner"], cache["data0"] * yvals)
    return grad

def logdet_reduced_objective(x: np.ndarray, cache: Dict[str, Any]) -> float:
    if "_F_matrix" in cache:
        F = cache["_F_matrix"].tocsc()
        F += 1e-8 * eye(F.shape[0], format="csc")
        fac = _factor(F)
        return fac.logdet()
    else:
        return logdet_objective(x, cache)

def logdet_reduced_gradient(_: np.ndarray, cache: Dict[str, Any]) -> np.ndarray:
    if "_F_matrix" in cache:
        F = cache["_F_matrix"].tocsc()
        fac = _factor(F)

        cols_unique = np.unique(cache["cols"])
        col_ptr = {c:i for i,c in enumerate(cols_unique)}

        d = F.shape[0]
        C = np.zeros((d, len(cols_unique)), dtype=cache["data0"].dtype)
        for r, c, v in zip(cache["rows"], cache["cols"], cache["data0"]):
            C[r, col_ptr[c]] += v

        if _FACTOR_BACKEND=="cholmod":
            Y = fac.solve_A(C)
        else:
            Y = fac.solve(C)

        grad = np.zeros(cache["owner"].max()+1)
        ptrs = [col_ptr[c] for c in cache["cols"]]
        yvals = Y[cache["rows"], ptrs]
        np.add.at(grad, cache["owner"], cache["data0"] * yvals)
        return grad
    else:
        return logdet_gradient(_, cache)

def min_eig_objective(x: np.ndarray, cache) -> float:
    F = combined_fim(x, cache).tocsc()
    F += 1e-10 * sp.eye(F.shape[0], format="csc")
    ilu = spilu(F)
    M = lambda v: ilu.solve(v)
    if not hasattr(min_eig_objective, "v0"):
        v0 = np.random.randn(F.shape[0])
        v0 /= np.linalg.norm(v0)
        min_eig_objective.v0 = v0
    X = min_eig_objective.v0.reshape(-1, 1)
    vals, vecs = lobpcg(F, X, M=M, largest=False, tol=1e-3, maxiter=1000)
    v = vecs[:, 0]
    min_eig_objective.v0 = v
    return float(vals[0])

def min_eig_gradient(x: np.ndarray, cache) -> np.ndarray:
    F = combined_fim(x, cache).tocsc()
    F += 1e-10 * sp.eye(F.shape[0], format="csc")
    ilu = spilu(F)
    M   = lambda v: ilu.solve(v)
    v0 = getattr(min_eig_objective, "v0", None)
    if v0 is None:
        v0 = np.random.randn(F.shape[0])
        v0 /= np.linalg.norm(v0)
    X = v0.reshape(-1, 1)
    vals, vecs = lobpcg(F, X, M=M, largest=False, tol=1e-3, maxiter=1000)
    v = vecs[:, 0]
    min_eig_objective.v0 = v
    Av_vals = cache["data0"] * v[cache["rows"]]
    contrib = Av_vals * v[cache["cols"]]
    grad    = np.zeros(cache["owner"].max() + 1)
    np.add.at(grad, cache["owner"], contrib)
    return grad

def reduced_laplacian(
    quads: List[Quadruplet],
    remove_index: int
) -> List[Quadruplet]:
    reduced = []
    for q in quads:
        # drop any entry in the removed row or column
        if q.row == remove_index or q.col == remove_index:
            continue
        # shift indices > remove_index down by 1
        new_row = q.row - 1 if q.row > remove_index else q.row
        new_col = q.col - 1 if q.col > remove_index else q.col
        reduced.append( Quadruplet(q.matrix_idx, new_row, new_col, q.value) )
    return reduced