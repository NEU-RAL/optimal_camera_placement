# ----------------------------------------------------------------------
#  Sparse information–matrix utilities for OASIS‑style problems
# ----------------------------------------------------------------------
import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Any


def precompute_fim_stack(inf_mats: List[sp.spmatrix],
                         H0: sp.spmatrix) -> Dict[str, Any]:
    """
    Compress   {A_k}   into one COO pattern so that we can form

        F(x) = H0 + Σ_k  x_k A_k

    Returns
    -------
    cache : dict
        pattern : csc_matrix with dummy 1's       (keeps sparsity pattern)
        data0   : np.ndarray (Nnz,)               (original A_k non‑zeros)
        owner   : np.ndarray (Nnz,)               (which design var each entry belongs to)
        rows / cols : np.ndarray (Nnz,)           (explicit indices, used later)
        H0      : the prior / odometry block
        shape   : (d, d)
    """
    rows, cols, data, owner = [], [], [], []
    for k, Ak in enumerate(inf_mats):
        r, c = Ak.nonzero()
        rows.append(r)
        cols.append(c)
        data.append(Ak.data.copy())
        owner.append(np.full(r.size, k, dtype=np.int32))

    rows  = np.concatenate(rows)
    cols  = np.concatenate(cols)
    data0 = np.concatenate(data)
    owner = np.concatenate(owner)

    pattern = sp.csc_matrix((np.ones_like(data0), (rows, cols)), shape=H0.shape)

    return dict(pattern=pattern,
                data0=data0,
                owner=owner,
                rows=rows,
                cols=cols,
                H0=H0,
                shape=H0.shape)


def combined_fim(x, cache):
    """
    Vectorised F(x) = H0 + Σ_k x_k A_k,
    """
    data = cache["data0"] * x[cache["owner"]]
    F_sparse = sp.coo_matrix(
        (data, (cache["rows"], cache["cols"])),
        shape=cache["shape"]
    ).tocsc()
    return cache["H0"] + F_sparse