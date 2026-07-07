"""Sparse PageRank for large graphs, using SciPy CSR matrices.

Identical mathematics to :func:`pagerank.core.power_iteration`, but the
adjacency matrix is stored sparsely and every iteration is a single
sparse matrix-vector product, so memory and time stay ``O(nnz)`` rather than
``O(n^2)``.  This is what lets PageRank run on graphs with tens of thousands of
nodes (or far more) on a laptop.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .core import PageRankResult, _teleport_vector

__all__ = ["pagerank_sparse"]


def pagerank_sparse(
    adjacency,
    damping: float = 0.85,
    personalization=None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> PageRankResult:
    """Compute PageRank on a sparse adjacency matrix.

    Parameters
    ----------
    adjacency:
        Anything SciPy can turn into a square CSR matrix (``csr_matrix``,
        ``coo_matrix``, dense array, ...).  ``A[i, j] > 0`` means ``i -> j``.
    damping, personalization, tol, max_iter:
        As in :func:`pagerank.core.power_iteration`.

    Notes
    -----
    The row-normalized transition ``H = D^{-1} A`` never needs to be built.
    Since ``(r @ H)[j] = sum_i r[i] A[i, j] / out_i = (A^T @ (r / out))[j]``,
    each step is one product against the pre-transposed CSR matrix.
    """
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must be in [0, 1)")

    a = sp.csr_matrix(adjacency, dtype=float)
    if a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be square")
    if a.nnz and a.data.min() < 0:
        raise ValueError("adjacency entries must be non-negative")

    n = a.shape[0]
    if n == 0:
        return PageRankResult(np.zeros(0), 0, True, [])

    v = _teleport_vector(n, personalization)

    out_degree = np.asarray(a.sum(axis=1)).ravel()
    dangling = out_degree == 0
    inv_out = np.where(dangling, 0.0, 1.0 / np.where(dangling, 1.0, out_degree))
    a_t = a.T.tocsr()  # so that A^T @ x is a fast CSR mat-vec

    r = v.copy()
    residuals: list[float] = []
    converged = False
    for _ in range(max_iter):
        dangling_mass = r[dangling].sum()
        contrib = a_t @ (r * inv_out)
        r_new = damping * contrib + (damping * dangling_mass + (1.0 - damping)) * v
        residual = float(np.abs(r_new - r).sum())
        residuals.append(residual)
        r = r_new
        if residual < tol:
            converged = True
            break

    r = r / r.sum()
    return PageRankResult(
        scores=r, iterations=len(residuals), converged=converged, residuals=residuals
    )
