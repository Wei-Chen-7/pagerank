"""Core PageRank: Google-matrix construction and power iteration.

The random-surfer model
-----------------------
Picture someone browsing a directed graph of pages.  At each step they either

* (with probability ``damping``) click a uniformly-random link on the current
  page, or
* (with probability ``1 - damping``) get bored and *teleport* to a page drawn
  from a distribution ``v`` (uniform by default).

The PageRank of a page is the long-run fraction of time the surfer spends on
it -- the stationary distribution of this Markov chain.

The linear algebra
-------------------
Let ``A`` be the adjacency matrix of ``n`` nodes (``A[i, j] > 0`` iff there is
a link ``i -> j``; weights are allowed).  Dividing each row by its out-degree
gives the row-stochastic *hyperlink matrix* ``H``.  Rows for *dangling* nodes
(no out-links) are all zero; we patch them so the surfer teleports, turning
``H`` into ``S = H + d v^T`` where ``d`` is the dangling indicator.  ``S`` is
row-stochastic.  The **Google matrix** is

    G = damping * S + (1 - damping) * 1 v^T ,

which is row-stochastic, irreducible and aperiodic.  By Perron-Frobenius it has
a unique stationary *row* vector ``pi`` with

    pi G = pi ,   pi >= 0 ,   sum(pi) = 1 .

That ``pi`` is the PageRank vector.  Power iteration ``pi <- pi G`` converges to
it geometrically; the error shrinks by a factor of about ``damping`` each step
(the modulus of the second-largest eigenvalue of ``G``).

We never need to form the dense ``G`` to iterate: the update

    r_new = damping * (H^T r) + (damping * dangling_mass + (1 - damping)) * v

is algebraically identical to ``r_new = r G`` and keeps ``sum(r) == 1`` exactly
(up to floating point), which is what lets the same routine scale to sparse
graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "PageRankResult",
    "hyperlink_matrix",
    "google_matrix",
    "power_iteration",
    "pagerank",
    "pagerank_eig",
    "subdominant_eigenvalue",
]


@dataclass
class PageRankResult:
    """Outcome of a power-iteration run.

    Attributes
    ----------
    scores:
        The PageRank vector -- non-negative, sums to 1.
    iterations:
        Number of iterations performed.
    converged:
        Whether the L1 residual dropped below ``tol`` before ``max_iter``.
    residuals:
        The L1 residual ``sum(|r_new - r|)`` recorded at each iteration; handy
        for plotting convergence.
    """

    scores: np.ndarray
    iterations: int
    converged: bool
    residuals: list[float] = field(default_factory=list)

    def top(self, k: int | None = None) -> np.ndarray:
        """Indices of the ``k`` highest-scoring nodes, best first."""
        order = np.argsort(self.scores)[::-1]
        return order if k is None else order[:k]


def _as_adjacency(adjacency) -> np.ndarray:
    """Validate and return a square, non-negative float adjacency matrix."""
    a = np.asarray(adjacency, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be a square 2D matrix")
    if np.any(a < 0):
        raise ValueError("adjacency entries must be non-negative")
    if not np.all(np.isfinite(a)):
        raise ValueError("adjacency entries must be finite")
    return a


def _teleport_vector(n: int, personalization) -> np.ndarray:
    """Return a normalized teleport/personalization distribution of length ``n``."""
    if personalization is None:
        return np.full(n, 1.0 / n)
    v = np.asarray(personalization, dtype=float)
    if v.shape != (n,):
        raise ValueError(f"personalization must have shape ({n},)")
    if np.any(v < 0):
        raise ValueError("personalization entries must be non-negative")
    total = v.sum()
    if total <= 0:
        raise ValueError("personalization must sum to a positive value")
    return v / total


def hyperlink_matrix(adjacency) -> tuple[np.ndarray, np.ndarray]:
    """Row-stochastic hyperlink matrix ``H`` and the dangling-node mask.

    ``H[i, j] = A[i, j] / out_degree(i)``.  Rows of dangling nodes (zero
    out-degree) are left as all-zeros; the second return value flags them.
    """
    a = _as_adjacency(adjacency)
    out_degree = a.sum(axis=1)
    dangling = out_degree == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        h = a / out_degree[:, None]
    h[dangling] = 0.0
    return h, dangling


def google_matrix(adjacency, damping: float = 0.85, personalization=None) -> np.ndarray:
    """Build the dense Google matrix ``G`` (row-stochastic).

    ``G = damping * S + (1 - damping) * 1 v^T`` where ``S`` is ``H`` with
    dangling rows replaced by the teleport vector ``v``.  Provided mainly so the
    power-iteration result can be checked against a true eigen-decomposition.
    """
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must be in [0, 1)")
    a = _as_adjacency(adjacency)
    n = a.shape[0]
    v = _teleport_vector(n, personalization)
    h, dangling = hyperlink_matrix(a)
    s = h.copy()
    s[dangling] = v  # dangling surfer teleports according to v
    return damping * s + (1.0 - damping) * np.outer(np.ones(n), v)


def power_iteration(
    adjacency,
    damping: float = 0.85,
    personalization=None,
    tol: float = 1e-10,
    max_iter: int = 1000,
    x0=None,
) -> PageRankResult:
    """Compute PageRank by power iteration on the (implicit) Google matrix.

    Iterates ``r <- r G`` using the mass-conserving vector update, so ``G`` is
    never materialized.  Stops when the L1 change between iterates falls below
    ``tol`` or after ``max_iter`` steps.
    """
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must be in [0, 1)")
    a = _as_adjacency(adjacency)
    n = a.shape[0]
    if n == 0:
        return PageRankResult(np.zeros(0), 0, True, [])

    v = _teleport_vector(n, personalization)
    h, dangling = hyperlink_matrix(a)
    h_t = h.T  # so that r @ H == h_t @ r

    if x0 is None:
        r = v.copy()
    else:
        r = np.asarray(x0, dtype=float)
        if r.shape != (n,):
            raise ValueError(f"x0 must have shape ({n},)")
        if r.sum() == 0:
            raise ValueError("x0 must not be the zero vector")
        r = r / r.sum()

    residuals: list[float] = []
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        dangling_mass = r[dangling].sum()
        r_new = damping * (h_t @ r) + (damping * dangling_mass + (1.0 - damping)) * v
        residual = float(np.abs(r_new - r).sum())
        residuals.append(residual)
        r = r_new
        if residual < tol:
            converged = True
            break

    r = r / r.sum()  # guard against floating-point drift
    return PageRankResult(scores=r, iterations=iterations, converged=converged, residuals=residuals)


def pagerank(
    adjacency,
    damping: float = 0.85,
    personalization=None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> np.ndarray:
    """Convenience wrapper returning just the PageRank score vector."""
    return power_iteration(
        adjacency,
        damping=damping,
        personalization=personalization,
        tol=tol,
        max_iter=max_iter,
    ).scores


def pagerank_eig(adjacency, damping: float = 0.85, personalization=None) -> np.ndarray:
    """Reference PageRank via the dominant left eigenvector of ``G``.

    Solves ``pi G = pi`` by taking the eigenvector of ``G^T`` whose eigenvalue is
    closest to 1, then normalizing to a probability vector.  Used to validate the
    power-iteration implementation; ``O(n^3)`` and dense, so only for small ``n``.
    """
    g = google_matrix(adjacency, damping=damping, personalization=personalization)
    values, vectors = np.linalg.eig(g.T)
    dominant = int(np.argmin(np.abs(values - 1.0)))
    vec = np.real(vectors[:, dominant])
    if vec.sum() < 0:  # np.linalg.eig fixes the sign arbitrarily; orient it positive
        vec = -vec
    vec = np.clip(vec, 0.0, None)  # Perron vector is non-negative up to fp noise
    total = vec.sum()
    if total == 0:
        raise RuntimeError("degenerate dominant eigenvector")
    return vec / total


def subdominant_eigenvalue(adjacency, damping: float = 0.85, personalization=None) -> float:
    """Modulus of the second-largest eigenvalue of the Google matrix, ``|λ₂|``.

    The largest eigenvalue is always 1 (its eigenvector is the PageRank vector).
    The *next* one governs how fast power iteration converges: the error shrinks
    by roughly this factor per step, so iterations-to-tolerance scale like
    ``log(tol) / log(|λ₂|)``.  For the Google matrix ``|λ₂| ≤ damping``, which is
    exactly why the damping factor is the knob that trades ranking fidelity
    against convergence speed.  Dense ``O(n³)``; for study and testing only.
    """
    g = google_matrix(adjacency, damping=damping, personalization=personalization)
    if g.shape[0] < 2:
        return 0.0
    magnitudes = np.sort(np.abs(np.linalg.eigvals(g)))
    return float(magnitudes[-2])
