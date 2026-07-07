"""A small directed-graph container with named nodes.

Keeps user-facing labels (strings, ints, anything hashable) separate from the
integer indices the linear algebra needs, and knows how to hand its structure
to the dense or sparse PageRank routines.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable

import numpy as np
import scipy.sparse as sp

from .core import power_iteration
from .sparse import pagerank_sparse

__all__ = ["DiGraph"]


class DiGraph:
    """A weighted directed graph indexed by hashable node labels."""

    def __init__(self) -> None:
        self._nodes: list[Hashable] = []
        self._index: dict[Hashable, int] = {}
        self._out: dict[int, dict[int, float]] = defaultdict(dict)

    # -- construction ----------------------------------------------------
    def add_node(self, node: Hashable) -> int:
        """Add ``node`` if new; return its integer index."""
        if node not in self._index:
            self._index[node] = len(self._nodes)
            self._nodes.append(node)
        return self._index[node]

    def add_edge(self, source: Hashable, target: Hashable, weight: float = 1.0) -> None:
        """Add a directed edge ``source -> target`` (accumulating weight)."""
        if weight < 0:
            raise ValueError("edge weight must be non-negative")
        i = self.add_node(source)
        j = self.add_node(target)
        self._out[i][j] = self._out[i].get(j, 0.0) + float(weight)

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple],
        nodes: Iterable[Hashable] | None = None,
    ) -> DiGraph:
        """Build a graph from ``(source, target)`` or ``(source, target, weight)`` tuples.

        ``nodes`` optionally seeds isolated or ordering-defining nodes first.
        """
        g = cls()
        for node in nodes or ():
            g.add_node(node)
        for edge in edges:
            if len(edge) == 2:
                g.add_edge(edge[0], edge[1])
            elif len(edge) == 3:
                g.add_edge(edge[0], edge[1], edge[2])
            else:
                raise ValueError("each edge must be (source, target[, weight])")
        return g

    # -- inspection ------------------------------------------------------
    @property
    def nodes(self) -> list[Hashable]:
        return list(self._nodes)

    def index_of(self, node: Hashable) -> int:
        return self._index[node]

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        return sum(len(targets) for targets in self._out.values())

    def in_degree(self) -> dict[Hashable, float]:
        """Weighted in-degree per node label."""
        deg = dict.fromkeys(self._nodes, 0.0)
        for targets in self._out.values():
            for j, w in targets.items():
                deg[self._nodes[j]] += w
        return deg

    def out_degree(self) -> dict[Hashable, float]:
        """Weighted out-degree per node label."""
        deg = dict.fromkeys(self._nodes, 0.0)
        for i, targets in self._out.items():
            deg[self._nodes[i]] = sum(targets.values())
        return deg

    # -- matrix views ----------------------------------------------------
    def adjacency_matrix(self) -> np.ndarray:
        """Dense ``n x n`` adjacency matrix in node-insertion order."""
        n = self.number_of_nodes()
        a = np.zeros((n, n))
        for i, targets in self._out.items():
            for j, w in targets.items():
                a[i, j] = w
        return a

    def adjacency_sparse(self) -> sp.csr_matrix:
        """Sparse (CSR) adjacency matrix in node-insertion order."""
        rows, cols, data = [], [], []
        for i, targets in self._out.items():
            for j, w in targets.items():
                rows.append(i)
                cols.append(j)
                data.append(w)
        n = self.number_of_nodes()
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)

    def _personalization_vector(self, personalization) -> np.ndarray | None:
        """Turn a ``{node: weight}`` mapping into an index-aligned vector."""
        if personalization is None:
            return None
        vec = np.zeros(self.number_of_nodes())
        for node, weight in personalization.items():
            vec[self._index[node]] = weight
        return vec

    # -- ranking ---------------------------------------------------------
    def pagerank(
        self,
        damping: float = 0.85,
        personalization: dict | None = None,
        tol: float = 1e-10,
        max_iter: int = 1000,
        sparse: bool = False,
    ) -> dict[Hashable, float]:
        """PageRank as a ``{node: score}`` dict.

        Set ``sparse=True`` to route through the SciPy-sparse solver, which is
        preferable for large graphs.  ``personalization`` is a ``{node: weight}``
        mapping (unlisted nodes get weight 0).
        """
        pvec = self._personalization_vector(personalization)
        if sparse:
            result = pagerank_sparse(
                self.adjacency_sparse(),
                damping=damping,
                personalization=pvec,
                tol=tol,
                max_iter=max_iter,
            )
        else:
            result = power_iteration(
                self.adjacency_matrix(),
                damping=damping,
                personalization=pvec,
                tol=tol,
                max_iter=max_iter,
            )
        return {self._nodes[i]: float(score) for i, score in enumerate(result.scores)}

    def rank(
        self,
        damping: float = 0.85,
        personalization: dict | None = None,
        **kwargs,
    ) -> list[tuple[Hashable, float]]:
        """PageRank as a list of ``(node, score)`` sorted from most to least important."""
        scores = self.pagerank(damping=damping, personalization=personalization, **kwargs)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"DiGraph(nodes={self.number_of_nodes()}, edges={self.number_of_edges()})"
