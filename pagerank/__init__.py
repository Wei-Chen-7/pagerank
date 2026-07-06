"""PageRank from first principles.

A small, honest implementation of PageRank: the Google matrix, power iteration,
a sparse solver for large graphs, personalized (topic-biased) ranking, and a
handful of example graphs.  Nothing here calls a library's ``pagerank()`` -- the
algorithm is built from the adjacency structure up.
"""

from .core import (
    PageRankResult,
    google_matrix,
    hyperlink_matrix,
    pagerank,
    pagerank_eig,
    power_iteration,
    subdominant_eigenvalue,
)
from .graph import DiGraph
from .sparse import pagerank_sparse
from . import datasets

__version__ = "0.1.0"

__all__ = [
    "PageRankResult",
    "hyperlink_matrix",
    "google_matrix",
    "power_iteration",
    "pagerank",
    "pagerank_eig",
    "subdominant_eigenvalue",
    "pagerank_sparse",
    "DiGraph",
    "datasets",
    "__version__",
]
