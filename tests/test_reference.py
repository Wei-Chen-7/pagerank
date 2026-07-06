"""Cross-validate our from-scratch PageRank against NetworkX's canonical one.

NetworkX is used here purely as an independent *oracle* for the tests -- the
library's ``pagerank()`` is never called from the package itself.  Agreement to
~1e-8 on graphs with dangling nodes, weighted edges and personalization is strong
evidence the implementation is correct, not just internally consistent.
"""

import numpy as np
import pytest

from pagerank import datasets
from pagerank.core import pagerank

nx = pytest.importorskip("networkx")


def _nx_digraph(adjacency: np.ndarray):
    g = nx.DiGraph()
    n = adjacency.shape[0]
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adjacency[i, j] != 0:
                g.add_edge(i, j, weight=float(adjacency[i, j]))
    return g


def _nx_pagerank(adjacency, **kwargs):
    ref = nx.pagerank(_nx_digraph(adjacency), weight="weight", tol=1e-13,
                      max_iter=2000, **kwargs)
    return np.array([ref[i] for i in range(adjacency.shape[0])])


@pytest.mark.parametrize(
    "adjacency",
    [
        datasets.tiny_graph().adjacency_matrix(),
        datasets.dangling_graph().adjacency_matrix(),
        datasets.web_of_influence().adjacency_matrix(),
    ],
    ids=["tiny", "dangling", "web_of_influence"],
)
def test_matches_networkx(adjacency):
    mine = pagerank(adjacency, damping=0.85, tol=1e-13)
    assert np.allclose(mine, _nx_pagerank(adjacency, alpha=0.85), atol=1e-8)


@pytest.mark.parametrize("seed", range(5))
def test_matches_networkx_random(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 10))
    adjacency = (rng.random((n, n)) < 0.4).astype(float)
    np.fill_diagonal(adjacency, 0)
    mine = pagerank(adjacency, damping=0.85, tol=1e-13)
    assert np.allclose(mine, _nx_pagerank(adjacency, alpha=0.85), atol=1e-8)


@pytest.mark.parametrize("damping", [0.5, 0.7, 0.9, 0.99])
def test_matches_networkx_across_damping(damping):
    adjacency = datasets.web_of_influence().adjacency_matrix()
    mine = pagerank(adjacency, damping=damping, tol=1e-13)
    assert np.allclose(mine, _nx_pagerank(adjacency, alpha=damping), atol=1e-8)


def test_matches_networkx_personalized():
    adjacency = datasets.tiny_graph().adjacency_matrix()
    vector = np.array([0.5, 0.2, 0.2, 0.1])
    mine = pagerank(adjacency, personalization=vector, tol=1e-13)
    personalization = {i: vector[i] for i in range(len(vector))}
    reference = _nx_pagerank(adjacency, alpha=0.85, personalization=personalization)
    assert np.allclose(mine, reference, atol=1e-8)
