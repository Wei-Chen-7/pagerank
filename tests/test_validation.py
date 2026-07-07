"""The three validations called for by the project spec.

1. Power iteration matches the dominant eigenvector of the Google matrix.
2. The PageRank vector is non-negative, sums to 1, and iteration converges.
3. A tiny hand-computable graph yields the expected ranking order.
"""

import numpy as np
import pytest

from pagerank import datasets
from pagerank.core import google_matrix, pagerank_eig, power_iteration


def test_power_iteration_matches_dominant_eigenvector():
    """Compare power iteration to numpy.linalg.eig on a small graph."""
    a = datasets.tiny_graph().adjacency_matrix()

    # Independently pull the dominant eigenvector of G^T (eigenvalue 1).
    g = google_matrix(a, damping=0.85)
    values, vectors = np.linalg.eig(g.T)
    dominant = int(np.argmin(np.abs(values - 1.0)))

    assert values[dominant].real == pytest.approx(1.0, abs=1e-9)

    eigvec = np.real(vectors[:, dominant])
    eigvec = eigvec / eigvec.sum()

    power = power_iteration(a, tol=1e-13).scores
    assert np.allclose(power, eigvec, atol=1e-7)
    # And the packaged reference agrees too.
    assert np.allclose(power, pagerank_eig(a), atol=1e-7)


def test_pagerank_is_a_probability_distribution_and_converges():
    for graph in (datasets.tiny_graph(), datasets.dangling_graph(), datasets.web_of_influence()):
        result = power_iteration(graph.adjacency_matrix())
        assert (result.scores >= 0).all(), "PageRank must be non-negative"
        assert result.scores.sum() == pytest.approx(1.0), "PageRank must sum to 1"
        assert result.converged, "power iteration must converge"


def test_tiny_graph_expected_ranking_order():
    """The four-node graph must rank C > A > B > D."""
    ranking = [node for node, _ in datasets.tiny_graph().rank()]
    assert ranking == ["C", "A", "B", "D"]


def test_dangling_node_outranks_its_source():
    """In A -> B with B dangling, B should outrank A."""
    scores = datasets.dangling_graph().pagerank()
    assert scores["B"] > scores["A"]
