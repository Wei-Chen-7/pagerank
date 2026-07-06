"""Tests for the spectral theory behind convergence: |λ₂| ≤ damping, and the
observed power-iteration rate equals |λ₂|."""

import numpy as np
import pytest

from pagerank import datasets
from pagerank.core import power_iteration, subdominant_eigenvalue


def _observed_rate(residuals):
    """Geometric mean of residual ratios in the clean middle band."""
    r = np.asarray(residuals)
    band = r[(r < 1e-2) & (r > 1e-11)]
    ratios = band[1:] / band[:-1]
    return float(np.exp(np.mean(np.log(ratios))))


@pytest.mark.parametrize(
    "adjacency",
    [
        datasets.tiny_graph().adjacency_matrix(),
        datasets.dangling_graph().adjacency_matrix(),
        datasets.web_of_influence().adjacency_matrix(),
        datasets.scale_free_graph(200, edges_per_node=3, seed=0).adjacency_matrix(),
    ],
    ids=["tiny", "dangling", "web", "scale_free"],
)
@pytest.mark.parametrize("damping", [0.5, 0.7, 0.85, 0.95, 0.99])
def test_subdominant_is_bounded_by_damping(adjacency, damping):
    """The core inequality: |λ₂| ≤ d (with a hair of floating-point slack)."""
    l2 = subdominant_eigenvalue(adjacency, damping=damping)
    assert l2 <= damping + 1e-9
    assert 0.0 <= l2 < 1.0  # strictly below the dominant eigenvalue 1


@pytest.mark.parametrize("damping", [0.5, 0.7, 0.85, 0.95])
def test_observed_rate_equals_subdominant(damping):
    """Power iteration's residual decays at exactly the |λ₂| rate."""
    adjacency = datasets.web_of_influence().adjacency_matrix()
    result = power_iteration(adjacency, damping=damping, tol=1e-13, max_iter=8000)
    observed = _observed_rate(result.residuals)
    expected = subdominant_eigenvalue(adjacency, damping=damping)
    assert observed == pytest.approx(expected, rel=0.03)


def test_iterations_grow_with_damping():
    """Higher damping means slower convergence -> more iterations to a fixed tol."""
    adjacency = datasets.web_of_influence().adjacency_matrix()
    iters = [
        power_iteration(adjacency, damping=d, tol=1e-10, max_iter=20000).iterations
        for d in (0.5, 0.85, 0.99)
    ]
    assert iters[0] < iters[1] < iters[2]


def test_single_node_has_no_subdominant():
    assert subdominant_eigenvalue(np.array([[0.0]])) == 0.0
