"""Tests for the dense core: Google matrix and power iteration."""

import numpy as np
import pytest

from pagerank import core
from pagerank.core import (
    google_matrix,
    hyperlink_matrix,
    pagerank,
    pagerank_eig,
    power_iteration,
)


@pytest.fixture
def simple_adjacency():
    # A->B, A->C, B->C, C->A, D->C
    return np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )


def test_hyperlink_matrix_row_stochastic(simple_adjacency):
    h, dangling = hyperlink_matrix(simple_adjacency)
    # No dangling nodes here; every row sums to 1.
    assert not dangling.any()
    assert np.allclose(h.sum(axis=1), 1.0)
    # A has out-degree 2, so its two links each carry weight 0.5.
    assert h[0, 1] == pytest.approx(0.5)
    assert h[0, 2] == pytest.approx(0.5)


def test_hyperlink_matrix_flags_dangling():
    a = np.array([[0, 1], [0, 0]], dtype=float)  # B dangles
    h, dangling = hyperlink_matrix(a)
    assert dangling.tolist() == [False, True]
    assert np.allclose(h[1], 0.0)  # dangling row is left as zeros


def test_google_matrix_is_row_stochastic(simple_adjacency):
    g = google_matrix(simple_adjacency, damping=0.85)
    assert np.allclose(g.sum(axis=1), 1.0)
    assert (g > 0).all()  # teleport makes every entry strictly positive


def test_google_matrix_dangling_row_is_stochastic():
    a = np.array([[0, 1], [0, 0]], dtype=float)
    g = google_matrix(a, damping=0.85)
    assert np.allclose(g.sum(axis=1), 1.0)


def test_power_iteration_sums_to_one_and_nonnegative(simple_adjacency):
    result = power_iteration(simple_adjacency)
    assert result.scores.sum() == pytest.approx(1.0)
    assert (result.scores >= 0).all()
    assert result.converged


def test_power_iteration_residuals_decrease(simple_adjacency):
    result = power_iteration(simple_adjacency, tol=1e-12)
    # Residuals should be monotonically shrinking for a contraction map.
    residuals = result.residuals
    assert all(a >= b for a, b in zip(residuals, residuals[1:]))


def test_matches_eigenvector(simple_adjacency):
    power = power_iteration(simple_adjacency).scores
    eigen = pagerank_eig(simple_adjacency)
    assert np.allclose(power, eigen, atol=1e-7)


@pytest.mark.parametrize("seed", range(6))
def test_matches_eigenvector_random_graphs(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(3, 9)
    a = (rng.random((n, n)) < 0.4).astype(float)
    np.fill_diagonal(a, 0)  # drop self-loops for variety
    power = power_iteration(a, tol=1e-13).scores
    eigen = pagerank_eig(a)
    assert np.allclose(power, eigen, atol=1e-6)


def test_dangling_node_matches_eig():
    # C is dangling (row of zeros).
    a = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 0]], dtype=float)
    power = power_iteration(a, tol=1e-13).scores
    eigen = pagerank_eig(a)
    assert power.sum() == pytest.approx(1.0)
    assert np.allclose(power, eigen, atol=1e-7)


def test_damping_zero_is_uniform(simple_adjacency):
    # With no link-following, the surfer always teleports -> uniform.
    scores = pagerank(simple_adjacency, damping=0.0)
    assert np.allclose(scores, 0.25)


def test_personalization_biases_result(simple_adjacency):
    v = np.array([1.0, 0.0, 0.0, 0.0])  # teleport only to node A
    scores = pagerank(simple_adjacency, personalization=v)
    assert scores.sum() == pytest.approx(1.0)
    eigen = pagerank_eig(simple_adjacency, personalization=v)
    assert np.allclose(scores, eigen, atol=1e-7)
    # Teleporting only to A concentrates rank near A relative to uniform.
    uniform = pagerank(simple_adjacency)
    assert scores[0] > uniform[0]


def test_weighted_edges_matter():
    # Same structure, but A favors C 9:1 over B.
    a = np.array([[0, 1, 9], [0, 0, 0], [0, 0, 0]], dtype=float)
    scores = pagerank(a)
    b_idx, c_idx = 1, 2
    assert scores[c_idx] > scores[b_idx]


def test_custom_x0_converges_to_same_answer(simple_adjacency):
    default = power_iteration(simple_adjacency).scores
    biased = power_iteration(simple_adjacency, x0=np.array([10.0, 1.0, 1.0, 1.0])).scores
    assert np.allclose(default, biased, atol=1e-8)


def test_single_node():
    assert np.allclose(pagerank(np.array([[0.0]])), [1.0])
    assert np.allclose(pagerank(np.array([[1.0]])), [1.0])


def test_empty_graph():
    result = power_iteration(np.zeros((0, 0)))
    assert result.scores.shape == (0,)
    assert result.converged


def test_top_helper(simple_adjacency):
    result = power_iteration(simple_adjacency)
    top2 = result.top(2)
    assert len(top2) == 2
    assert result.scores[top2[0]] >= result.scores[top2[1]]


def test_teleport_uniform_default():
    v = core._teleport_vector(4, None)
    assert np.allclose(v, 0.25)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([1, 2, 3]),  # not 2D
        np.array([[1, 2, 3]]),  # not square
        np.array([[0, -1], [0, 0]]),  # negative
    ],
)
def test_invalid_adjacency_raises(bad):
    with pytest.raises(ValueError):
        power_iteration(bad)


def test_invalid_damping_raises(simple_adjacency):
    with pytest.raises(ValueError):
        power_iteration(simple_adjacency, damping=1.0)
    with pytest.raises(ValueError):
        google_matrix(simple_adjacency, damping=-0.1)


def test_invalid_personalization_raises(simple_adjacency):
    with pytest.raises(ValueError):
        pagerank(simple_adjacency, personalization=np.array([1.0, 2.0]))  # wrong length
    with pytest.raises(ValueError):
        pagerank(simple_adjacency, personalization=np.zeros(4))  # sums to zero
