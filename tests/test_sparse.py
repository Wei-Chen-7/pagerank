"""Tests for the SciPy-sparse solver -- it must agree with the dense one."""

import numpy as np
import pytest
import scipy.sparse as sp

from pagerank import datasets
from pagerank.core import power_iteration
from pagerank.sparse import pagerank_sparse


def _dense(graph):
    return power_iteration(graph.adjacency_matrix(), tol=1e-13).scores


def test_sparse_matches_dense_tiny():
    g = datasets.tiny_graph()
    sparse = pagerank_sparse(g.adjacency_sparse(), tol=1e-13).scores
    assert np.allclose(sparse, _dense(g), atol=1e-9)


def test_sparse_matches_dense_with_dangling():
    g = datasets.dangling_graph()
    sparse = pagerank_sparse(g.adjacency_sparse(), tol=1e-13).scores
    assert np.allclose(sparse, _dense(g), atol=1e-9)


def test_sparse_matches_dense_web():
    g = datasets.web_of_influence()
    sparse = pagerank_sparse(g.adjacency_sparse(), tol=1e-13).scores
    assert np.allclose(sparse, _dense(g), atol=1e-9)


def test_sparse_personalization_matches_dense():
    g = datasets.tiny_graph()
    v = np.array([1.0, 0.0, 0.0, 0.0])
    sparse = pagerank_sparse(g.adjacency_sparse(), personalization=v, tol=1e-13).scores
    dense = power_iteration(g.adjacency_matrix(), personalization=v, tol=1e-13).scores
    assert np.allclose(sparse, dense, atol=1e-9)


def test_sparse_accepts_dense_input():
    a = datasets.tiny_graph().adjacency_matrix()
    result = pagerank_sparse(a)
    assert result.scores.sum() == pytest.approx(1.0)


def test_sparse_large_graph_is_valid_distribution():
    g = datasets.scale_free_graph(5000, edges_per_node=3, seed=1)
    result = pagerank_sparse(g.adjacency_sparse(), tol=1e-9)
    assert result.converged
    assert result.scores.sum() == pytest.approx(1.0)
    assert (result.scores >= 0).all()


def test_sparse_rejects_negative_entries():
    a = sp.csr_matrix(np.array([[0.0, -1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError):
        pagerank_sparse(a)


def test_sparse_rejects_non_square():
    a = sp.csr_matrix(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        pagerank_sparse(a)
