"""Tests for the DiGraph container and its ranking helpers."""

import numpy as np
import pytest

from pagerank.graph import DiGraph


def test_add_nodes_and_edges():
    g = DiGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2
    assert set(g.nodes) == {"a", "b", "c"}


def test_from_edges_with_weights_and_seed_nodes():
    g = DiGraph.from_edges([("a", "b", 2.0), ("a", "c")], nodes=["z", "a"])
    # 'z' is seeded first, so it takes index 0 and stays isolated.
    assert g.nodes[0] == "z"
    assert g.out_degree()["a"] == pytest.approx(3.0)
    assert g.out_degree()["z"] == 0.0


def test_repeated_edge_accumulates_weight():
    g = DiGraph()
    g.add_edge("a", "b")
    g.add_edge("a", "b", 2.0)
    assert g.adjacency_matrix()[0, 1] == pytest.approx(3.0)


def test_adjacency_matrix_layout():
    g = DiGraph.from_edges([("a", "b"), ("b", "a"), ("b", "c")])
    a = g.adjacency_matrix()
    expected = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=float)
    assert np.array_equal(a, expected)


def test_adjacency_sparse_matches_dense():
    g = DiGraph.from_edges([("a", "b"), ("b", "a"), ("b", "c", 2.0)])
    assert np.array_equal(g.adjacency_sparse().toarray(), g.adjacency_matrix())


def test_in_and_out_degree():
    g = DiGraph.from_edges([("a", "c"), ("b", "c"), ("c", "a")])
    assert g.in_degree() == {"a": 1.0, "b": 0.0, "c": 2.0}
    assert g.out_degree() == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_pagerank_dict_sums_to_one():
    g = DiGraph.from_edges([("a", "b"), ("b", "c"), ("c", "a")])
    scores = g.pagerank()
    assert set(scores) == {"a", "b", "c"}
    assert sum(scores.values()) == pytest.approx(1.0)
    # A symmetric 3-cycle gives every node equal rank.
    assert all(abs(v - 1 / 3) < 1e-9 for v in scores.values())


def test_rank_is_sorted_descending():
    g = DiGraph.from_edges([("a", "c"), ("b", "c"), ("c", "a")])
    ranking = g.rank()
    scores = [s for _, s in ranking]
    assert scores == sorted(scores, reverse=True)
    assert ranking[0][0] == "c"  # everyone points to c


def test_sparse_and_dense_paths_agree():
    g = DiGraph.from_edges([("a", "b"), ("a", "c"), ("b", "c"), ("c", "a")])
    dense = g.pagerank(sparse=False)
    sparse = g.pagerank(sparse=True)
    assert all(dense[n] == pytest.approx(sparse[n], abs=1e-9) for n in g.nodes)


def test_personalization_by_node_label():
    g = DiGraph.from_edges([("a", "b"), ("b", "c"), ("c", "a")])
    scores = g.pagerank(personalization={"a": 1.0})
    assert sum(scores.values()) == pytest.approx(1.0)
    # Teleporting to 'a' pushes rank around the cycle toward b (a's out-link).
    assert scores["b"] > scores["c"]


def test_negative_weight_raises():
    g = DiGraph()
    with pytest.raises(ValueError):
        g.add_edge("a", "b", -1.0)


def test_from_edges_rejects_malformed_edge():
    with pytest.raises(ValueError):
        DiGraph.from_edges([("a", "b", 1.0, 2.0)])
