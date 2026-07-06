"""Tests for the bundled example graphs -- including the headline
in-degree-vs-PageRank disagreement."""

import numpy as np
import pytest

from pagerank import datasets


def test_tiny_graph_shape():
    g = datasets.tiny_graph()
    assert g.number_of_nodes() == 4
    assert set(g.nodes) == {"A", "B", "C", "D"}


def test_web_of_influence_indegree_and_pagerank_disagree():
    """The whole point of PageRank: a node made important by *one important
    link* (insight) outranks a node with many *unimportant* links (megablog),
    even though raw in-degree says the opposite."""
    g = datasets.web_of_influence(n_casual=8)
    scores = g.pagerank()
    indeg = g.in_degree()

    # Raw popularity: megablog is linked far more than insight.
    assert indeg["megablog"] > indeg["insight"]
    assert indeg["megablog"] == 8.0
    assert indeg["insight"] == 1.0

    # Eigenvector centrality flips it: insight wins.
    assert scores["insight"] > scores["megablog"]

    # So the two rankings genuinely invert this pair: in-degree puts megablog
    # above insight, PageRank does the reverse.
    by_indegree = [n for n, _ in sorted(indeg.items(), key=lambda kv: kv[1], reverse=True)]
    by_pagerank = [n for n, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    assert by_indegree.index("megablog") < by_indegree.index("insight")
    assert by_pagerank.index("insight") < by_pagerank.index("megablog")


def test_scale_free_graph_is_deterministic():
    g1 = datasets.scale_free_graph(200, edges_per_node=3, seed=42)
    g2 = datasets.scale_free_graph(200, edges_per_node=3, seed=42)
    assert np.array_equal(g1.adjacency_matrix(), g2.adjacency_matrix())


def test_scale_free_graph_size_and_out_degree():
    n, m = 500, 3
    g = datasets.scale_free_graph(n, edges_per_node=m, seed=0)
    assert g.number_of_nodes() == n
    out = g.out_degree()
    # Every node added after the seed clique has exactly m out-links.
    later = [out[i] for i in range(m + 1, n)]
    assert all(d == m for d in later)


def test_scale_free_graph_is_heavy_tailed():
    g = datasets.scale_free_graph(2000, edges_per_node=3, seed=0)
    indeg = np.array(list(g.in_degree().values()))
    # Preferential attachment: the max in-degree dwarfs the mean.
    assert indeg.max() > 10 * indeg.mean()


def test_scale_free_rejects_too_few_nodes():
    with pytest.raises(ValueError):
        datasets.scale_free_graph(3, edges_per_node=5)
