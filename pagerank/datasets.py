"""Example graphs used by the tests and the ``examples/`` scripts."""

from __future__ import annotations

import numpy as np

from .graph import DiGraph

__all__ = ["tiny_graph", "dangling_graph", "web_of_influence", "scale_free_graph"]


def tiny_graph() -> DiGraph:
    """A four-node graph small enough to reason about by hand.

    ::

        A -> B, A -> C, B -> C, C -> A, D -> C

    ``C`` is linked by everyone (A, B, D) and inherits A's weight through the
    C<->A loop, so it ranks first.  ``D`` links out but nothing links to it, so
    it ranks last.  Expected order: ``C > A > B > D``.
    """
    return DiGraph.from_edges(
        [("A", "B"), ("A", "C"), ("B", "C"), ("C", "A"), ("D", "C")],
        nodes=["A", "B", "C", "D"],
    )


def dangling_graph() -> DiGraph:
    """Two nodes where ``B`` is dangling: ``A -> B`` and ``B`` links nowhere.

    Exercises dangling-node handling; ``B`` should outrank ``A``.
    """
    return DiGraph.from_edges([("A", "B")], nodes=["A", "B"])


def web_of_influence(n_casual: int = 8) -> DiGraph:
    """A story graph where in-degree and PageRank disagree.

    * ``n_casual`` casual users each link to both ``megablog`` and ``guru``.
    * ``megablog`` also links to ``guru`` (it respects the guru).
    * ``guru`` links only to ``insight`` -- a single niche post.

    So ``megablog`` has a large in-degree (all the casual users) while
    ``insight`` has in-degree 1.  Yet ``guru`` is highly ranked and pours nearly
    all of its PageRank into its sole out-link, so ``insight`` ends up *above*
    ``megablog``.  Being vouched for by something important beats being linked by
    a crowd of nobodies -- exactly what eigenvector centrality captures and raw
    link-counting misses.
    """
    edges: list[tuple] = []
    for k in range(1, n_casual + 1):
        user = f"user{k:02d}"
        edges.append((user, "megablog"))
        edges.append((user, "guru"))
    edges.append(("megablog", "guru"))
    edges.append(("guru", "insight"))
    nodes = [f"user{k:02d}" for k in range(1, n_casual + 1)]
    nodes += ["megablog", "guru", "insight"]
    return DiGraph.from_edges(edges, nodes=nodes)


def scale_free_graph(
    n_nodes: int,
    edges_per_node: int = 3,
    seed: int | None = None,
) -> DiGraph:
    """A directed scale-free graph via preferential attachment.

    Each new node points to ``edges_per_node`` existing nodes chosen with
    probability proportional to their current in-degree ("rich get richer"),
    which yields the heavy-tailed in-degree distribution seen on the real web --
    a realistic stress test for the sparse solver.  Deterministic given ``seed``.
    """
    if n_nodes < edges_per_node + 1:
        raise ValueError("n_nodes must exceed edges_per_node")
    rng = np.random.default_rng(seed)

    # `targets` is a multiset: each node id appears once per in-link it has
    # received, so uniform sampling from it is preferential attachment.
    seeds = list(range(edges_per_node + 1))
    targets: list[int] = []
    edges: list[tuple[int, int]] = []
    for a in seeds:
        for b in seeds:
            if a != b:
                edges.append((a, b))
                targets.append(b)

    for new_node in range(edges_per_node + 1, n_nodes):
        chosen: set[int] = set()
        while len(chosen) < edges_per_node:
            chosen.add(targets[rng.integers(len(targets))])
        for t in chosen:
            edges.append((new_node, t))
            targets.append(t)
        targets.extend([new_node] * edges_per_node)

    return DiGraph.from_edges(edges, nodes=list(range(n_nodes)))
