"""Scale PageRank to a large graph with SciPy sparse matrices, and time it.

Builds a scale-free graph of tens of thousands of nodes, runs the sparse solver,
and reports timing plus convergence.  Saves ``figures/convergence.png`` (L1
residual per iteration) and prints a timing table.
"""

from __future__ import annotations

import time

import numpy as np
from _common import ACCENT, GRID, MUTED, apply_style, figure_path, plt

from pagerank import datasets
from pagerank.sparse import pagerank_sparse


def main() -> None:
    apply_style()

    sizes = [1_000, 10_000, 50_000]
    edges_per_node = 4
    rows = []
    residual_curve = None

    for n in sizes:
        t0 = time.perf_counter()
        graph = datasets.scale_free_graph(n, edges_per_node=edges_per_node, seed=0)
        adjacency = graph.adjacency_sparse()
        t_build = time.perf_counter() - t0

        t1 = time.perf_counter()
        result = pagerank_sparse(adjacency, tol=1e-10)
        t_solve = time.perf_counter() - t1

        rows.append((n, adjacency.nnz, result.iterations, t_build, t_solve))
        if n == sizes[-1]:
            residual_curve = result.residuals
            top = np.argsort(result.scores)[::-1][:5]
            top_nodes = [(int(i), float(result.scores[i])) for i in top]
            assert abs(result.scores.sum() - 1.0) < 1e-9

    print(f"{'nodes':>8} {'edges':>9} {'iters':>6} {'build(s)':>9} {'solve(s)':>9}")
    print("-" * 45)
    for n, nnz, iters, tb, ts in rows:
        print(f"{n:>8} {nnz:>9} {iters:>6} {tb:>9.3f} {ts:>9.3f}")

    print(f"\nTop nodes on the {sizes[-1]:,}-node graph (by PageRank):")
    for idx, score in top_nodes:
        print(f"  node {idx:<6} {score:.5f}")

    # Convergence plot for the largest graph.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(
        range(1, len(residual_curve) + 1),
        residual_curve,
        color=ACCENT,
        marker="o",
        markersize=4,
        linewidth=1.8,
    )
    ax.axhline(1e-10, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(
        "tolerance 1e-10",
        (1, 1e-10),
        textcoords="offset points",
        xytext=(4, 6),
        color=MUTED,
        fontsize=9,
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel("L1 residual  ‖rₖ₊₁ − rₖ‖₁")
    ax.grid(True, which="both", color=GRID, linewidth=0.6)
    ax.set_title(
        f"Power iteration converges geometrically\n"
        f"{sizes[-1]:,}-node scale-free graph, damping 0.85 — "
        f"{len(residual_curve)} iterations to 1e-10",
        fontsize=12,
        loc="left",
    )
    fig.tight_layout()
    out = figure_path("convergence.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
