"""Draw the web-of-influence graph with node size proportional to PageRank.

Saves ``figures/graph.png``.  Uses NetworkX only for the force-directed layout
and arrow drawing -- the PageRank scores themselves come from our own solver.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from _common import GRID, INK, MUTED, apply_style, figure_path, plt

from pagerank import datasets


def main() -> None:
    apply_style()
    graph = datasets.web_of_influence(n_casual=8)
    scores = graph.pagerank()

    # Build a NetworkX view purely for layout + drawing.
    nxg = nx.DiGraph()
    nxg.add_nodes_from(graph.nodes)
    for u in graph.nodes:
        i = graph.index_of(u)
        for j, w in graph._out.get(i, {}).items():
            nxg.add_edge(u, graph.nodes[j], weight=w)

    pos = nx.spring_layout(nxg, seed=7, k=1.1, iterations=200)

    order = list(graph.nodes)
    pr = np.array([scores[n] for n in order])
    sizes = 300 + pr / pr.max() * 4200  # area ∝ PageRank
    colors = pr  # hue ∝ PageRank too

    fig, ax = plt.subplots(figsize=(9, 6.5))
    nx.draw_networkx_edges(
        nxg,
        pos,
        ax=ax,
        edge_color=MUTED,
        width=1.2,
        arrowsize=14,
        arrowstyle="-|>",
        alpha=0.7,
        connectionstyle="arc3,rad=0.06",
        node_size=sizes,
    )
    nodes = nx.draw_networkx_nodes(
        nxg,
        pos,
        ax=ax,
        nodelist=order,
        node_size=sizes,
        node_color=colors,
        cmap="viridis",
        edgecolors="white",
        linewidths=1.5,
    )
    # Label only the nodes that matter to the story; leave the crowd unlabeled.
    labels = {n: n for n in ("guru", "insight", "megablog")}
    nx.draw_networkx_labels(
        nxg, pos, labels=labels, ax=ax, font_size=11, font_color=INK, font_weight="bold"
    )

    cbar = fig.colorbar(nodes, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("PageRank", color=INK)
    cbar.outline.set_edgecolor(GRID)

    ax.set_title(
        "Web of influence — node size and color ∝ PageRank\n"
        "'insight' is small in the crowd but large in rank: the guru vouches for it",
        fontsize=12,
        loc="left",
    )
    ax.set_axis_off()
    fig.tight_layout()
    out = figure_path("graph.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")

    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print("\nPageRank (highest first):")
    for node, score in ranking:
        print(f"  {node:>9}  {score:.4f}")


if __name__ == "__main__":
    main()
