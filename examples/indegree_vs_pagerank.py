"""Show where PageRank and raw in-degree disagree, and why it matters.

Saves ``figures/indegree_vs_pagerank.png`` -- a slope chart: each node's rank by
in-degree on the left, by PageRank on the right.  Lines that cross are nodes the
two methods disagree about.  The headline crossing is 'insight' (one great
in-link) overtaking 'megablog' (a crowd of weak in-links).
"""

from __future__ import annotations

from _common import ACCENT, MUTED, WARN, apply_style, figure_path, plt

from pagerank import datasets


def _ranks(scores: dict) -> dict:
    """Map node -> 1-based rank (1 = most important)."""
    order = sorted(scores, key=lambda n: scores[n], reverse=True)
    return {node: i + 1 for i, node in enumerate(order)}


def main() -> None:
    apply_style()
    graph = datasets.web_of_influence(n_casual=8)
    pagerank = graph.pagerank()
    indegree = graph.in_degree()

    pr_rank = _ranks(pagerank)
    in_rank = _ranks(indegree)

    highlight = {"insight", "megablog"}
    fig, ax = plt.subplots(figsize=(8.5, 6))

    for node in graph.nodes:
        y_left, y_right = in_rank[node], pr_rank[node]
        crossing = node in highlight
        color = ACCENT if node == "insight" else WARN if node == "megablog" else MUTED
        ax.plot(
            [0, 1],
            [y_left, y_right],
            color=color,
            alpha=0.95 if crossing else 0.35,
            linewidth=2.6 if crossing else 1.2,
            zorder=3 if crossing else 1,
            marker="o",
            markersize=7 if crossing else 4,
        )
        if crossing or node in ("guru",):
            ax.annotate(
                node,
                (0, y_left),
                textcoords="offset points",
                xytext=(-8, 0),
                ha="right",
                va="center",
                fontweight="bold" if crossing else "normal",
                color=color,
            )
            ax.annotate(
                node,
                (1, y_right),
                textcoords="offset points",
                xytext=(8, 0),
                ha="left",
                va="center",
                fontweight="bold" if crossing else "normal",
                color=color,
            )

    ax.set_xlim(-0.35, 1.35)
    ax.invert_yaxis()  # rank 1 at the top
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Rank by\nin-degree", "Rank by\nPageRank"], fontsize=11)
    ax.set_yticks([])
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_title(
        "Same graph, two rankings — the lines that cross are the disagreements\n"
        "'insight' has in-degree 1 but PageRank pulls it above 'megablog' "
        "(in-degree 8)",
        fontsize=12,
        loc="left",
    )
    fig.tight_layout()
    out = figure_path("indegree_vs_pagerank.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")

    print(f"\n{'node':>9} | {'in-degree':>9} {'in-rank':>7} | {'pagerank':>9} {'pr-rank':>7}")
    print("-" * 52)
    for node in sorted(graph.nodes, key=lambda n: pagerank[n], reverse=True):
        print(
            f"{node:>9} | {indegree[node]:>9.1f} {in_rank[node]:>7} | "
            f"{pagerank[node]:>9.4f} {pr_rank[node]:>7}"
        )
    print(
        "\nDisagreement: insight (in-degree 1) outranks megablog (in-degree 8) "
        "because its single link comes from the high-PageRank guru."
    )


if __name__ == "__main__":
    main()
