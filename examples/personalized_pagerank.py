"""Personalized (topic-biased) PageRank.

Standard PageRank teleports uniformly.  If instead the surfer always teleports
back to one favorite node (or a topic set), the whole ranking tilts toward pages
near that node -- this is the engine behind "related pages" and topic-sensitive
search.  Saves ``figures/personalized.png`` comparing uniform PageRank with two
personalized runs on the same graph.
"""

from __future__ import annotations

import numpy as np
from _common import ACCENT, MUTED, WARN, apply_style, figure_path, plt

from pagerank import datasets


def main() -> None:
    apply_style()
    graph = datasets.web_of_influence(n_casual=8)
    nodes = ["guru", "insight", "megablog"] + [f"user{k:02d}" for k in range(1, 9)]

    uniform = graph.pagerank()
    to_megablog = graph.pagerank(personalization={"megablog": 1.0})
    to_user = graph.pagerank(personalization={"user01": 1.0})

    series = [
        ("uniform teleport", uniform, MUTED),
        ("teleport → megablog", to_megablog, WARN),
        ("teleport → user01", to_user, ACCENT),
    ]

    x = np.arange(len(nodes))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, (label, scores, color) in zip((-1, 0, 1), series):
        ax.bar(
            x + offset * width,
            [scores[n] for n in nodes],
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(nodes, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("PageRank")
    ax.legend(frameon=False)
    ax.set_title(
        "Personalized PageRank tilts the ranking toward the teleport target\n"
        "biasing toward 'user01' lifts user01 and its neighbors; "
        "biasing toward 'megablog' lifts megablog and the guru it links to",
        fontsize=12,
        loc="left",
    )
    fig.tight_layout()
    out = figure_path("personalized.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")

    print(f"\n{'node':>9} | {'uniform':>8} | {'→megablog':>9} | {'→user01':>8}")
    print("-" * 44)
    for n in nodes:
        print(f"{n:>9} | {uniform[n]:>8.4f} | {to_megablog[n]:>9.4f} | {to_user[n]:>8.4f}")


if __name__ == "__main__":
    main()
