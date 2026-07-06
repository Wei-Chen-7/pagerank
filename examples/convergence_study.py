"""How the damping factor controls convergence *and* the ranking itself.

Saves ``figures/damping_study.png``, a 2×2 panel:

* (top-left)  residual vs iteration for several damping factors — power
  iteration converges geometrically, and the slope steepens as damping drops.
* (top-right) the observed convergence rate equals the subdominant eigenvalue
  ``|λ₂|`` of the Google matrix, which sits on or below the theoretical bound
  ``|λ₂| ≤ d``.
* (bottom-left) iterations to reach 1e-10 grow sharply as ``d → 1``.
* (bottom-right) the ranking is not static: as damping rises from 0 the surfer
  follows links more and structure emerges — at ``d = 0`` everything is uniform,
  and past a threshold ``insight`` overtakes ``megablog``.

Takeaway: damping trades ranking fidelity (higher ``d`` = more faithful to link
structure) against convergence speed (higher ``d`` = slower). ``0.85`` is the
conventional compromise.
"""

from __future__ import annotations

import numpy as np

from _common import ACCENT, GRID, INK, MUTED, WARN, apply_style, figure_path
from pagerank import datasets
from pagerank.core import power_iteration, subdominant_eigenvalue

import matplotlib.pyplot as plt


def empirical_rate(residuals: list[float]) -> float:
    """Geometric mean of consecutive residual ratios in the clean middle band.

    Skips the initial transient and the floating-point floor so the estimate
    reflects the asymptotic decay rate, not startup or round-off.
    """
    r = np.asarray(residuals)
    band = r[(r < 1e-2) & (r > 1e-11)]
    if band.size < 4:
        return float("nan")
    ratios = band[1:] / band[:-1]
    return float(np.exp(np.mean(np.log(ratios))))


def main() -> None:
    apply_style()
    adjacency = datasets.web_of_influence().adjacency_matrix()

    dampings = [0.50, 0.70, 0.85, 0.95, 0.99]
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(dampings)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_res, ax_rate, ax_iters, ax_rank = axes.flat

    # (top-left) residual curves.
    for d, color in zip(dampings, colors):
        result = power_iteration(adjacency, damping=d, tol=1e-13, max_iter=5000)
        ax_res.semilogy(range(1, result.iterations + 1), result.residuals,
                        color=color, linewidth=1.8, label=f"d = {d:.2f}")
    ax_res.set_xlabel("iteration")
    ax_res.set_ylabel("L1 residual")
    ax_res.set_xlim(0, 160)
    ax_res.grid(True, which="both", color=GRID, linewidth=0.6)
    ax_res.legend(frameon=False, fontsize=9)
    ax_res.set_title("Geometric convergence, steeper for smaller damping", loc="left")

    # (top-right) rate vs damping: bound d, actual |λ₂|, empirical.
    fine = np.linspace(0.05, 0.995, 40)
    lambda2 = [subdominant_eigenvalue(adjacency, damping=d) for d in fine]
    emp = []
    for d in fine:
        res = power_iteration(adjacency, damping=d, tol=1e-13, max_iter=8000)
        emp.append(empirical_rate(res.residuals))
    ax_rate.plot(fine, fine, "--", color=MUTED, linewidth=1.4, label="bound  |λ₂| ≤ d")
    ax_rate.plot(fine, lambda2, color=ACCENT, linewidth=2.2, label="actual  |λ₂|")
    ax_rate.plot(fine, emp, "o", color=WARN, markersize=4, label="observed rate")
    ax_rate.set_xlabel("damping factor  d")
    ax_rate.set_ylabel("convergence rate per step")
    ax_rate.grid(True, color=GRID, linewidth=0.6)
    ax_rate.legend(frameon=False, fontsize=9, loc="upper left")
    ax_rate.set_title("The observed rate is |λ₂|, and |λ₂| ≤ d", loc="left")

    # (bottom-left) iterations to 1e-10 vs damping.
    iters = []
    for d in fine:
        res = power_iteration(adjacency, damping=d, tol=1e-10, max_iter=20000)
        iters.append(res.iterations)
    ax_iters.plot(fine, iters, color=INK, linewidth=2.0)
    ax_iters.axvline(0.85, color=WARN, linestyle=":", linewidth=1.4)
    ax_iters.annotate("d = 0.85", (0.85, max(iters) * 0.9), color=WARN,
                      fontsize=9, ha="right")
    ax_iters.set_xlabel("damping factor  d")
    ax_iters.set_ylabel("iterations to reach 1e-10")
    ax_iters.grid(True, color=GRID, linewidth=0.6)
    ax_iters.set_title("Cost blows up as d → 1", loc="left")

    # (bottom-right) how the ranking shifts with damping.
    grid = np.linspace(0.0, 0.99, 60)
    watch = ["guru", "insight", "megablog", "user01"]
    node_colors = {"guru": ACCENT, "insight": WARN, "megablog": INK, "user01": MUTED}
    graph = datasets.web_of_influence()
    tracks = {name: [] for name in watch}
    for d in grid:
        scores = graph.pagerank(damping=d)
        for name in watch:
            tracks[name].append(scores[name])
    for name in watch:
        ax_rank.plot(grid, tracks[name], color=node_colors[name], linewidth=2.0, label=name)
    # Mark where insight overtakes megablog.
    diff = np.array(tracks["insight"]) - np.array(tracks["megablog"])
    cross = np.where(np.diff(np.sign(diff)) > 0)[0]
    if cross.size:
        dc = grid[cross[0]]
        ax_rank.axvline(dc, color=GRID, linewidth=1.2)
        ax_rank.annotate(f"insight overtakes\nmegablog (d ≈ {dc:.2f})",
                         (dc, 0.26), fontsize=9, ha="left", color=INK)
    ax_rank.axhline(1 / graph.number_of_nodes(), color=GRID, linestyle="--", linewidth=1)
    ax_rank.annotate("uniform 1/n", (0.01, 1 / graph.number_of_nodes()),
                     textcoords="offset points", xytext=(2, 4), color=MUTED, fontsize=8)
    ax_rank.set_xlabel("damping factor  d")
    ax_rank.set_ylabel("PageRank")
    ax_rank.grid(True, color=GRID, linewidth=0.6)
    ax_rank.legend(frameon=False, fontsize=9, loc="upper left")
    ax_rank.set_title("Damping reshapes the ranking, not just its speed", loc="left")

    fig.suptitle("The damping factor: one knob, two effects", fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = figure_path("damping_study.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")

    print(f"\n{'d':>5} {'|λ₂|':>8} {'observed':>9} {'iters→1e-10':>12}")
    print("-" * 38)
    for d in dampings:
        l2 = subdominant_eigenvalue(adjacency, damping=d)
        res = power_iteration(adjacency, damping=d, tol=1e-10, max_iter=20000)
        print(f"{d:>5.2f} {l2:>8.4f} {empirical_rate(res.residuals):>9.4f} {res.iterations:>12}")


if __name__ == "__main__":
    main()
