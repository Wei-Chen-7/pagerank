"""Command-line interface: rank the nodes of an edge-list file by PageRank.

Reads a directed graph as a two- or three-column edge list and prints the nodes
ordered by PageRank.  Installed as the ``pagerank`` console script (see
``pyproject.toml``); also runnable as ``python -m pagerank``.

Edge-list format (whitespace- or comma-separated, ``#`` comments, blank lines
ignored)::

    # source target [weight]
    A B
    A C
    C A 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

from .graph import DiGraph


def read_edgelist(lines: Iterable[str], delimiter: str | None = None) -> DiGraph:
    """Parse an iterable of text lines into a :class:`DiGraph`.

    Each non-comment, non-blank line must have 2 fields (``source target``) or 3
    (``source target weight``).  With ``delimiter=None`` fields are split on any
    whitespace or comma; otherwise on the given delimiter.
    """
    graph = DiGraph()
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if delimiter is None:
            fields = line.replace(",", " ").split()
        else:
            fields = [f.strip() for f in line.split(delimiter) if f.strip()]
        if len(fields) == 2:
            graph.add_edge(fields[0], fields[1])
        elif len(fields) == 3:
            try:
                weight = float(fields[2])
            except ValueError as exc:
                raise ValueError(f"line {lineno}: weight '{fields[2]}' is not a number") from exc
            graph.add_edge(fields[0], fields[1], weight)
        else:
            raise ValueError(
                f"line {lineno}: expected 'source target [weight]', got {len(fields)} fields"
            )
    if graph.number_of_nodes() == 0:
        raise ValueError("no edges found in input")
    return graph


def _parse_bias(entries: list[str] | None) -> dict | None:
    """Turn ``["A", "B=2"]`` into ``{"A": 1.0, "B": 2.0}`` for personalization."""
    if not entries:
        return None
    bias: dict[str, float] = {}
    for entry in entries:
        if "=" in entry:
            node, _, weight = entry.partition("=")
            bias[node.strip()] = float(weight)
        else:
            bias[entry.strip()] = 1.0
    return bias


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pagerank",
        description="Rank the nodes of a directed graph (edge-list file) by PageRank.",
    )
    parser.add_argument(
        "edgelist",
        help="path to an edge-list file, or '-' to read from stdin",
    )
    parser.add_argument(
        "-d", "--damping", type=float, default=0.85,
        help="damping factor in [0, 1) (default: 0.85)",
    )
    parser.add_argument(
        "-n", "--top", type=int, default=None,
        help="show only the top N nodes (default: all)",
    )
    parser.add_argument(
        "--sparse", action="store_true",
        help="use the SciPy-sparse solver (preferable for large graphs)",
    )
    parser.add_argument(
        "--bias", action="append", metavar="NODE[=WEIGHT]",
        help="personalize: teleport toward NODE (repeatable; default weight 1)",
    )
    parser.add_argument(
        "--delimiter", default=None,
        help="field delimiter (default: any whitespace or comma)",
    )
    parser.add_argument("--tol", type=float, default=1e-10, help="convergence tolerance")
    parser.add_argument("--max-iter", type=int, default=1000, help="maximum iterations")
    parser.add_argument(
        "--format", choices=("table", "json"), default="table",
        help="output format (default: table)",
    )
    return parser


def _format_table(ranking: list[tuple], total: int) -> str:
    width = max((len(str(node)) for node, _ in ranking), default=4)
    width = max(width, len("node"))
    lines = [f"{'rank':>4}  {'node':<{width}}  {'score':>10}"]
    lines.append(f"{'-' * 4}  {'-' * width}  {'-' * 10}")
    for i, (node, score) in enumerate(ranking, start=1):
        lines.append(f"{i:>4}  {str(node):<{width}}  {score:>10.6f}")
    if len(ranking) < total:
        lines.append(f"      ... ({total - len(ranking)} more)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.edgelist == "-":
            graph = read_edgelist(sys.stdin, delimiter=args.delimiter)
        else:
            with open(args.edgelist, "r", encoding="utf-8") as handle:
                graph = read_edgelist(handle, delimiter=args.delimiter)

        bias = _parse_bias(args.bias)
        ranking = graph.rank(
            damping=args.damping,
            personalization=bias,
            tol=args.tol,
            max_iter=args.max_iter,
            sparse=args.sparse,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    total = len(ranking)
    shown = ranking[: args.top] if args.top is not None else ranking

    if args.format == "json":
        print(json.dumps([{"node": str(n), "score": s} for n, s in shown], indent=2))
    else:
        print(
            f"# {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
            f"damping {args.damping}"
        )
        print(_format_table(shown, total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
