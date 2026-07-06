"""Tests for the command-line interface."""

import json

import pytest

from pagerank.cli import _parse_bias, main, read_edgelist


def test_read_edgelist_two_and_three_columns():
    lines = [
        "# a comment",
        "",
        "A B",
        "A C 2.0",
        "C,A",  # comma-separated is fine too
    ]
    g = read_edgelist(lines)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 3
    assert g.adjacency_matrix()[g.index_of("A"), g.index_of("C")] == pytest.approx(2.0)


def test_read_edgelist_custom_delimiter():
    g = read_edgelist(["A|B", "B|C"], delimiter="|")
    assert g.number_of_edges() == 2


def test_read_edgelist_bad_weight():
    with pytest.raises(ValueError, match="not a number"):
        read_edgelist(["A B notanumber"])


def test_read_edgelist_wrong_field_count():
    with pytest.raises(ValueError, match="expected"):
        read_edgelist(["A B C D"])


def test_read_edgelist_empty():
    with pytest.raises(ValueError, match="no edges"):
        read_edgelist(["# only comments", ""])


def test_parse_bias():
    assert _parse_bias(None) is None
    assert _parse_bias(["A", "B=2.5"]) == {"A": 1.0, "B": 2.5}


@pytest.fixture
def edgelist_file(tmp_path):
    path = tmp_path / "g.edgelist"
    path.write_text("A B\nA C\nB C\nC A\nD C\n")
    return str(path)


def test_main_table_output(edgelist_file, capsys):
    code = main([edgelist_file])
    assert code == 0
    out = capsys.readouterr().out
    assert "rank" in out and "node" in out and "score" in out
    # C should be ranked first in this graph.
    body = [ln for ln in out.splitlines() if ln.strip().startswith("1")]
    assert body and "C" in body[0]


def test_main_json_and_top(edgelist_file, capsys):
    code = main([edgelist_file, "--format", "json", "--top", "2"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    assert payload[0]["node"] == "C"
    assert payload[0]["score"] > payload[1]["score"]


def test_main_sparse_matches_dense(edgelist_file, capsys):
    main([edgelist_file, "--format", "json"])
    dense = json.loads(capsys.readouterr().out)
    main([edgelist_file, "--format", "json", "--sparse"])
    sparse = json.loads(capsys.readouterr().out)
    for d, s in zip(dense, sparse):
        assert d["node"] == s["node"]
        assert d["score"] == pytest.approx(s["score"], abs=1e-9)


def test_main_personalization_changes_ranking(edgelist_file, capsys):
    main([edgelist_file, "--format", "json", "--bias", "D"])
    scores = {row["node"]: row["score"] for row in json.loads(capsys.readouterr().out)}
    # Teleporting to D lifts D well above its usual last place.
    assert scores["D"] > min(scores.values())


def test_main_missing_file_returns_error(capsys):
    code = main(["/no/such/file.edgelist"])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_main_reads_stdin(monkeypatch, capsys):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("X Y\nY X\n"))
    code = main(["-", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["node"] for row in payload} == {"X", "Y"}
