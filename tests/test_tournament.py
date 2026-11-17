"""Tournament runner, statistics and the report builder's guards.

Owner: Jainam Parekh Naresh, Reviewer: Manas Tripathi
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from resource_war.report_builder import ReportError, render
from resource_war.tournament.runner import play_match, side_swapped_specs
from resource_war.tournament.stats import bootstrap_mean, cluster_bootstrap_mean, verdict, wilson_interval


def test_wilson_matches_textbook_value() -> None:
    iv = wilson_interval(8, 10)
    assert iv.estimate == 0.8
    assert iv.low == pytest.approx(0.4902, abs=1e-4) and iv.high == pytest.approx(0.9433, abs=1e-4)
    zero = wilson_interval(0, 50)
    assert zero.low == 0.0 and 0 < zero.high < 0.1  # stays inside [0, 1], unlike p +- 1.96 SE
    with pytest.raises(ValueError):
        wilson_interval(3, 0)


def test_bootstrap_is_deterministic_and_collapses_on_constant_data() -> None:
    a = cluster_bootstrap_mean([[1.0, 3.0], [2.0, 2.0]], np.random.default_rng(1), 500)
    b = cluster_bootstrap_mean([[1.0, 3.0], [2.0, 2.0]], np.random.default_rng(1), 500)
    assert a == b and a.estimate == 2.0 and a.low == a.high == 2.0
    spread = bootstrap_mean(list(range(100)), np.random.default_rng(2), 2000)
    assert spread.low < 49.5 < spread.high


def test_decision_rule() -> None:
    from resource_war.tournament.stats import Interval

    assert verdict(Interval(0.5, 0.1, 0.9)) == "first beats second"
    assert verdict(Interval(-0.5, -0.9, -0.1)) == "second beats first"
    assert verdict(Interval(0.1, -0.2, 0.4)) == "no detectable difference"


def test_side_swapped_rows_are_reported_from_the_pairings_point_of_view(configs) -> None:
    specs = side_swapped_specs("t", "equilibrium", 1, "fixed_mixed", 3, [11])
    rows = [play_match(s, configs)["game"] for s in specs]
    assert [r["swapped"] for r in rows] == [False, True]
    for r in rows:
        assert (r["first_kind"], r["first_depth"], r["second_kind"], r["second_depth"]) == ("equilibrium", 1, "fixed_mixed", 3)
        assert r["first_win"] + r["second_win"] + r["tie"] == 1
        assert r["score_diff"] == pytest.approx(r["first_score"] - r["second_score"])


def test_report_builder_values_tables_and_guards(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "verdicts.json").write_text(json.dumps({"H": {"status": "supported", "ratio": 0.12345, "items": [{"x": 7}]}}), encoding="utf-8")
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "t.md").write_text("| a |\n|---|\n| 1 |\n", encoding="utf-8")
    text = render("{{value:verdicts.H.ratio|.2f}} {{value:verdicts.H.items.0.x}} {{check:verdicts.H.status == supported}}\n{{table:t}}", results, generated)
    assert text.startswith("0.12 7 ") and "| 1 |" in text
    with pytest.raises(ReportError, match="no longer matches the data"):
        render("{{check:verdicts.H.status == refuted}}", results, generated)
    with pytest.raises(ReportError, match="no key"):
        render("{{value:verdicts.H.missing}}", results, generated)
