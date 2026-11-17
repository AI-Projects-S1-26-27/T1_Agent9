"""Live demo: robustness cases, CLI commands, and a headless run of the Streamlit app.

Owner: Jainam Parekh Naresh, Reviewer: Rohit Gangil
"""

from __future__ import annotations

from pathlib import Path

import pytest

from resource_war.demo import cli
from resource_war.demo.edge_cases import CASES, run_cases

APP = Path(__file__).resolve().parents[1] / "src" / "resource_war" / "demo" / "app.py"


def test_every_edge_case_is_rejected_with_a_message() -> None:
    results = run_cases()
    assert len(results) == len(CASES) >= 15
    for name, message in results:
        assert "NOT REJECTED" not in message, name
        assert ":" in message and "Traceback" not in message


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze", "--value", "6"],
        ["analyze", "--value", "4", "--attack-cost", "2"],
        ["battle", "--depths", "1", "5", "--seed", "3"],
        ["play", "--agents", "equilibrium", "fixed_mixed", "--depths", "3", "3", "--seed", "7", "--cards", "first"],
        ["mini", "--agents", "expected_utility", "fixed_mixed", "--games", "4"],
    ],
)
def test_cli_commands_succeed(argv) -> None:
    assert cli.main(argv) == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["play", "--depths", "3", "4"],
        ["play", "--seed", "-1"],
        ["analyze", "--value", "abc"],
        ["analyze", "--value", "6", "--p-row", "0.8", "--p-col", "0.5"],
        ["battle", "--depths", "0", "5"],
        ["mini", "--games", "1000"],
        ["play", "--payoffs", "no_such_file.yaml"],
    ],
)
def test_cli_rejects_bad_input_without_traceback(argv, capsys) -> None:
    assert cli.main(argv) == 2
    out = capsys.readouterr().out
    assert "Error:" in out and "Traceback" not in out


def test_cli_custom_matrix_file(tmp_path) -> None:
    good = tmp_path / "pd.yaml"
    good.write_text("schema_version: 1\nrow: [[3, 0], [5, 1]]\ncol: [[3, 5], [0, 1]]\nrow_labels: [C, D]\ncol_labels: [C, D]\n", encoding="utf-8")
    assert cli.main(["analyze", "--matrix", str(good)]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\nrow: [[3, 0], [5]]\ncol: [[3, 5], [0, 1]]\n", encoding="utf-8")
    assert cli.main(["analyze", "--matrix", str(bad)]) == 2


def test_streamlit_app_runs_headless() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 5
    at.button[0].click().run()  # New game
    assert not at.exception
    next_encounter = next(b for b in at.button if b.label == "Next encounter")
    next_encounter.click().run()
    assert not at.exception
    assert "Turn" in " ".join(m.value for m in at.markdown)


def test_streamlit_custom_matrix_error_is_shown_not_raised() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.text_input[0].set_value("3, 0; 5").run()  # ragged row
    next(b for b in at.button if b.label == "Analyse custom game").click().run()
    assert not at.exception
    assert any("ragged" in e.value for e in at.error)
