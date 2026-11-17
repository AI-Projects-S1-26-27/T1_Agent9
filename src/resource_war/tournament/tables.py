"""Markdown tables for the report, generated from the result files only.

Owner: Manas Tripathi, Reviewer: Jainam Parekh Naresh

The report never contains a hand-typed result. Each table here reads
experiments/results/*.json|csv and writes report/generated/<name>.md, which
the report includes. Rerunning ``python -m resource_war.reproduce``
regenerates every table from scratch.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _ci(iv: dict | None, digits: int = 3) -> str:
    if not iv:
        return "n/a"
    return f"{iv['estimate']:.{digits}f} [{iv['low']:.{digits}f}, {iv['high']:.{digits}f}]"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _load(results: Path, name: str):
    return json.loads((results / name).read_text(encoding="utf-8"))


def _mix(values: list[float]) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in values) + ")"


def verdict_table(results: Path) -> str:
    v = _load(results, "verdicts.json")
    rows = [
        ["H1a", "alpha-beta returns minimax's value AND move on every benchmark position, depths 1-5", v["H1"]["H1a"]["status"], f"{v['H1']['H1a']['comparisons']} comparisons"],
        ["H1b", "mean node ratio (alpha-beta / minimax) at depth 5 <= 0.5", v["H1"]["H1b"]["status"], f"ratio {_ci(v['H1']['H1b']['depth5_mean_ratio'])}"],
        ["Gate", f"plain minimax at depth 5 <= {v['gate']['gate_seconds']} s per decision", "passed" if v["gate"]["passed"] else "FAILED", f"max {v['gate']['minimax_max_seconds']:.3f} s, mean {v['gate']['minimax_mean_seconds']:.3f} s"],
        ["H2", "no significant DECREASE in battle win rate with depth (1 <= 3 <= 5)", v["H2"]["status"], f"proposal criterion (3 > 1 significant): {'met' if v['H2']['proposal_criterion_met'] else 'not met'}"],
        ["H3a", "deeper player wins more battles (CI of p_deep - p_shallow > 0)", v["H3"]["H3a"]["status"], ", ".join(f"{k}: {'yes' if ok else 'no'}" for k, ok in v["H3"]["H3a"]["checks"].items())],
        ["H3b", "deeper player: higher value, attacks less, expands more; matches closed form; nashpy agrees", v["H3"]["H3b"]["status"], ", ".join(f"{k}: {'yes' if ok else 'no'}" for k, ok in v["H3"]["H3b"]["checks"].items())],
        ["H3c", "deeper player's realized encounter payoff exceeds the shallower's (CI > 0)", v["H3"]["H3c"]["status"], ", ".join(f"{k}: {'yes' if ok else 'no'}" for k, ok in v["H3"]["H3c"]["checks"].items())],
        ["H4a", "at V = 6, 8 the Equilibrium agent earns LESS per encounter than FixedMixed and PureHeuristic", v["H4"]["H4a"]["status"], ", ".join(f"{k}: {'yes' if ok else 'no'}" for k, ok in v["H4"]["H4a"]["checks"].items())],
        ["H4b", "at V = 2, 4 Equilibrium ties PureHeuristic exactly and beats FixedMixed", v["H4"]["H4b"]["status"], ", ".join(f"{k}: {'yes' if ok else 'no'}" for k, ok in v["H4"]["H4b"]["checks"].items())],
        ["H4c", "ExpectedUtility outscores FixedMixed per encounter at every V", v["H4"]["H4c"]["status"], ", ".join(f"{k}: {'yes' if ok else 'no'}" for k, ok in v["H4"]["H4c"]["checks"].items())],
        ["E5", "structure rule (V < 2c pure (E,E); V = 2c degenerate; V > 2c cyclic and Pareto-dominated) holds at every grid point", v["E5"]["status"], f"{v['E5']['mismatches']} mismatches over {v['E5']['grid_points']} grid points"],
    ]
    return _table(["Prediction", "Pre-registered rule", "Outcome", "Evidence"], rows)


def pairing_table(results: Path, name: str) -> str:
    pairs = _load(results, name)["pairings"]
    rows = [
        [p["pairing"], p["games"], _ci(p["first_win_rate"]), _ci(p["second_win_rate"]), _ci(p["tie_rate"]), _ci(p["mean_score_diff"], 2), p["verdict"].replace("first", "first-named").replace("second", "second-named")]
        for p in pairs
    ]
    return _table(["Pairing (first vs second)", "Games", "First win rate", "Second win rate", "Tie rate", "Score diff (first - second)", "Verdict"], rows)


def metrics_table(results: Path, name: str) -> str:
    pairs = _load(results, name)["pairings"]
    rows = [
        [p["pairing"], f"{p['first_mean_score']:.2f} / {p['second_mean_score']:.2f}", f"{p['first_mean_resources']:.2f} / {p['second_mean_resources']:.2f}", f"{p['first_mean_territory']:.2f} / {p['second_mean_territory']:.2f}", f"{p['mean_encounters']:.1f}", f"{p['mean_battles']:.2f}"]
        for p in pairs
    ]
    return _table(["Pairing", "Mean score (first / second)", "Resources collected", "Territory held at end", "Encounters / game", "Battles / game"], rows)


def h4_table(results: Path) -> str:
    h4 = _load(results, "E1_summary.json")["h4"]
    rows = []
    for i, r in enumerate(h4["eq_vs_fm"]):
        rows.append(
            [f"{r['value']:g}"]
            + [f"{h4[k][i]['x_mean']:.3f} vs {h4[k][i]['y_mean']:.3f}; diff {_ci(h4[k][i]['diff'])} (n={h4[k][i]['encounters']})" for k in ("eq_vs_fm", "eq_vs_ph", "eu_vs_fm", "eu_vs_ph")]
        )
    return _table(["V", "Equilibrium vs FixedMixed", "Equilibrium vs PureHeuristic", "ExpectedUtility vs FixedMixed", "ExpectedUtility vs PureHeuristic"], rows)


def h2_table(results: Path) -> str:
    h2 = _load(results, "H2_summary.json")
    rows = [[f"depth {d}", s["battles"], _ci(s["win_rate"]), _ci(s["tie_rate"]), _ci(s["loss_rate"])] for d, s in sorted(h2["depths"].items(), key=lambda kv: int(kv[0]))]
    diffs = [[f"w({k.split('-')[0]}) - w({k.split('-')[1]})", _ci(iv)] for k, iv in h2["differences"].items()]
    joint = _load(results, "verdicts.json")["H2"]["joint_3_vs_5"]
    joint_rows = [[f"depth 3 {a}"] + [joint[f"{a}_{b}"] for b in ("win", "tie", "loss")] for a in ("win", "tie", "loss")]
    return (
        _table([f"Tested depth (vs depth {h2['opponent_depth']})", "Battles", "Win rate", "Tie rate", "Loss rate"], rows)
        + "\n"
        + _table(["Paired difference", "Estimate [95% CI]"], diffs)
        + "\nSame start and first mover, outcome at depth 3 (rows) vs depth 5 (columns):\n\n"
        + _table(["", "depth 5 win", "depth 5 tie", "depth 5 loss"], joint_rows)
    )


def e3_table(results: Path) -> str:
    v = _load(results, "verdicts.json")["H1"]["by_depth"]
    rows = [
        [d, s["positions"], f"{s['mean_minimax_nodes']:,.0f}", f"{s['mean_alphabeta_nodes']:,.0f}", f"{s['mean_ratio']:.3f}", f"{1 - s['mean_ratio']:.1%}", f"{s['mean_minimax_seconds'] * 1000:.2f}", f"{s['mean_alphabeta_seconds'] * 1000:.2f}", "yes" if s["all_same"] else "NO"]
        for d, s in sorted(v.items(), key=lambda kv: int(kv[0]))
    ]
    return _table(["Depth", "Positions", "Minimax nodes", "Alpha-beta nodes", "Node ratio", "Nodes saved", "Minimax ms", "Alpha-beta ms", "Same value and move"], rows)


def gate_table(results: Path) -> str:
    g = _load(results, "verdicts.json")["gate"]
    rows = [
        ["Full tree nodes, b = 8, depth 5 (analytic upper bound)", f"{g['estimate_full_tree_nodes']:,}"],
        ["Knuth-Moore best-case leaves (analytic)", f"{g['estimate_knuth_moore_best_case_leaves']:,}"],
        ["Measured minimax nodes at depth 5: mean / max", f"{g['minimax_mean_nodes']:,.0f} / {g['minimax_max_nodes']:,}"],
        ["Measured minimax seconds at depth 5: mean / 95th pct / max", f"{g['minimax_mean_seconds']:.3f} / {g['minimax_p95_seconds']:.3f} / {g['minimax_max_seconds']:.3f}"],
        ["Gate (max must be <= this)", f"{g['gate_seconds']} s: {'passed' if g['passed'] else 'FAILED'}"],
    ]
    return _table(["Quantity", "Value"], rows)


def e4_table(results: Path) -> str:
    pairs = _load(results, "E4_summary.json")["pairs"]
    rows = [
        [f"{p['deep']} vs {p['shallow']}", p["battles"], f"{p['p_deep']} = {_ci(p['p_deep_ci'])}", f"{p['p_shallow']} = {_ci(p['p_shallow_ci'])}", f"{p['tie_rate']:.3f}", _mix(p["deep_mix_float"]), _mix(p["shallow_mix_float"]), f"{p['deep_value']:.3f} / {p['shallow_value']:.3f}", _ci(p["h3c_diff"])]
        for p in pairs
    ]
    return _table(["Depths (deep vs shallow)", "Battles", "p_deep", "p_shallow", "Tie rate", "Deep mix (A, D, E)", "Shallow mix (A, D, E)", "Equilibrium value deep / shallow", "Realized payoff diff (deep - shallow)"], rows)


def e5_table(results: Path) -> str:
    s = _load(results, "E5_summary.json")
    rows = [[k, n] for k, n in s["structure_counts"].items()]
    deg = ", ".join(f"(V={v:g}, c={c:g}, d={d:g})" for v, c, d in s["degenerate_points"])
    return _table(["Structure", "Grid points"], rows) + f"\nDegenerate grid points: {deg}.\n\nMismatches with the pre-registered rule: {len(s['mismatches'])} of {s['grid_points']}.\n"


def p1_table(results: Path) -> str:
    with (results / "P1_encounter_frequency.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    body = [
        [r["placement"], r["targeting"], r["tie_break"], r["regrow_turns"] or "none", f"{float(r['mean_encounters']):.2f}", f"{float(r['share_games_with_encounter']):.2f}", f"{float(r['mean_value_at_encounter']):.2f}"]
        for r in rows
    ]
    return _table(["Placement", "Targeting", "Tie-break", "Regrow after", "Encounters / game", "Share of games with >= 1", "Mean V at encounter"], body)


def write_all(results: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "verdicts.md": verdict_table(results),
        "e1_pairings.md": pairing_table(results, "E1_summary.json"),
        "e1_metrics.md": metrics_table(results, "E1_summary.json"),
        "h4_per_value.md": h4_table(results),
        "e2_pairings.md": pairing_table(results, "E2_summary.json"),
        "e2_metrics.md": metrics_table(results, "E2_summary.json"),
        "h2_battle_depth.md": h2_table(results),
        "e3_benchmark.md": e3_table(results),
        "e3_gate.md": gate_table(results),
        "e4_coupling.md": e4_table(results),
        "e5_sensitivity.md": e5_table(results),
        "p1_frequency.md": p1_table(results),
    }
    paths = []
    for name, text in tables.items():
        path = out / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
