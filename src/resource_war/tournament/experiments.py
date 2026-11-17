"""Experiments E1, E2, E4, E5 and H2, and the pre-registered verdicts for H1-H4.

Owner: Manas Tripathi, Reviewer: Jainam Parekh Naresh

Every function writes its raw rows (CSV) and a summary (JSON) to an output
directory, and every verdict is computed by code from the decision rules
in docs/PREDICTIONS.md, never typed by hand. The report's numbers and
tables are generated from these files (tables.py, report builder).

Seed blocks keep experiments independent but reproducible:
E1 uses base_seed + 100000 + i, E2 base_seed + 200000 + i, H2 starts
base_seed + 300000, E4 starts base_seed + 400000 + pair index.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np

from resource_war.battle.play import play_battle
from resource_war.battle.state import random_start
from resource_war.config import DEFAULT_CONFIG_DIR, Configs
from resource_war.encounter.analysis import analyze
from resource_war.encounter.matrix import ATTACK, build_encounter_game, realized_gains_and_costs
from resource_war.equilibrium.crosscheck import crosscheck
from resource_war.equilibrium.selection import select_equilibrium
from resource_war.equilibrium.solve import solve_game
from resource_war.agents.base import sample_action
from resource_war.tournament.runner import MatchSpec, git_commit, run_matches, side_swapped_specs
from resource_war.tournament.stats import Interval, cluster_bootstrap_mean, verdict, wilson_interval

AGENT_LABELS = {
    "pure_heuristic": "PureHeuristic",
    "fixed_mixed": "FixedMixed",
    "equilibrium": "Equilibrium",
    "expected_utility": "ExpectedUtility",
    "depth_limited": "DepthLimited",
}


@dataclass
class Context:
    configs: Configs
    out: Path
    processes: int | None = None
    quick: bool = False

    @property
    def base_seed(self) -> int:
        return self.configs.experiments.base_seed

    def rng(self, salt: int) -> np.random.Generator:
        return np.random.default_rng([self.base_seed, salt])

    def n(self, configured: int, quick_value: int) -> int:
        return quick_value if self.quick else configured

    @property
    def confidence(self) -> float:
        return self.configs.experiments.confidence

    @property
    def resamples(self) -> int:
        return 2000 if self.quick else self.configs.experiments.bootstrap_resamples


# ------------------------------------------------------------------ io helpers
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")


def interval_dict(iv: Interval) -> dict[str, float]:
    return {"estimate": iv.estimate, "low": iv.low, "high": iv.high}


def metadata(ctx: Context) -> dict[str, Any]:
    files = sorted(DEFAULT_CONFIG_DIR.glob("*.yaml"))
    import numpy
    import nashpy

    return {
        "git_commit": git_commit(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick_mode": ctx.quick,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "nashpy": nashpy.__version__,
        "base_seed": ctx.base_seed,
        "config_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    }


# ------------------------------------------------------------------ tournament summaries
def summarize_pairings(games: list[dict[str, Any]], ctx: Context, salt: int) -> list[dict[str, Any]]:
    """Win rates (Wilson) and mean score difference (cluster bootstrap over seeds) per pairing."""
    out = []
    rng = ctx.rng(salt)
    for pairing in dict.fromkeys(g["pairing"] for g in games):
        rows = [g for g in games if g["pairing"] == pairing]
        n = len(rows)
        first_w = sum(r["first_win"] for r in rows)
        second_w = sum(r["second_win"] for r in rows)
        ties = sum(r["tie"] for r in rows)
        by_seed: dict[int, list[float]] = {}
        for r in rows:
            by_seed.setdefault(r["seed"], []).append(r["score_diff"])
        diff = cluster_bootstrap_mean(list(by_seed.values()), rng, ctx.resamples, ctx.confidence)
        out.append(
            {
                "pairing": pairing,
                "first": rows[0]["first_kind"],
                "second": rows[0]["second_kind"],
                "first_depth": rows[0]["first_depth"],
                "second_depth": rows[0]["second_depth"],
                "games": n,
                "first_win_rate": interval_dict(wilson_interval(first_w, n, ctx.confidence)),
                "second_win_rate": interval_dict(wilson_interval(second_w, n, ctx.confidence)),
                "tie_rate": interval_dict(wilson_interval(ties, n, ctx.confidence)),
                "mean_score_diff": interval_dict(diff),
                "verdict": verdict(diff),
                "first_mean_score": statistics.fmean(r["first_score"] for r in rows),
                "second_mean_score": statistics.fmean(r["second_score"] for r in rows),
                "first_mean_resources": statistics.fmean(r["first_resources"] for r in rows),
                "second_mean_resources": statistics.fmean(r["second_resources"] for r in rows),
                "first_mean_territory": statistics.fmean(r["first_territory"] for r in rows),
                "second_mean_territory": statistics.fmean(r["second_territory"] for r in rows),
                "mean_encounters": statistics.fmean(r["encounters"] for r in rows),
                "mean_battles": statistics.fmean(r["battles"] for r in rows),
            }
        )
    return out


def per_value_payoffs(encounters: list[dict[str, Any]], x: str, y: str, ctx: Context, salt: int) -> list[dict[str, Any]]:
    """Per-encounter payoffs of agent x vs agent y for each V, with a seed-clustered CI of (x - y)."""
    rng = ctx.rng(salt)
    rows = []
    relevant = [e for e in encounters if {e["first_kind"], e["second_kind"]} == {x, y}]
    for value in sorted({e["value"] for e in relevant}):
        clusters: dict[int, list[float]] = {}
        xs, ys = [], []
        for e in relevant:
            if e["value"] != value:
                continue
            xp, yp = (e["first_payoff"], e["second_payoff"]) if e["first_kind"] == x else (e["second_payoff"], e["first_payoff"])
            xs.append(xp)
            ys.append(yp)
            clusters.setdefault(e["seed"], []).append(xp - yp)
        diff = cluster_bootstrap_mean(list(clusters.values()), rng, ctx.resamples, ctx.confidence)
        rows.append({"x": x, "y": y, "value": value, "encounters": len(xs), "x_mean": statistics.fmean(xs), "y_mean": statistics.fmean(ys), "diff": interval_dict(diff)})
    return rows


def _tag_kinds(encounters: list[dict[str, Any]], games: list[dict[str, Any]]) -> None:
    kinds = {(g["pairing"], g["seed"], g["swapped"]): (g["first_kind"], g["second_kind"]) for g in games}
    for e in encounters:
        e["first_kind"], e["second_kind"] = kinds[(e["pairing"], e["seed"], e["swapped"])]


# ------------------------------------------------------------------ E1 and H4
def run_e1(ctx: Context) -> dict[str, Any]:
    sec = ctx.configs.experiments.section("E1_round_robin")
    depth = int(sec["battle_depth"])
    seeds = range(ctx.base_seed + 100000, ctx.base_seed + 100000 + ctx.n(int(sec["seeds_per_pairing"]), 20))
    specs: list[MatchSpec] = []
    for a, b in combinations(sec["agents"], 2):
        specs += side_swapped_specs("E1", a, depth, b, depth, seeds, f"{AGENT_LABELS[a]} vs {AGENT_LABELS[b]}")
    print(f"E1 round robin: {len(specs)} games")
    results = run_matches(specs, ctx.configs, ctx.processes)
    games = [r["game"] for r in results]
    encounters = [e for r in results for e in r["encounters"]]
    _tag_kinds(encounters, games)
    write_csv(ctx.out / "E1_games.csv", games)
    write_csv(ctx.out / "E1_encounters.csv", encounters)
    pairings = summarize_pairings(games, ctx, 11)
    h4 = {
        "eq_vs_fm": per_value_payoffs(encounters, "equilibrium", "fixed_mixed", ctx, 12),
        "eq_vs_ph": per_value_payoffs(encounters, "equilibrium", "pure_heuristic", ctx, 13),
        "eu_vs_fm": per_value_payoffs(encounters, "expected_utility", "fixed_mixed", ctx, 14),
        "eu_vs_ph": per_value_payoffs(encounters, "expected_utility", "pure_heuristic", ctx, 15),
    }
    summary = {"seeds": [seeds.start, seeds.stop], "battle_depth": depth, "pairings": pairings, "h4": h4}
    write_json(ctx.out / "E1_summary.json", summary)
    return summary


def h4_verdicts(e1: dict[str, Any]) -> dict[str, Any]:
    def at(rows, v):
        return next(r for r in rows if r["value"] == v)

    h4 = e1["h4"]
    # H4a: CI of (opponent - Equilibrium) above 0  <=>  CI of (Equilibrium - opponent) entirely below 0
    a_checks = {f"{name} V={int(v)}": at(h4[key], v)["diff"]["high"] < 0
                for key, name in (("eq_vs_fm", "vs FixedMixed"), ("eq_vs_ph", "vs PureHeuristic")) for v in (6.0, 8.0)}
    b_ties = {f"vs PureHeuristic V={int(v)}": at(h4["eq_vs_ph"], v)["diff"]["estimate"] == 0 for v in (2.0, 4.0)}
    b_beats = {f"vs FixedMixed V={int(v)}": at(h4["eq_vs_fm"], v)["diff"]["low"] > 0 for v in (2.0, 4.0)}
    c_checks = {f"V={int(v)}": at(h4["eu_vs_fm"], v)["diff"]["low"] > 0 for v in (2.0, 4.0, 6.0, 8.0)}
    status = lambda checks: "supported" if all(checks.values()) else "not supported"  # noqa: E731
    return {
        "H4a": {"status": status(a_checks), "checks": a_checks},
        "H4b": {"status": status({**b_ties, **b_beats}), "checks": {**b_ties, **b_beats}},
        "H4c": {"status": status(c_checks), "checks": c_checks},
        "H4d": {"status": "exploratory", "pairings": {p["pairing"]: p["verdict"] for p in e1["pairings"]}},
    }


# ------------------------------------------------------------------ E2
def run_e2(ctx: Context) -> dict[str, Any]:
    sec = ctx.configs.experiments.section("E2_depth")
    seeds = range(ctx.base_seed + 200000, ctx.base_seed + 200000 + ctx.n(int(sec["seeds_per_pairing"]), 20))
    specs: list[MatchSpec] = []
    for shallow, deep in sec["pairings"]:
        specs += side_swapped_specs("E2", "depth_limited", int(deep), "depth_limited", int(shallow), seeds, f"depth {deep} vs depth {shallow}")
    print(f"E2 depth: {len(specs)} games")
    results = run_matches(specs, ctx.configs, ctx.processes)
    games = [r["game"] for r in results]
    encounters = [e for r in results for e in r["encounters"]]
    _tag_kinds(encounters, games)
    write_csv(ctx.out / "E2_games.csv", games)
    write_csv(ctx.out / "E2_encounters.csv", encounters)
    summary = {"seeds": [seeds.start, seeds.stop], "pairings": summarize_pairings(games, ctx, 21)}
    battles = [e for e in encounters if e["battle_winner"]]
    for p in summary["pairings"]:
        bs = [e for e in battles if e["pairing"] == p["pairing"]]
        deep_w = sum(1 for e in bs if e["battle_winner"] == "first")
        p["battles"] = len(bs)
        p["deep_battle_win_rate"] = interval_dict(wilson_interval(deep_w, len(bs), ctx.confidence)) if bs else None
    write_json(ctx.out / "E2_summary.json", summary)
    return summary


# ------------------------------------------------------------------ battle-level experiments (H2, E4)
def battle_starts(ctx: Context, salt: int, count: int):
    rng = ctx.rng(salt)
    return [random_start(rng, ctx.configs.battle, 0) for _ in range(count)]


def play_paired(starts, depths: tuple[int, int], ctx: Context) -> list[list[int | None]]:
    """Each start played twice (first mover 0, then 1). Returns winners per start."""
    cfg = ctx.configs.battle
    out = []
    for s in starts:
        out.append([play_battle(s._replace(to_move=fm), depths, cfg).winner for fm in (0, 1)])
    return out


def run_h2(ctx: Context) -> dict[str, Any]:
    sec = ctx.configs.experiments.section("H2_battle_depth")
    opp = int(sec["fixed_opponent_depth"])
    starts = battle_starts(ctx, 300000, ctx.n(int(sec["start_positions"]), 60))
    print(f"H2 battle depth: {len(starts)} starts x 2 orders x 3 depths")
    per_depth: dict[int, list[list[int | None]]] = {d: play_paired(starts, (d, opp), ctx) for d in ctx.configs.battle.allowed_depths}
    rows = []
    for i, s in enumerate(starts):
        for d, winners in per_depth.items():
            for fm, w in zip((0, 1), winners[i]):
                rows.append({"start": i, "depth": d, "first_mover": fm, "winner": "tested" if w == 0 else "opponent" if w == 1 else "tie"})
    write_csv(ctx.out / "H2_battles.csv", rows)
    rng = ctx.rng(31)
    summary: dict[str, Any] = {"opponent_depth": opp, "starts": len(starts), "depths": {}, "differences": {}}
    for d, winners in per_depth.items():
        flat = [w for pair in winners for w in pair]
        n = len(flat)
        summary["depths"][d] = {
            "battles": n,
            "win_rate": interval_dict(wilson_interval(sum(1 for w in flat if w == 0), n, ctx.confidence)),
            "tie_rate": interval_dict(wilson_interval(sum(1 for w in flat if w is None), n, ctx.confidence)),
            "loss_rate": interval_dict(wilson_interval(sum(1 for w in flat if w == 1), n, ctx.confidence)),
        }
    depths = sorted(per_depth)
    for lo, hi in combinations(depths, 2):
        clusters = [[(1.0 if a == 0 else 0.0) - (1.0 if b == 0 else 0.0) for a, b in zip(per_depth[hi][i], per_depth[lo][i])] for i in range(len(starts))]
        summary["differences"][f"{hi}-{lo}"] = interval_dict(cluster_bootstrap_mean(clusters, rng, ctx.resamples, ctx.confidence))
    write_json(ctx.out / "H2_summary.json", summary)
    return summary


def h2_verdict(h2: dict[str, Any]) -> dict[str, Any]:
    diffs = h2["differences"]
    refuted = [k for k, iv in diffs.items() if iv["high"] < 0]
    proposal = diffs.get("3-1", {"low": 0})["low"] > 0
    return {
        "status": "refuted" if refuted else "supported",
        "refuting_pairs": refuted,
        "proposal_criterion_met": proposal,
        "note": "Supported means no pair shows a significant DECREASE with depth (pre-registered rule).",
    }


def mix_f(r: Fraction) -> tuple[Fraction, ...]:
    """H3 closed form at V=6, c=2, d=1 (PREDICTIONS.md)."""
    den = 90 - 18 * r
    return (Fraction(39) / den, (24 * r - 3) / den, (54 - 42 * r) / den)


def run_e4(ctx: Context) -> dict[str, Any]:
    sec = ctx.configs.experiments.section("E4_coupling")
    value = Fraction(int(sec["resource_value"]))
    cfg = ctx.configs
    summary: dict[str, Any] = {"value": str(value), "pairs": []}
    battle_rows = []
    for k, (shallow, deep) in enumerate(sec["depth_pairs"]):
        shallow, deep = int(shallow), int(deep)
        starts = battle_starts(ctx, 400000 + k, ctx.n(int(sec["start_positions"]), 60))
        print(f"E4 coupling: depth {deep} vs {shallow}, {len(starts)} starts x 2 orders")
        winners = play_paired(starts, (deep, shallow), ctx)  # side 0 = deep
        flat = [w for pair in winners for w in pair]
        n = len(flat)
        deep_w = sum(1 for w in flat if w == 0)
        shallow_w = sum(1 for w in flat if w == 1)
        for i, pair in enumerate(winners):
            for fm, w in zip((0, 1), pair):
                battle_rows.append({"deep": deep, "shallow": shallow, "start": i, "first_mover": fm, "winner": "deep" if w == 0 else "shallow" if w == 1 else "tie"})
        rng = ctx.rng(40 + k)
        clusters = [[(1.0 if w == 0 else 0.0) - (1.0 if w == 1 else 0.0) for w in pair] for pair in winners]
        h3a = cluster_bootstrap_mean(clusters, rng, ctx.resamples, ctx.confidence)
        p_deep, p_shallow = Fraction(deep_w, n), Fraction(shallow_w, n)
        game = build_encounter_game(value, cfg.payoffs, p_deep, p_shallow)  # row = deep player
        eq_set = solve_game(game)
        check = crosscheck(game)
        eq = select_equilibrium(eq_set.equilibria)
        closed_form_valid = all(x > 0 for x in mix_f(p_shallow)) and all(x > 0 for x in mix_f(p_deep))
        matches_closed_form = (not closed_form_valid) or (eq.row == mix_f(p_shallow) and eq.col == mix_f(p_deep))
        symmetric = select_equilibrium(solve_game(build_encounter_game(value, cfg.payoffs)).equilibria)
        # H3c: realized encounters with equilibrium play and real battles
        sim_rng = ctx.rng(4000 + k)
        deep_pay, shallow_pay = [], []
        for _ in range(ctx.n(int(sec["realized_encounters"]), 300)):
            a_deep = sample_action(eq.row, sim_rng)
            a_shallow = sample_action(eq.col, sim_rng)
            winner = None
            if a_deep == ATTACK and a_shallow == ATTACK:
                start = random_start(sim_rng, cfg.battle, int(sim_rng.integers(2)))
                winner = play_battle(start, (deep, shallow), cfg.battle).winner
            gains, costs = realized_gains_and_costs(a_deep, a_shallow, value, cfg.payoffs, winner)
            deep_pay.append(float(gains[0] - costs[0]))
            shallow_pay.append(float(gains[1] - costs[1]))
        h3c = cluster_bootstrap_mean([[a - b] for a, b in zip(deep_pay, shallow_pay)], ctx.rng(4100 + k), ctx.resamples, ctx.confidence)
        summary["pairs"].append(
            {
                "deep": deep,
                "shallow": shallow,
                "battles": n,
                "p_deep": str(p_deep),
                "p_shallow": str(p_shallow),
                "p_deep_float": float(p_deep),
                "p_shallow_float": float(p_shallow),
                "tie_rate": float(1 - p_deep - p_shallow),
                "p_deep_ci": interval_dict(wilson_interval(deep_w, n, ctx.confidence)),
                "p_shallow_ci": interval_dict(wilson_interval(shallow_w, n, ctx.confidence)),
                "h3a_diff": interval_dict(h3a),
                "deep_mix": [str(x) for x in eq.row],
                "shallow_mix": [str(x) for x in eq.col],
                "deep_mix_float": [float(x) for x in eq.row],
                "shallow_mix_float": [float(x) for x in eq.col],
                "symmetric_mix_float": [float(x) for x in symmetric.row],
                "deep_value": float(eq.row_payoff),
                "shallow_value": float(eq.col_payoff),
                "symmetric_value": float(symmetric.row_payoff),
                "unique_equilibrium": len(eq_set.equilibria) == 1,
                "nashpy_agrees": check.agree,
                "closed_form_valid": closed_form_valid,
                "matches_closed_form": matches_closed_form,
                "deep_attacks_less": eq.row[ATTACK] < eq.col[ATTACK],
                "deep_expands_more": eq.row[2] > eq.col[2],
                "deep_value_higher": eq.row_payoff > eq.col_payoff,
                "realized_encounters": len(deep_pay),
                "realized_deep_mean": statistics.fmean(deep_pay),
                "realized_shallow_mean": statistics.fmean(shallow_pay),
                "h3c_diff": interval_dict(h3c),
                "analysis_card": analyze(game).to_text(),
            }
        )
    write_csv(ctx.out / "E4_battles.csv", battle_rows)
    write_json(ctx.out / "E4_summary.json", summary)
    return summary


def h3_verdicts(e4: dict[str, Any]) -> dict[str, Any]:
    pairs = e4["pairs"]
    a = {f"{p['deep']}v{p['shallow']}": p["h3a_diff"]["low"] > 0 for p in pairs}
    b = {
        f"{p['deep']}v{p['shallow']}": p["deep_value_higher"] and p["deep_attacks_less"] and p["deep_expands_more"] and p["matches_closed_form"] and p["nashpy_agrees"]
        for p in pairs
    }
    c = {f"{p['deep']}v{p['shallow']}": p["h3c_diff"]["low"] > 0 for p in pairs}
    status = lambda checks: "supported" if all(checks.values()) else "not supported"  # noqa: E731
    # EXPLORATORY (not pre-registered): restate H3b with "stronger" = higher MEASURED battle-win
    # probability instead of "deeper". This separates the game-theory claim (indifference
    # principle) from the empirical premise that depth buys battle strength (H3a).
    exploratory = {}
    for p in pairs:
        deep_mix = [Fraction(x) for x in p["deep_mix"]]
        shallow_mix = [Fraction(x) for x in p["shallow_mix"]]
        pd, ps = Fraction(p["p_deep"]), Fraction(p["p_shallow"])
        if pd == ps:
            holds = deep_mix == shallow_mix
        else:
            strong, weak = (deep_mix, shallow_mix) if pd > ps else (shallow_mix, deep_mix)
            holds = strong[0] < weak[0] and strong[2] > weak[2] and strong[1] < weak[1]
        exploratory[f"{p['deep']}v{p['shallow']}"] = {
            "stronger_by_measurement": "deep" if pd > ps else "shallow" if ps > pd else "equal",
            "indifference_prediction_holds": holds,
            "predicted_value_gap": p["deep_value"] - p["shallow_value"],
            "realized_gap_ci": p["h3c_diff"],
        }
    return {
        "H3a": {"status": status(a), "checks": a},
        "H3b": {"status": status(b), "checks": b},
        "H3c": {"status": status(c), "checks": c},
        "exploratory_by_measured_strength": exploratory,
    }


# ------------------------------------------------------------------ E5
def structure_of(game) -> dict[str, Any]:
    a = analyze(game)
    eqs = a.equilibria.equilibria
    if a.strict_elimination.solved:
        kind = "pure (E,E) by iterated strict dominance" if a.strict_elimination.remaining_rows == (2,) else "pure by dominance"
    elif a.equilibria.degenerate:
        kind = "degenerate (continuum)"
    elif len(eqs) == 1 and len(eqs[0].row_support) == 3:
        kind = "cyclic, unique fully mixed"
    else:
        kind = "other"
    sel = a.selected_row
    return {
        "structure": kind,
        "equilibria": len(eqs),
        "selected_A": float(sel.row[0]),
        "selected_D": float(sel.row[1]),
        "selected_E": float(sel.row[2]),
        "value": float(sel.row_payoff),
        "pareto_dominated": bool(a.pareto_dominated_equilibria),
    }


def predicted_structure(v: Fraction, c: Fraction) -> str:
    if v < 2 * c:
        return "pure (E,E) by iterated strict dominance"
    if v == 2 * c:
        return "degenerate (continuum)"
    return "cyclic, unique fully mixed"


def run_e5(ctx: Context) -> dict[str, Any]:
    sec = ctx.configs.experiments.section("E5_sensitivity")
    base = ctx.configs.payoffs
    factors = [Fraction(str(f)) for f in sec["scale_factors"]]
    rows = []
    for v0, kv, kc, kd in product(sec["values"], factors, factors, factors):
        v = Fraction(v0) * kv
        c = base.attack_cost * kc
        d = base.defend_cost * kd
        game = build_encounter_game(v, base.with_overrides(attack_cost=c, defend_cost=d))
        s = structure_of(game)
        pred = predicted_structure(v, c)
        pareto_ok = s["pareto_dominated"] == (v > 2 * c)
        rows.append({"V": float(v), "c": float(c), "d": float(d), "base_V": v0, **s, "predicted": pred, "matches_prediction": s["structure"] == pred and (pareto_ok or v == 2 * c)})
    write_csv(ctx.out / "E5_sensitivity.csv", rows)
    sweep = []
    for k in range(1, 25):
        v = Fraction(k, 2)
        s = structure_of(build_encounter_game(v, base))
        sweep.append({"V": float(v), **s, "predicted": predicted_structure(v, base.attack_cost)})
    write_csv(ctx.out / "E5_value_sweep.csv", sweep)
    distinct = {(r["V"], r["c"], r["d"]) for r in rows}
    summary = {
        "grid_points": len(rows),
        "distinct_games": len(distinct),
        "mismatches": [r for r in rows if not r["matches_prediction"]],
        "degenerate_points": sorted({(r["V"], r["c"], r["d"]) for r in rows if r["structure"].startswith("degenerate")}),
        "structure_counts": {k: sum(1 for r in rows if r["structure"] == k) for k in sorted({r["structure"] for r in rows})},
    }
    write_json(ctx.out / "E5_summary.json", summary)
    return summary


# ------------------------------------------------------------------ H1 from E3
def h1_verdicts(e3_rows: list[dict[str, str]], ctx: Context) -> dict[str, Any]:
    same = all(r["same_value"] == "True" and r["same_move"] == "True" for r in e3_rows)
    d5 = [float(r["node_ratio"]) for r in e3_rows if r["depth"] == "5"]
    ci = cluster_bootstrap_mean([[x] for x in d5], ctx.rng(51), ctx.resamples, ctx.confidence)
    by_depth = {}
    for depth in sorted({r["depth"] for r in e3_rows}, key=int):
        rs = [r for r in e3_rows if r["depth"] == depth]
        by_depth[depth] = {
            "positions": len(rs),
            "mean_minimax_nodes": statistics.fmean(int(r["minimax_nodes"]) for r in rs),
            "mean_alphabeta_nodes": statistics.fmean(int(r["alphabeta_nodes"]) for r in rs),
            "mean_ratio": statistics.fmean(float(r["node_ratio"]) for r in rs),
            "mean_minimax_seconds": statistics.fmean(float(r["minimax_seconds"]) for r in rs),
            "mean_alphabeta_seconds": statistics.fmean(float(r["alphabeta_seconds"]) for r in rs),
            "all_same": all(r["same_value"] == "True" and r["same_move"] == "True" for r in rs),
        }
    return {
        "H1a": {"status": "supported" if same else "refuted", "comparisons": len(e3_rows)},
        "H1b": {"status": "supported" if ci.estimate <= 0.5 else "refuted", "depth5_mean_ratio": interval_dict(ci)},
        "by_depth": by_depth,
    }


def e5_verdict(e5: dict[str, Any]) -> dict[str, Any]:
    return {"status": "supported" if not e5["mismatches"] else "refuted", "mismatches": len(e5["mismatches"]), "grid_points": e5["grid_points"]}


def h2_joint(out: Path, lo: int = 3, hi: int = 5) -> dict[str, int]:
    """Joint outcomes of the SAME start and first mover at two depths (from H2_battles.csv).

    Explains H2's non-monotonicity: it shows which outcomes the deeper search
    gains and loses relative to the shallower one. Labels are win / tie / loss
    for the tested agent; keys are "<depth lo outcome>_<depth hi outcome>".
    """
    label = {"tested": "win", "tie": "tie", "opponent": "loss"}
    by: dict[tuple[str, str, int], str] = {}
    for r in read_csv(out / "H2_battles.csv"):
        by[(r["start"], r["first_mover"], int(r["depth"]))] = label[r["winner"]]
    joint = {f"{a}_{b}": 0 for a in ("win", "tie", "loss") for b in ("win", "tie", "loss")}
    for (start, fm, depth), outcome in by.items():
        if depth == lo:
            joint[f"{outcome}_{by[(start, fm, hi)]}"] += 1
    return joint
