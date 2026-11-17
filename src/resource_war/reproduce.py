"""Regenerate every result, table and figure in the report with one command.

Owner: Manas Tripathi, Reviewer: Jainam Parekh Naresh

    python -m resource_war.reproduce            # full run (as in the report)
    python -m resource_war.reproduce --quick    # small smoke run, separate output folder

Steps, in order:
  1. P1 gate: encounter frequency for every map/policy variant
  2. E3: alpha-beta vs minimax on the FROZEN benchmark set, plus the depth-5 gate
  3. E1: round robin of the four encounter-policy agents (and the H4 encounter log)
  4. E2: DepthLimited full games
  5. H2: Battle Mode win rate by depth against a fixed depth-1 opponent
  6. E4: coupling experiment (H3)
  7. E5: payoff sensitivity sweep and value sweep
  8. verdicts.json: every pre-registered rule from docs/PREDICTIONS.md, evaluated by code
  9. markdown tables (report/generated) and figures (report/figures)
 10. the report itself (report/report.md) from report/template.md

Quick mode writes ONLY to experiments/results_quick/, so a smoke test can
never overwrite the real results the report is built from.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from resource_war.battle.benchmark import depth5_gate, load_positions, run_benchmark
from resource_war.config import load_all
from resource_war.engine.frequency import variant_rows, write_rows
from resource_war.tournament import experiments as ex
from resource_war.tournament import plots, tables

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate all experiment results, tables and figures.")
    parser.add_argument("--quick", action="store_true", help="small smoke run into experiments/results_quick/")
    parser.add_argument("--processes", type=int, default=None, help="worker processes for tournaments (default: CPUs - 1)")
    parser.add_argument("--only-report", action="store_true", help="skip experiments; rebuild verdicts, tables, figures and report")
    args = parser.parse_args(argv)

    configs = load_all()
    out = ROOT / "experiments" / ("results_quick" if args.quick else "results")
    gen = out / "generated" if args.quick else ROOT / "report" / "generated"
    figs = out / "figures" if args.quick else ROOT / "report" / "figures"
    ctx = ex.Context(configs, out, args.processes, args.quick)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if not args.only_report:
        print("[1/7] P1 gate: encounter frequency per variant")
        freq = variant_rows(configs, 20 if args.quick else 200, verbose=False)
        write_rows(freq, out / "P1_encounter_frequency.csv")
        g = configs.game
        chosen = next(r for r in freq if (r["placement"], r["targeting"], r["tie_break"], r["regrow_turns"]) == (g.placement, g.targeting, g.tie_break, g.regrow_turns))
        no_regrow = [r for r in freq if r["regrow_turns"] is None]
        ex.write_json(
            out / "P1_summary.json",
            {
                "chosen": chosen,
                "no_regrow_min_mean": min(r["mean_encounters"] for r in no_regrow),
                "no_regrow_max_mean": max(r["mean_encounters"] for r in no_regrow),
                "no_regrow_worst_share_without_encounter": 1 - min(r["share_games_with_encounter"] for r in no_regrow),
            },
        )

        print("[2/7] E3: alpha-beta vs minimax on the frozen benchmark set")
        e3 = configs.experiments.section("E3_alphabeta_benchmark")
        positions = load_positions()[: 40 if args.quick else None]
        rows = run_benchmark(positions, list(e3["depths"]), configs.battle)
        ex.write_csv(out / "E3_alphabeta_vs_minimax.csv", rows)
        ex.write_json(out / "E3_depth5_gate.json", depth5_gate(rows, configs.battle))

        print("[3/7] E1: round robin")
        ex.run_e1(ctx)
        print("[4/7] E2: depth, full games")
        ex.run_e2(ctx)
        print("[5/7] H2: battle depth")
        ex.run_h2(ctx)
        print("[6/7] E4: coupling")
        ex.run_e4(ctx)
        print("[7/7] E5: sensitivity")
        ex.run_e5(ctx)
        ex.write_json(out / "metadata.json", ex.metadata(ctx))

    e3_rows = ex.read_csv(out / "E3_alphabeta_vs_minimax.csv")
    verdicts = {
        "H1": ex.h1_verdicts(e3_rows, ctx),
        "gate": json.loads((out / "E3_depth5_gate.json").read_text(encoding="utf-8")),
        "H2": {**ex.h2_verdict(json.loads((out / "H2_summary.json").read_text(encoding="utf-8"))), "joint_3_vs_5": ex.h2_joint(out)},
        "H3": ex.h3_verdicts(json.loads((out / "E4_summary.json").read_text(encoding="utf-8"))),
        "H4": ex.h4_verdicts(json.loads((out / "E1_summary.json").read_text(encoding="utf-8"))),
        "E5": ex.e5_verdict(json.loads((out / "E5_summary.json").read_text(encoding="utf-8"))),
    }
    ex.write_json(out / "verdicts.json", verdicts)
    tables.write_all(out, gen)
    plots.make_all(out, figs)
    if not args.quick and (ROOT / "report" / "template.md").exists():
        from resource_war.report_builder import build_report

        build_report(out, gen)

    print(f"\nDone in {time.perf_counter() - t0:.0f} s. Results: {out}")
    print(f"Tables: {gen}\nFigures: {figs}")
    print("Verdicts:")
    print(f"  H1a {verdicts['H1']['H1a']['status']}, H1b {verdicts['H1']['H1b']['status']}, depth-5 gate {'passed' if verdicts['gate']['passed'] else 'FAILED'}")
    print(f"  H2 {verdicts['H2']['status']} (proposal criterion met: {verdicts['H2']['proposal_criterion_met']})")
    print(f"  H3a {verdicts['H3']['H3a']['status']}, H3b {verdicts['H3']['H3b']['status']}, H3c {verdicts['H3']['H3c']['status']}")
    print(f"  H4a {verdicts['H4']['H4a']['status']}, H4b {verdicts['H4']['H4b']['status']}, H4c {verdicts['H4']['H4c']['status']}")
    print(f"  E5 {verdicts['E5']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
