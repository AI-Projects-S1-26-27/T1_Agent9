"""Streamlit live demo: play matches, explore configurations, inspect battles, show results.

Owner: Rohit Gangil, Reviewer: Atharv Singhal

Run:  streamlit run src/resource_war/demo/app.py

Tabs:
1. Play a match: pick two agents, depths and a seed; step turn by turn or
   jump to the next encounter. Each encounter shows its Game Analysis card,
   and each battle shows minimax and alpha-beta side by side. "Replay this
   seed" rebuilds the identical game.
2. Configuration explorer: sliders for V, c, d and the battle-win
   probabilities. The analysis updates live: Attack dominated below 2c, the
   weak-dominance edge at 2c, the cyclic mixed equilibrium above it. This
   answers the first-round feedback ("how outcomes change under different
   game configurations").
3. Custom matrix: type any bimatrix game; it is validated and analysed.
4. Battle viewer: one Battle Mode fight in detail.
5. Results: the precomputed tournament tables and figures (so the demo
   never waits on a 4,000-game run), plus a live 20-game mini-tournament.

Robustness: every user action goes through ``guard``, which turns
ConfigError / IllegalMoveError / ValueError into an on-screen message.
"""

from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import numpy as np
import streamlit as st

from resource_war.agents.library import AGENT_TYPES, make_agent
from resource_war.config import ConfigError, load_all, parse_payoffs, parse_yaml_text, validate_seed
from resource_war.demo import render
from resource_war.encounter.analysis import analyze
from resource_war.encounter.matrix import build_encounter_game, make_game
from resource_war.engine.game import Game
from resource_war.engine.state import IllegalMoveError

ROOT = Path(__file__).resolve().parents[3]
PLAYER_COLORS = ("#2a78d6", "#eb6834")
OWNED_TINT = ("#dce9f9", "#fbe1d6")


@contextmanager
def guard():
    try:
        yield
    except (ConfigError, IllegalMoveError, ValueError, FileNotFoundError) as exc:
        st.error(f"{type(exc).__name__}: {exc}")


@st.cache_resource
def configs():
    return load_all()


def board_html(state, bases) -> str:
    cells = render.board_cells(state, bases)
    rows = []
    for row in cells:
        tds = []
        for cell in row:
            bg = OWNED_TINT[cell.owner] if cell.owner is not None else "#fcfcfb"
            border = "3px solid #0b0b0b" if cell.contested else "1px solid #d8d7d2"
            parts = []
            if cell.base is not None:
                parts.append(f"<b>B{cell.base}</b>")
            if cell.resource is not None:
                parts.append(f"<span style='color:#52514e'>${cell.resource}</span>")
            for p in (0, 1):
                if cell.units[p]:
                    parts.append(f"<span style='color:{PLAYER_COLORS[p]};font-weight:700'>{'&#9679;' * cell.units[p]}</span>")
            tds.append(f"<td style='width:56px;height:44px;text-align:center;background:{bg};border:{border};font-size:13px'>{'<br>'.join(parts)}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    legend = (
        f"<div style='font-size:12px;color:#52514e;margin-top:6px'><span style='color:{PLAYER_COLORS[0]}'>&#9679;</span> player 0 unit &nbsp; "
        f"<span style='color:{PLAYER_COLORS[1]}'>&#9679;</span> player 1 unit &nbsp; $V resource &nbsp; tinted = owned territory &nbsp; heavy border = contested</div>"
    )
    return "<table style='border-collapse:collapse'>" + "".join(rows) + "</table>" + legend


def tab_play(cfg) -> None:
    c1, c2, c3, c4 = st.columns(4)
    a0 = c1.selectbox("Player 0 agent", AGENT_TYPES, index=2, format_func=lambda k: render.AGENT_LABELS[k])
    a1 = c2.selectbox("Player 1 agent", AGENT_TYPES, index=1, format_func=lambda k: render.AGENT_LABELS[k])
    d0 = c3.selectbox("Player 0 battle depth", cfg.battle.allowed_depths, index=1)
    d1 = c3.selectbox("Player 1 battle depth", cfg.battle.allowed_depths, index=1)
    seed = c4.number_input("Seed", min_value=0, max_value=2**32 - 1, value=7, step=1)
    upload = st.file_uploader("Optional: load a payoff config (payoffs.yaml format)", type=["yaml", "yml"])
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("New game / Replay this seed", type="primary"):
        with guard():
            run_cfg = cfg
            if upload is not None:
                payoffs = parse_payoffs(parse_yaml_text(upload.getvalue().decode("utf-8", errors="replace"), upload.name), upload.name)
                run_cfg = cfg.__class__(payoffs, cfg.game, cfg.battle, cfg.agents, cfg.experiments)
            agents = (make_agent(a0, run_cfg.agents, d0), make_agent(a1, run_cfg.agents, d1))
            st.session_state.game = Game(run_cfg, agents, validate_seed(int(seed)), compare_minimax=True)
            st.session_state.events = []
    game: Game | None = st.session_state.get("game")
    if game is None:
        st.info("Choose agents, depths and a seed, then press 'New game'.")
        return
    with guard():
        if b2.button("Next turn", disabled=game.finished):
            st.session_state.events = game.step()
        if b3.button("Next encounter", disabled=game.finished):
            events: list = []
            while not game.finished and not any(e["event"] == "encounter" for e in events):
                events = game.step()
            st.session_state.events = events
        if b4.button("Play to the end", disabled=game.finished):
            while not game.finished:
                game.step()
            st.session_state.events = []
    s0, s1 = game.scores()
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Turn {game.state.turn} / {game.cfg.turn_limit}**: score player 0 **{float(s0):.1f}**, player 1 **{float(s1):.1f}**")
        st.markdown(board_html(game.state, game.cfg.bases), unsafe_allow_html=True)
        if game.finished:
            r = game.result()
            st.success(f"Game over. Winner: {'tie' if r.winner is None else 'player ' + str(r.winner)}. Resources {[float(x) for x in r.gross_collected]}, territory {list(r.territory)}, encounters {len(r.encounters)}.")
    with right:
        st.markdown("**This turn**")
        for e in st.session_state.get("events", []):
            if e["event"] == "encounter":
                st.markdown(f"Encounter at {tuple(e['cell'])}, V = {e['value']}: **{e['actions'][0]}** vs **{e['actions'][1]}** -> {e['payoffs'][0]}, {e['payoffs'][1]}")
            elif e["event"] in ("collect", "regrow", "expand"):
                st.caption(f"{e['event']}: {e.get('cell')} {e.get('value', '')}")
    st.markdown(f"### Encounters so far ({len(game.encounters)})")
    for k, rec in enumerate(reversed(game.encounters[-12:])):
        title = f"Turn {rec.turn}, cell {rec.cell}, V = {rec.value}: {'ADE'[rec.actions[0]]} vs {'ADE'[rec.actions[1]]}, payoffs ({rec.payoffs[0]}, {rec.payoffs[1]})"
        with st.expander(title, expanded=(k == 0)):
            g0 = build_encounter_game(rec.value, game.payoffs, *game.battle_probs(*game.depths))
            st.code(analyze(g0).to_text(), language=None)
            if rec.battle:
                st.markdown(f"**{render.battle_outcome_text(rec.battle)}**. Minimax and alpha-beta searched every decision; they always chose the same move.")
                st.dataframe(render.battle_rows(rec.battle), hide_index=True)


def tab_explorer(cfg) -> None:
    st.markdown("Move the sliders: the analysis is recomputed exactly (fractions) from these numbers.")
    c1, c2, c3 = st.columns(3)
    v = c1.slider("Cell value V", 0.5, 12.0, 6.0, 0.5)
    c = c2.slider("Attack cost c", 0.0, 5.0, float(cfg.payoffs.attack_cost), 0.25)
    d = c3.slider("Defend cost d", 0.0, 3.0, float(cfg.payoffs.defend_cost), 0.25)
    p1, p2 = st.columns(2)
    pr = p1.slider("Row player's battle-win probability", 0.0, 1.0, 0.5, 0.05)
    pc = p2.slider("Column player's battle-win probability", 0.0, 1.0, 0.5, 0.05)
    with guard():
        payoffs = cfg.payoffs.with_overrides(attack_cost=Fraction(str(c)), defend_cost=Fraction(str(d)), p_row=Fraction(str(pr)), p_col=Fraction(str(pc)))
        game = build_encounter_game(Fraction(str(v)), payoffs)
        analysis = analyze(game)
        two_c = 2 * payoffs.attack_cost
        regime = "below 2c: Attack is dominated" if Fraction(str(v)) < two_c else "exactly 2c: the degenerate edge" if Fraction(str(v)) == two_c else "above 2c: cyclic structure"
        st.info(f"V = {v} is {regime} (2c = {float(two_c):g}). {render.structure_banner(analysis)}")
        st.dataframe(
            {lab: [f"({game.row_payoffs[i][j]}, {game.col_payoffs[i][j]})" for i in range(3)] for j, lab in enumerate(game.col_labels)},
            hide_index=False,
        )
        st.code(analysis.to_text(), language=None)


def tab_custom() -> None:
    st.markdown("Enter any two-player game. Rows separated by `;`, entries by `,`. Example: Prisoner's Dilemma.")
    row = st.text_input("Row player's payoffs", "3, 0; 5, 1")
    col = st.text_input("Column player's payoffs", "3, 5; 0, 1")
    labels = st.text_input("Action labels (optional, comma-separated, rows then columns separated by |)", "C, D | C, D")
    if st.button("Analyse custom game"):
        with guard():
            parse = lambda text: [[x.strip() for x in r.split(",")] for r in text.split(";") if r.strip()]  # noqa: E731
            rl, cl = (None, None)
            if "|" in labels:
                rl = [x.strip() for x in labels.split("|")[0].split(",") if x.strip()]
                cl = [x.strip() for x in labels.split("|")[1].split(",") if x.strip()]
            game = make_game(parse(row), parse(col), rl or None, cl or None, "custom game")
            analysis = analyze(game)
            st.info(render.structure_banner(analysis))
            st.code(analysis.to_text(), language=None)


def tab_battle(cfg) -> None:
    from resource_war.battle.play import resolve_battle

    c1, c2, c3 = st.columns(3)
    d0 = c1.selectbox("Side 0 depth", cfg.battle.allowed_depths, index=0, key="bd0")
    d1 = c2.selectbox("Side 1 depth", cfg.battle.allowed_depths, index=2, key="bd1")
    seed = c3.number_input("Battle seed", min_value=0, max_value=2**32 - 1, value=3, step=1)
    if st.button("Fight"):
        with guard():
            _, summary = resolve_battle(np.random.default_rng(validate_seed(int(seed))), cfg.battle, (d0, d1), compare_minimax=True)
            st.markdown(f"Start: `{summary['start']}`")
            st.markdown(f"**{render.battle_outcome_text(summary)}**")
            total_mm, total_ab = summary.get("minimax_nodes", 0), summary["alphabeta_nodes"]
            st.metric("Nodes saved by alpha-beta in this battle", f"{100 * (1 - total_ab / total_mm):.0f}%" if total_mm else "-", help=f"{total_mm:,} minimax vs {total_ab:,} alpha-beta nodes")
            st.dataframe(render.battle_rows(summary), hide_index=True)


def tab_results(cfg) -> None:
    gen, figs = ROOT / "report" / "generated", ROOT / "report" / "figures"
    if (gen / "verdicts.md").exists():
        st.markdown("#### Pre-registered predictions: outcomes")
        st.markdown((gen / "verdicts.md").read_text(encoding="utf-8"))
        for fig, caption in (
            ("fig_e1_score_diff.png", "E1 round robin"),
            ("fig_h4_per_value.png", "H4 per-encounter payoffs by V"),
            ("fig_h2_battle_depth.png", "H2 battle depth"),
            ("fig_e3_nodes.png", "E3 alpha-beta vs minimax"),
            ("fig_e4_coupling.png", "E4 coupling"),
            ("fig_e5_value_sweep.png", "E5 value sweep"),
        ):
            if (figs / fig).exists():
                st.image(str(figs / fig), caption=caption)
    else:
        st.warning("No precomputed results yet. Run `python -m resource_war.reproduce`.")
    st.markdown("#### Live mini-tournament")
    c1, c2, c3 = st.columns(3)
    a = c1.selectbox("Agent A", AGENT_TYPES, index=3, key="ma")
    b = c2.selectbox("Agent B", AGENT_TYPES, index=1, key="mb")
    games = c3.slider("Games (each seed played with sides swapped)", 2, 40, 20, 2)
    if st.button("Run mini-tournament"):
        with guard():
            from resource_war.tournament.runner import run_matches, side_swapped_specs
            from resource_war.tournament.stats import wilson_interval

            specs = side_swapped_specs("mini", a, 3, b, 3, range(1, 1 + games // 2))
            rows = [r["game"] for r in run_matches(specs, cfg, processes=1, progress=None)]
            n = len(rows)
            st.table(
                [
                    {"agent": a, "wins": sum(r["first_win"] for r in rows), "win rate [95% CI]": wilson_interval(sum(r["first_win"] for r in rows), n).fmt(), "mean score": round(sum(r["first_score"] for r in rows) / n, 2)},
                    {"agent": b, "wins": sum(r["second_win"] for r in rows), "win rate [95% CI]": wilson_interval(sum(r["second_win"] for r in rows), n).fmt(), "mean score": round(sum(r["second_score"] for r in rows) / n, 2)},
                    {"agent": "ties", "wins": sum(r["tie"] for r in rows), "win rate [95% CI]": wilson_interval(sum(r["tie"] for r in rows), n).fmt(), "mean score": ""},
                ]
            )


def main() -> None:
    st.set_page_config(page_title="Agent9 Resource War", layout="wide")
    st.title("Agent9: strategic agents in a territory-conquest game")
    st.caption("Simultaneous encounter game (dominance, Nash, Pareto) coupled with sequential Battle Mode (minimax, alpha-beta).")
    try:
        cfg = configs()
    except ConfigError as exc:
        st.error(f"Configuration problem: {exc}")
        return
    tabs = st.tabs(["Play a match", "Configuration explorer", "Custom matrix", "Battle viewer", "Results"])
    with tabs[0]:
        tab_play(cfg)
    with tabs[1]:
        tab_explorer(cfg)
    with tabs[2]:
        tab_custom()
    with tabs[3]:
        tab_battle(cfg)
    with tabs[4]:
        tab_results(cfg)


main()
