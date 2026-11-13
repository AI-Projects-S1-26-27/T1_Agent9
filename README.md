# Agent9: Strategic Agents in a Territory-Conquest Game


The layers are **coupled**: the Attack-vs-Attack payoff comes from the *measured* battle-win
probability at the two agents' search depths. That lets us measure how a lookahead advantage
changes the equilibrium of the simultaneous game. We did not find a teaching/demo environment that
exposes this coupling for live inspection *(the team is still verifying this statement against the
literature)*.

Repository: https://github.com/AI-Projects-S1-26-27/T1_Agent9. The proposal's link,
`Rohit-Gangil/agent9-resource-war`, redirects here.

## Quickstart

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest                                   # full suite, about 90 s
python -m resource_war.reproduce         # every result, table, figure and the report (about 2 min)
python -m resource_war.demo.web          # the game-style live demo, opens in your browser
```

`make install | test | reproduce | quick | web | demo | cli | edge-cases` wrap the same commands.

## The live demo

`python -m resource_war.demo.web` opens the game at http://127.0.0.1:8765. It has three modes:

- **Watch.** Two AI agents play. Every step (moves, standoffs, battle moves) is a frame you can
  step through forwards and backwards; each frame explains its decision:
  - each side's reasoning and odds;
  - the payoff matrix with the chosen cell;
  - the Nash equilibrium and the Pareto comparison;
  - for battles, the ranked engine moves, an evaluation bar, and how much alpha-beta saved over
    minimax.
- **Play.** You are Blue against any AI. You choose every standoff (Attack / Defend / Expand) and
  every battle move, with optional advice.
  - "Undo my last choice" and "Go back and choose differently here" rewind the game: it is
    deterministic, so it is simply replayed with your earlier choices.
- **Explore.** Sliders for the crystal value, the costs and the battle-win odds; the game theory
  updates live.

It runs offline: Python standard library plus plain HTML/CSS/JS. The game logic is the same engine
as the experiments, and a test checks that a traced game is identical to a plain one. The URL
records the match and the step, so a refresh or a shared link reopens the same moment.

The Streamlit dashboard (`streamlit run src/resource_war/demo/app.py`) and the CLI below remain as
alternatives.

### CLI fallback (works even if the GUI does not)

```bash
python -m resource_war.demo.cli play --agents equilibrium fixed_mixed --depths 3 3 --seed 7
python -m resource_war.demo.cli analyze --value 6        # Game Analysis card for an encounter
python -m resource_war.demo.cli battle --depths 1 5 --seed 3   # minimax vs alpha-beta side by side
python -m resource_war.demo.cli mini --agents expected_utility fixed_mixed --games 20
python -m resource_war.demo.cli results
python -m resource_war.demo.edge_cases                   # every invalid input -> clear message
```

## Results

- Report (generated; do not edit by hand): [report/report.md](report/report.md), built from
  [report/template.md](report/template.md).
- Pre-registered predictions and their outcomes:
  [report/generated/verdicts.md](report/generated/verdicts.md).
- Every number comes from `experiments/results/`, which `python -m resource_war.reproduce`
  regenerates from the seeds and `config/*.yaml`.

## Documentation

| Document | What it holds |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | Design: locked rules, payoffs, derivations |
| [docs/PREDICTIONS.md](docs/PREDICTIONS.md) | Pre-registered predictions (H1–H4), with dated addenda only |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Every decision: why, alternatives, owner; plus contributions |
| [docs/ROLES.md](docs/ROLES.md) | One owner and one reviewer per component |
| [docs/AI_USAGE.md](docs/AI_USAGE.md) | AI usage log, with how each output was verified |
| [docs/viva_notes/](docs/viva_notes/) | Per-owner explanations of their module |
| [docs/demo_video_script.md](docs/demo_video_script.md) | 5–7 minute demo walkthrough |

## Layout

```
config/                  all numbers (payoffs, game, battle, agents, experiments)
src/resource_war/
  engine/                board, shared policy, seeded turn loop, P1 gate
  encounter/             payoff matrix, dominance, pure Nash, Pareto, analysis card
  equilibrium/           exact support/vertex enumeration, verifier, nashpy cross-check
  battle/                Battle Mode rules, evaluation, minimax, alpha-beta, E3 benchmark
  agents/                the five agent types
  tournament/            runner, statistics, experiments, plots, tables
  demo/                  Streamlit app, CLI, robustness cases
  reproduce.py           one command for everything
experiments/benchmark/   frozen E3 positions (never regenerated)
experiments/results/     result CSV/JSON (regenerated)
report/                  template, generated tables and figures, report.md
tests/                   pytest suite
```

## Roles (draft until the team confirms)

| Component | Owner | Reviewer |
|---|---|---|
| Integration, repo, CI, live demo UI | Rohit Gangil | Atharv Singhal |
| Game engine: board, turn loop, rules, seeded RNG | Vaibhav Malik | Harshit Agrawal |
| Payoff-matrix builder, dominance, pure Nash, Pareto | Atharv Singhal | Gursidak Singh Bassi |
| Mixed Nash, nashpy validation, best-response verifier | Gursidak Singh Bassi | Atharv Singhal |
| Battle Mode: state, move generation, evaluation, minimax | Harshit Agrawal | Raunak Sarawgi |
| Alpha-beta, instrumentation, depth benchmark (E3) | Raunak Sarawgi | Harshit Agrawal |
| Agent library (all five agent types) | Tushya Jain | Gursidak Singh Bassi |
| Tournament runner, statistics, plots (E1, E2, E4, E5) | Manas Tripathi | Jainam Parekh Naresh |
| Test suite, edge cases, robustness, input validation | Jainam Parekh Naresh | Vaibhav Malik |
