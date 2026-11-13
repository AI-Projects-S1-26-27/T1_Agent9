# Roles: one owner, one reviewer per component

This table answers proposal feedback point 4 ("who is accountable for what is not clean"):

- Each member **owns exactly one** primary component.
- Every component has **exactly one** named reviewer.
- Every module docstring carries an `Owner: X, Reviewer: Y` line, and `tests/test_skeleton.py` fails
  the build if one is missing.

| # | Component | Package / files | Owner | Reviewer |
|---|---|---|---|---|
| 1 | Integration, repo, CI, live demo UI | `demo/`, `.github/`, `Makefile`, README | Rohit Gangil | Atharv Singhal |
| 2 | Game engine: board, turn loop, rules, seeded RNG | `engine/` | Vaibhav Malik | Harshit Agrawal |
| 3 | Payoff-matrix builder, dominance, pure Nash, Pareto | `encounter/` | Atharv Singhal | Gursidak Singh Bassi |
| 4 | Mixed Nash (support enumeration), nashpy validation, best-response verifier | `equilibrium/` | Gursidak Singh Bassi | Atharv Singhal |
| 5 | Battle Mode: state, move generation, evaluation, minimax | `battle/` (except the modules in row 6) | Harshit Agrawal | Raunak Sarawgi |
| 6 | Alpha-beta, instrumentation, depth benchmark (E3) | `battle/alphabeta.py`, `battle/instrument.py`, `battle/benchmark.py` | Raunak Sarawgi | Harshit Agrawal |
| 7 | Agent library (all five agent types) | `agents/` | Tushya Jain | Gursidak Singh Bassi |
| 8 | Tournament runner, statistics, plots (E1, E2, E4, E5) | `tournament/` | Manas Tripathi | Jainam Parekh Naresh |
| 9 | Test suite, edge cases, robustness, input validation | `tests/`, config validation | Jainam Parekh Naresh | Vaibhav Malik |

## What "owner" and "reviewer" mean

- **Owner:** understands every line of the component, commits it from their own account, and is
  accountable for it in the final evaluation. Code produced with AI assistance is reviewed and tested
  by its owner before it is committed, and recorded in [AI_USAGE.md](AI_USAGE.md).
- **Reviewer:** reads the owner's change before it is merged and checks it against
  [DESIGN.md](DESIGN.md). The reviewer must be able to explain the component's interface and its
  key decisions.

## Changes from the proposal's role table

| Proposal | Now |
|---|---|
| Game engine: Vaibhav Malik, Manas Tripathi | Vaibhav (owner); Manas moves to the tournament runner |
| Battle Mode tree and minimax: Harshit Agrawal, Vaibhav Malik | Harshit (owner) |
| Agent library *and evaluation function*: Tushya Jain, Jainam Parekh Naresh | Tushya owns the agent library; **the evaluation function moves to Battle Mode (Harshit)** |
| Tournament, statistics, plots, report editing, unit tests: Manas, Jainam | Manas owns the tournament runner; Jainam owns the test suite; report editing is split by section |
