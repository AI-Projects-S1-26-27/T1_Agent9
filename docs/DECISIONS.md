# Decision log and contributions

Source material for the final report: Method, "Changes since proposal", Individual Contributions and
the AI Usage section. It also helps each member prepare for the viva.

- **Append-only.** Every phase adds rows. A decision that is later reversed gets a *new* row that
  references the old one, so the reasoning trail survives.
- Each row records **what** was decided, **why**, **which alternatives were rejected**, **who owns
  it**, and the **source**.
- Web pages used only as inspiration are listed with their URL and marked "inspiration, not cited".
  They go into the report's references only after a teammate has read and checked them.

Source codes:

| Code | Meaning |
|---|---|
| P | approved proposal |
| F | instructor feedback |
| T | team build instructions |
| D | derived or analysed in DESIGN.md |
| A | agreed default (Rohit, 2026-11-13) |
| W | web inspiration |

---

## Decisions

### P0 — 2026-11-13

| ID | Decision | Why | Alternatives rejected | Owner | Source |
|---|---|---|---|---|---|
| D01 | Use the org repo `AI-Projects-S1-26-27/T1_Agent9` | The proposal's link redirects there; it had zero commits | Creating a new repo would lose the proposal link | Rohit | — |
| D02 | Encounter action set A/D/E (3×3) | Readable analysis; gives the rock-paper-scissors structure for V > 2c | 4×4 (the proposal allowed it), but the matrices would be hard to read live | Atharv | P, T |
| D03 | Payoffs locked: c = 2, d = 1, V ∈ {2,4,6,8}, plus the D-vs-D, D-vs-E and E-vs-A cells | Feedback said deferred numbers cost a mark. A non-zero d stops Defend from being free | Leaving the numbers to tuning (penalised); d = 0 (Defend never costly, so the structure is less interesting) | Atharv | F, T |
| D04 | A-vs-A uses separate p_row and p_col | Battles can tie at the ply cap, so (1 − p_row) overstates the column's payoff. H3 depends on the column's own probability | (1 − p) form: wrong whenever ties occur | Gursidak / Harshit | A (1) |
| D05 | Contested cells end **unowned**; the score change of an encounter = its matrix entry | Keeps the game-theory analysis predictive of tournament scores | Giving the cell to the winner, which adds the territory weight on top and breaks the equivalence | Vaibhav | D §3.3 |
| D06 | Shared non-encounter policy (greedy nearest resource), player-relative mirror tie-breaking | Experimental control: agents differ only in encounter policy and depth. Mirroring removes any tie-break advantage | Per-agent movement (would confound the comparison); absolute row-major tie-breaks (would favour one side) | Vaibhav | T, D §3.2 |
| D07 | 12 resources as 6 point-symmetric pairs, one value per pair | Fairness between the two bases | Independent random placement (unfair maps) | Vaibhav | T |
| D08 | P1 gate: measure encounters per game before building agents | Rare encounters would make every agent tie, so no "measurable difference" (first-round feedback) | Discovering it only in P6 | Vaibhav / Manas | A (9), F |
| D09 | Battle on a two-rank (Front/Back) mini-board, with ADVANCE added | Keeps branching ≤ 8 so depth 5 is feasible; ADVANCE lets Back units rejoin the fight | Larger grid mini-board (depth 5 infeasible) | Harshit | P, T |
| D10 | Withdrawn units survive but don't count; a side with no units in the battle loses | Otherwise a side could withdraw at full HP and "win" | Counting withdrawn HP | Harshit | A (3) |
| D11 | First mover drawn by coin in the game; both orders played when estimating p | p(d, d) must be symmetric | A fixed first mover (biased) | Harshit / Manas | A (4) |
| D12 | Terminal utility ±1000 + HP difference; evaluation function only at the depth cutoff | The Front term could otherwise score an exact HP tie as a win; 1000 > max\|eval\| = 64 | One function for both (inconsistent with the win rule) | Harshit | A (5) |
| D13 | Fixed move ordering + root keeps the first maximum (strict >) | Then alpha-beta returns the same *move* as minimax, not just the same value (proof in DESIGN §4.6) | Random tie-breaks (would break the equivalence test) | Raunak / Harshit | T |
| D14 | Simultaneous death treated as unreachable and asserted by a property test | One action per ply deals at most 1 damage | Special-case code for an impossible event | Jainam | A (2) |
| D15 | Agent interface = `choose()` + `observe()` | The proposal's agent loop includes "update"; fictitious play needs it | `choose()` only (the EU agent couldn't learn) | Tushya | P (wins over T) |
| D16 | PureHeuristic attacks iff V > T (T = 5) | The proposal says "above a threshold"; this matters at V = 5, which the sweep produces | V ≥ T | Tushya | P |
| D17 | EU agent: Dirichlet(1,1,1) counts per V; lexicographic tie-break; model reset each game | PureHeuristic's play depends on V; a reset keeps games independent for bootstrap CIs | Pooled counts (mis-models the opponent); persistent model (games not independent) | Tushya | A (8) |
| D18 | Equilibrium selection: maximise own payoff, then lexicographic order | Deterministic and documented; picks (E,E) at V = 4 | Random selection (not reproducible) | Gursidak / Tushya | T |
| D19 | nashpy cross-check uses `vertex_enumeration` on degenerate games | At V = 4, `support_enumeration` misses 3 of the 4 extreme equilibria | `support_enumeration` only (false agreement) | Gursidak | D §2.3 |
| D20 | E1/E2 use p = ½; E4 uses the measured p | Isolates the effect of depth on battles; the coupling is studied separately | Measured p everywhere (assumes agents know each other's depth) | Manas | A (7) |
| D21 | H2 measured at battle level against a fixed depth-1 opponent; E2 is the full-game head-to-head | "Against a fixed opponent" and "pairings" measure different things | Using only E2 for H2 | Manas | A (6) |
| D22 | H4 stated per encounter and per V; E1 logs every encounter | Exact payoffs point in opposite directions at low and high V | A single game-level prediction (direction unknowable) | Manas | D |
| D23 | PREDICTIONS.md committed alone, before any experiment code; later changes only as addenda | Proves predictions came before results | Editing predictions after the fact | Whole team | T |
| D24 | Pin numpy 2.4.6 / scipy 1.17.1 | Newest versions that still support Python 3.11; identical RNG streams on every laptop | numpy 2.5 / scipy 1.18 (require Python 3.12+) | Rohit | — |
| D25 | CI on Ubuntu + Windows × Python 3.11 / 3.13 | The team uses Windows laptops; 3.11 is the floor | Ubuntu only | Rohit | — |
| D26 | Owner/reviewer line in every module, enforced by a test | Feedback point 4 (accountability) | A table in the README only | Jainam | F, T |
| D27 | One owner + one reviewer per component; the evaluation function moves to Battle Mode | Feedback point 4 | The proposal's shared-ownership table | Whole team | F, T |

### P1 — 2026-11-13 (engine, config validation)

| ID | Decision | Why | Alternatives rejected | Owner | Source |
|---|---|---|---|---|---|
| D28 | **Used-up resource cells regrow after 8 turns** | P1 gate (`experiments/results/P1_encounter_frequency.csv`, 200 seeds, both players always Expand). Without regrowth: 1.29–2.73 encounters per game, and 14–51% of games have none, so agents would mostly tie. With regrowth after 8 turns: 14–25 per game | No regrowth (fails the gate); 5 turns (22.9 per game but 3% of games with none); 12 turns (13.5 per game) | Vaibhav | D, measured |
| D29 | `band` placement (resource pairs near the anti-diagonal) + `value_per_distance` targeting | 19.15 encounters per game, every game has at least one, all four V occur (shares 19/23/24/34% for V = 2/4/6/8), first encounter around turn 12 | `point_symmetric` + `nearest`: 14.0 per game, 4% of games with none | Vaibhav | measured |
| D30 | Keep player-relative (mirror) tie-breaking | Exact mirror fairness, proven by a test that positions stay mirror images all game | `absolute` tie-break: slightly more encounters (21.6) but treats the two sides differently | Vaibhav | D06 |
| D31 | A player's units claim distinct targets | Otherwise all 3 units leave the base in lockstep and act as one | Independent targeting | Vaibhav | — |
| D32 | Encounters never change ownership; collecting a regrown cell transfers it to the collector | Keeps the payoff-to-score invariant, which the engine checks with an assertion at runtime | Awarding the cell to the encounter winner | Vaibhav | D05 |
| D33 | If both players Expand onto the same cell in the same turn, neither claims it | A simple, symmetric rule for a rare event | Coin flip (extra randomness for nothing) | Vaibhav | — |
| D34 | A mixed strategy is sampled with one `rng.random()` against cumulative probabilities | Keeps the random stream simple and stable for byte-identical replays | `rng.choice` (internals can change between numpy versions) | Tushya | — |
| D35 | Config: reject booleans, NaN and inf; decimals become exact Fractions via `str` | `True == 1` in Python; 0.35 must mean exactly 7/20 | Plain float parsing | Jainam | T |
| D36 | Seeds limited to [0, 2^32 − 1] | Easy to type, validate and replay in the demo | Unbounded integers | Jainam | — |

### P2 — 2026-11-13 (encounter analysis)

| ID | Decision | Why | Alternatives rejected | Owner | Source |
|---|---|---|---|---|---|
| D37 | Iterated weak elimination uses one documented order (row first, then column, all dominated strategies of a player per round), always printed with an "order-dependent" caveat | Weak elimination can delete equilibria (at V = 4, A is played in a continuum of equilibria) | Hiding weak elimination; claiming a unique result | Atharv | D |
| D38 | Dominance by a *mixture* of strategies is checked with an LP and reported separately; pure dominance is exact | Mixed dominance is a stronger, standard notion; the LP needs floats, so it is kept apart from the exact checks | Pure dominance only | Atharv | — |
| D39 | Pareto analysis compares the pure outcomes (matrix cells) | That is what the card shows, and it answers "is the equilibrium Pareto-dominated?" | Comparing over all mixed outcomes | Atharv | P |

### P3 — 2026-11-13 (mixed equilibria)

| ID | Decision | Why | Alternatives rejected | Owner | Source |
|---|---|---|---|---|---|
| D40 | Degenerate games: exact vertex enumeration of best-response polytopes, plus an **exact degeneracy test** (some polytope vertex has more labels than its dimension) | Equal-size support enumeration misses equilibria in degenerate games. Hypothesis found a 2×3 game whose extra equilibrium has supports of sizes 2 and 1, which our first degeneracy signals missed | Support enumeration only (misses equilibria); unequal-support enumeration (still misses continua) | Gursidak | measured (property test) |
| D41 | The nashpy cross-check compares against `vertex_enumeration` (primary); `support_enumeration` is reported as secondary | nashpy's support enumeration drops true equilibria due to float rounding: default tol = 1e-16, and a hard-coded `prob >= 0` check rejects −5.55e-17. Both cases are reproduced in tests | Trusting nashpy's support enumeration | Gursidak | measured |
| D42 | Selection: maximise own payoff, then the lexicographically smallest (own, opponent) strategies; the card warns on mis-coordination | Deterministic. Shows the Battle of the Sexes failure live, while our encounter games coordinate on (E,E) | Random selection | Gursidak | T, D18 |
| D43 | `solve_game` and `analyze` are cached per (immutable) game | Agents query the same few matrices thousands of times | Recomputing every time | Gursidak | — |

### P4 — 2026-11-13 (Battle Mode, minimax, alpha-beta, E3)

| ID | Decision | Why | Alternatives rejected | Owner | Source |
|---|---|---|---|---|---|
| D44 | Battle states are immutable NamedTuples; moves build new states | A search can never corrupt a sibling state; states are hashable and comparable in tests | Mutable state with undo (faster, but riskier to explain and verify) | Harshit | — |
| D45 | Fail-soft alpha-beta; the root updates only on strict `>` and passes alpha down | Gives minimax's exact move (proof in `alphabeta.py`); checked on 2,500 test pairs and 2,500 frozen comparisons | Fail-hard (same move, less informative bounds) | Raunak | T |
| D46 | Counting: nodes = every recursive call (root included); leaves = scored nodes; pruned = skipped children | The same convention for both searches, so H1 compares like with like | Counting leaves only | Raunak | D §4.6 |
| D47 | Frozen benchmark: 500 mid-battle positions (random start + 0–5 random moves); `generate` refuses to overwrite | Results always refer to the same positions; the set covers wounded, defended and withdrawn units | Fresh random positions on each run | Raunak | T |
| D48 | **Depth-5 gate passed without shrinking the battle.** Measured in the first full run on 2026-11-13: mean 0.033 s, max 0.149 s per decision on 500 positions, against a 2 s gate. Wall times vary by machine; the current values are in `report/generated/e3_gate.md` | Feedback point 3 (feasibility asserted, not estimated) | — | Raunak | measured |
| D49 | The battle summary in the game log holds only deterministic fields; wall times sit under "timing", which the engine strips | Keeps replays byte-identical | Logging times (breaks the replay test) | Harshit / Vaibhav | — |

### P5 — 2026-11-13 (agents)

| ID | Decision | Why | Alternatives rejected | Owner | Source |
|---|---|---|---|---|---|
| D50 | A fresh agent for every game, built by `make_agent` with validated depth and type | Games are independent; bad input gives a clear ConfigError | Reusing agent objects across games | Tushya | D17 |
| D51 | PREDICTIONS Addendum 1: the Dirichlet prior keeps P(D) > 0, so ExpectedUtility plays E (not A) against an Expander at V = 4 | Found by an agent test before any experiment. Recorded as a dated addendum, not an edit | Silently editing the pre-registered table | Tushya | measured |

### P6 — 2026-11-13 (tournaments, statistics, experiments)

| ID | Decision | Why | Alternatives rejected | Owner | Source |
|---|---|---|---|---|---|
| D52 | Bootstrap resamples **clusters**: a seed's two side-swapped games, a battle start's two first-mover orders, or one game's encounters | Observations in a cluster share a map or start, so they are correlated | Resampling single games or encounters (overstates certainty) | Manas | T |
| D53 | Separate seed blocks per experiment (E1 +100000, E2 +200000, H2 +300000, E4 +400000) | Experiments are independent but reproducible | One shared seed range | Manas | — |
| D54 | Verdicts are computed by code from the pre-registered rules; exploratory analyses are labelled (H3 by measured strength, H4d) | No hand-judged outcomes; a clear line between confirmatory and exploratory | Judging the plots by eye | Manas | T |
| D55 | **Results reported as they came out: H2 refuted (depth 5 battles worse than depth 3); H3a–c not supported.** We did not change tie-breaking or the evaluation after seeing them | Pre-registration; the H2 mechanism (deep search proves a draw and gives up chances against a fallible opponent) is explained in the report and kept as future work | "Fixing" tie-breaking so that H2 passes | Manas / Harshit | measured |
| D56 | Quick mode writes only to `experiments/results_quick/` (git-ignored) | A smoke run can never overwrite the results the report is built from | Shared output folder | Manas | — |
| D57 | Figures: validated 3-slot palette (all-pairs CVD ΔE ≥ 9.2); one y-axis per panel; every figure has a generated table | Readable for colour-blind readers and in print; no number is only a picture | Default matplotlib colours | Manas | — |

### P7 — 2026-11-13 (demo)

| ID | Decision | Why | Alternatives rejected | Owner | Source |
|---|---|---|---|---|---|
| D58 | Streamlit app and rich CLI share `demo/render.py` and the same engine | The CLI is a real fallback with identical behaviour | A separate CLI implementation | Rohit | T |
| D59 | Every user action goes through one guard that turns ConfigError, IllegalMoveError and ValueError into a message; the CLI exits with code 2 | Demo-track criterion: graceful handling of unexpected input | Per-widget validation only | Rohit / Jainam | T |
| D60 | The Streamlit app is tested headless with `streamlit.testing.v1.AppTest` | The GUI is covered by CI, not just by manual clicking | Manual testing only | Jainam | — |

### P8 — 2026-11-13 (deliverables)

| ID | Decision | Why | Alternatives rejected | Owner | Source |
|---|---|---|---|---|---|
| D61 | The report is generated from `report/template.md`: placeholders for every number, `{{check:...}}` guards for every outcome-dependent sentence | No hand-typed results; the build fails if the narrative stops matching the data | Writing numbers into the report by hand | Manas / Rohit | T |
| D62 | Individual Contributions entries are left for each member to write | Only the member can state truthfully what they did; the viva checks it | Pre-filling contribution claims | Whole team | guidelines §6 |

### P9 — 2026-11-13 (game-style web demo, requested by Rohit)

| ID | Decision | Why | Alternatives rejected | Owner | Source |
|---|---|---|---|---|---|
| D63 | Web demo on a Python standard-library HTTP server with plain HTML/CSS/JS, no external libraries; fonts fall back to system fonts offline | Works in a classroom without internet; no new dependencies to install or explain | React/Vite front end (build step, npm); FastAPI (new dependency) | Rohit | — |
| D64 | Every step is a *frame* holding a full board snapshot, so stepping back is instant. Frames are built from the engine's event log through a new `observer` hook | "Revert a step" needs past states; the event log already defines the steps | Re-simulating to step k each time; reconstructing state in JavaScript (duplicate rules) | Rohit / Vaibhav | — |
| D65 | Play mode is stateless: the page sends the seed plus your decisions and the server replays the game to the next decision | The game is deterministic given the seed and your choices, so undo and "choose differently here" are just shorter decision lists. There is no server session to lose, and the URL holds the whole game | Server-side sessions with undo stacks | Rohit | — |
| D66 | A traced game must equal the plain engine game. Explanations use no randomness; the tracing battle resolver consumes the random stream like `battle/play.py` | The demo shows the real system, not a re-enactment. `tests/test_web.py` compares scores, winners and encounter counts | A separate demo simulation | Rohit | — |
| D67 | `rank_root_moves` added to alpha-beta: the exact value of every root move, shown as ranked "engine lines" | Explains *why* a battle move was chosen, the way chess analysis boards do. Tested: the best value and first best move equal `alphabeta_search` | Showing only the chosen move | Raunak | — |
| D68 | Visual design: indigo "strategy table", enamel blue/orange tokens (colour-blind-validated pair, different emblem shapes per side), jade crystals, brass only for "where you are" (cursor, chosen cell, your turn). Big Shoulders Display for scores and titles, Atkinson Hyperlegible for explanations | The board is the memorable element; explanations must stay readable; identity never relies on colour alone | Neon-on-black or cream-and-terracotta default looks | Rohit | frontend-design guidance |
| D69 | Forms skip browser validation (`novalidate`), so bad input reaches the server and gets the same specific message as the CLI | One consistent error voice; found by the end-to-end test, where native validation blocked the form silently | Native validation bubbles | Rohit / Jainam | measured (e2e) |
| D70 | Units glide only on a single step forward or back; first draws and timeline jumps snap into place | Found in screenshots: tokens were caught mid-flight from the board corner | Always animating | Rohit | measured (screenshots) |

---

## Contributions (for the report's Individual Contributions section)

Record what each member actually did, per phase. Each member checks their own row before the
report is written.

| Member | Component | P0 | P1–P8 (2026-11-13) |
|---|---|---|---|
| Rohit Gangil | Integration, repo, CI, demo UI | Set up the repo; directed and reviewed the AI-assisted drafting; approved defaults 1–9 | Directed the AI-assisted build of all phases; demo modules (`demo/`) drafted: review pending |
| Vaibhav Malik | Game engine | Review DESIGN §3 — pending | `engine/` drafted by AI: owner review pending |
| Atharv Singhal | Encounter analysis | Review DESIGN §2 — pending | `encounter/` drafted by AI: owner review pending |
| Gursidak Singh Bassi | Mixed Nash / verifier | DESIGN.md sections 2.3 and 5 | Adapted the supplied equilibrium code and integrated the independent best-response verifier, exact linear-system solver, support enumeration, vertex enumeration, and equilibrium selection. Updated the solver to use the polytope test for degeneracy. Improved the Nashpy comparison by checking strategy lengths and requiring each equilibrium to match a distinct reference result. |
| Harshit Agrawal | Battle Mode / minimax | Review DESIGN §4 — pending | `battle/` rules, evaluation, minimax, play drafted by AI: owner review pending |
| Raunak Sarawgi | Alpha-beta / E3 | Review DESIGN §4.6–4.7 — pending | `battle/alphabeta.py`, `instrument.py`, `benchmark.py` drafted by AI: owner review pending |
| Tushya Jain | Agents | Review DESIGN §6 — pending | `agents/` drafted by AI: owner review pending |
| Manas Tripathi | Tournaments / stats | Review DESIGN §8, PREDICTIONS — pending | `tournament/`, `reproduce.py`, `report_builder.py` drafted by AI: owner review pending |
| Jainam Parekh Naresh | Tests / validation | Review skeleton tests — pending | `config.py` and `tests/` drafted by AI: owner review pending |

"Drafted by AI: owner review pending" is deliberate. The course guidelines require each member to
understand and explain their part. This table is updated only when the owner has actually reviewed,
run and understood the module.
