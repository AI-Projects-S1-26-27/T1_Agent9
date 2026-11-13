# Design: Strategic Agents in a Territory-Conquest Game

**Team Agent9 · CS F407 · Topic: Agents & Game Theory · Track: Demo** (both fixed by the approved proposal)

Status: P0 draft, 2026-11-13. Section owners are listed in [ROLES.md](ROLES.md). Pre-registered
predictions are in [PREDICTIONS.md](PREDICTIONS.md). Where this document and the approved proposal
disagree, the proposal wins; every deliberate deviation is listed in §9.

Reference numbers like [6] refer to the proposal's reference list, reproduced in §10.

---

## 1. Scope, gap and research questions

The live demo answers the proposal's three questions, plus one added in response to feedback:

1. Do agents that compute equilibrium strategies outperform fixed heuristics? (H4)
2. How much does bounded rationality (limited lookahead) cost an agent? (H2)
3. How much does alpha-beta pruning save in battles without changing outcomes? (H1)
4. *Added:* how does an advantage in sequential lookahead propagate into the simultaneous game? (H3)

**Gap (feedback point 1).** Standard tools treat the two layers as separate objects: nashpy [4] and
similar solvers analyse normal-form games, while frameworks such as OpenSpiel [8] run extensive-form
and search-based agents. In our environment the layers are **coupled**. The Attack-vs-Attack cell of
the encounter matrix is not a constant; it is computed from the *measured* Battle Mode win
probabilities at the two agents' search depths (§5). That lets us measure how an advantage in
sequential lookahead changes the equilibrium, and the payoffs, of the simultaneous game.

Wording rule for README and report: *"We did not find a teaching/demo environment that exposes this
coupling for live inspection."* We do **not** claim to be the first. TODO (team): verify this
statement against the literature before the report; add only citations someone has actually read.

---

## 2. Encounter game (simultaneous layer)

### 2.1 When an encounter happens

An encounter happens when, after the turn's simultaneous movement, an uncollected resource cell is
occupied by units of **both** players (§3.3). Each player then chooses one action from the reduced
set **A**ttack, **D**efend, **E**xpand. The proposal allowed "3×3 or 4×4"; we use 3×3.

### 2.2 Payoffs (locked; all in `config/payoffs.yaml`)

| Symbol | Meaning | Default |
|---|---|---|
| V | cell value | {2, 4, 6, 8} |
| c | attack cost, paid by every attacker | 2 |
| d | defend (fortify) cost, paid by every defender | 1 |
| p_row, p_col | each player's probability of winning Battle Mode | ½, ½ |

Row player's payoff (the column player's payoff is the mirror image, using p_col):

|       | A | D | E |
|---|---|---|---|
| **A** | p_row·V − c | −c | V − c |
| **D** | V − d | −d | −d |
| **E** | 0 | V | V/2 |

**Meaning of each action.** These justify the matrix entries:

- **Expand** builds a claim on the cell. Two builders split it (V/2 each).
- **Attack** seizes the cell from a builder but always pays c.
- **Defend** fortifies. It beats an attack and then controls the cell (V − d, attacker gets −c), but
  it builds nothing. So against a builder the defender just pays d and the uncontested builder takes
  the cell (V).
- **Attack vs Attack** starts Battle Mode (§4). The expected payoffs are p_row·V − c and p_col·V − c.
  - Battles can end in an exact tie (§4.4). On a tie both attackers still pay c and nobody gets V.
    So p_row + p_col ≤ 1, and **the column payoff is p_col·V − c, not (1 − p_row)·V − c** (decision 1).
  - With p_row = p_col = ½ the game is symmetric: the column payoff matrix is the transpose of the
    row matrix.

### 2.3 Structure of the game (derived; checked exactly for the defaults)

These results come from comparing payoffs entry by entry, with p = p_row = p_col = ½.

- **E strictly dominates A iff V < min(c/p, 2c)**, which is V < 4 at the defaults.
  - Against an attacker: E pays 0, A pays pV − c.
  - Against a builder: E pays V/2, A pays V − c.
  - After A is removed for both players, E strictly dominates D for every V > 0 (V > −d).
  - So for **V < 2c**, iterated elimination of strictly dominated strategies leaves **(E,E)**, the
    unique equilibrium ("peaceful expansion"). Here (E,E) is also Pareto-optimal.
- **V = 2c (V = 4 at the defaults) is a degenerate edge.**
  - E only *weakly* dominates A, and the best response to E is {A, E}.
  - The equilibrium set is a continuum: every profile in which both players mix only A and E, each
    playing E with probability ≥ ½.
  - Its extreme points are (E,E), (E, ½A+½E), (½A+½E, E) and (½A+½E, ½A+½E).
  - The selection rule (§6) picks (E,E), with payoff 2 each against 1 at the ½/½ corner.
- **V > 2c gives a cyclic, rock-paper-scissors structure.**
  - The best response to A is D, to D is E, and to E is A.
  - No strategy is dominated and there is no pure equilibrium; the unique equilibrium is fully mixed.
- **d never changes this structure** for c > d (true at every ±25% sweep point). It only moves the
  mixing probabilities.

**Worked results at c = 2, d = 1, p = ½.** All were checked exactly before P0 (see
[AI_USAGE.md](AI_USAGE.md)) and become `Fraction` unit tests in P2/P3:

| V | Dominance | Equilibria | Equilibrium payoff | Pareto |
|---|---|---|---|---|
| 2 | E ≻ A strictly; then E ≻ D | (E,E) unique | (1, 1) | (E,E) Pareto-optimal |
| 4 | E ≽ A weakly only | continuum (4 extreme points, above) | (2, 2) at the selected (E,E) | (E,E) Pareto-optimal |
| 6 | none | (13/27, 3/27, 11/27) for both, unique | (51/27, 51/27) ≈ 1.889 | (E,E) = (3, 3) Pareto-dominates it |
| 8 | none | (17/36, 1/6, 13/36) for both, unique | (25/9, 25/9) ≈ 2.778 | (E,E) = (4, 4) Pareto-dominates it |

The V = 6 row is the proposal's "(2,2) vs (4,4)" inefficiency story with our real numbers: both
players would be better off expanding peacefully, but (E,E) is not an equilibrium, because each
player would then prefer to attack.

**Why V = 4 needs care in code.**

- nashpy's methods disagree on V = 4:
  - `support_enumeration` returns only (E,E);
  - `vertex_enumeration` returns all four extreme points;
  - Lemke–Howson returns the ½/½ point.
- The P3 cross-check therefore compares our solver against `vertex_enumeration` on degenerate games.
  Comparing against `support_enumeration` alone would agree while missing three of the four extreme
  points.
- The ±25% sweep lands exactly on V = 2c at nine grid points (V = 3, 4, 5), so this case also shows
  up in E5.

### 2.4 Analysis report ("Game Analysis" card)

For every encounter the software reports:

- the payoff matrix;
- strictly and weakly dominated strategies, with the steps of iterated elimination;
- pure Nash equilibria;
- mixed Nash equilibria (P3);
- Pareto-optimal outcomes;
- whether the selected equilibrium is Pareto-dominated by some outcome.

---

## 3. Main game (environment; `config/game.yaml`)

### 3.1 Board

- 8×8 grid, 2 players, bases at (0,0) and (7,7), 3 units each, starting on the base.
- **Point-symmetric resources.** 6 cells are drawn without replacement by the seeded RNG from the
  first half of the board in row-major order (excluding the bases). Each is mirrored through the
  centre, (r,c) → (7−r, 7−c), giving 12 cells.
  - With `band` placement (the default, decision D29), only cells with |r + c − 7| ≤ 2 are eligible.
    That is the "front line" between the bases.
- **Regrowth (D28).** A used-up resource cell regrows with its original value 8 turns later. Without
  regrowth, encounters are too rare (§3.5).
  - Each mirror pair gets one value drawn from {2, 4, 6, 8}, so both players see the same resources.
  - No cell is its own mirror on an 8×8 board.
- 40-turn limit. **Score = resources collected + territory_weight × cells owned** (territory_weight = 1).
- Winner = higher final score. An equal score is a tie, reported separately.

### 3.2 Actions and the shared non-encounter policy

Unit actions: Move (one step, 4-neighbourhood), Collect, Expand, Attack, Defend, Pass. All players
choose simultaneously.

**Experimental control.** Outside encounters, *all* agents use the same deterministic policy below.
Agents differ **only** in their encounter policy and their Battle Mode depth, so any difference in
outcomes is attributable to the decision technique and nothing else. For each unit, in unit-index
order:

1. If the unit is standing on an uncollected resource cell → **Collect**.
2. Else, if any uncollected resource cell exists → **Move** one step toward the best one: the
   highest V / (distance + 1) (decision D29).
   - Distance is Manhattan.
   - A player's units claim distinct targets while free targets remain (D31).
   - Ties go to the first in player-relative row-major order.
   - The step reduces the row distance first, then the column distance.
3. Else, if the unit is standing on an unowned cell → **Expand** (claim it as territory).
4. Else → Move toward the nearest unowned cell; if there is none → **Pass**.

"Player-relative" means player 1 sees the board rotated 180°, so the two players' policies are exact
mirror images and neither side gets a tie-breaking advantage.

Attack and Defend are only legal inside an encounter. Elsewhere the engine rejects them with a clear
message (an illegal move).

### 3.3 Resolution order within a turn

1. Apply all moves simultaneously. Units of different players may share a cell.
2. Every uncollected resource cell now occupied by both players is **contested**. Each contested cell
   is one encounter, processed in absolute row-major order.
3. Uncontested Collect: the collector gains V, the cell is depleted and becomes owned by the
   collector (+territory).
4. Uncontested Expand claims the cell.

**Payoff-to-score invariant.** The score change applied for an encounter is *exactly* the matrix
entry for the realized pair of actions. For A-vs-A, the realized battle result is applied: V − c to
the winner and −c to the loser, or −c each on a tie. This invariant is what makes the game-theoretic
analysis predict tournament scores. To keep it exact:

- If someone obtained V (including the E-vs-E split), the cell is depleted and stays **unowned**, so
  no territory point is awarded on top of the matrix payoff.
- If nobody obtained V (D-vs-D, or an A-vs-A tie), the cell stays uncollected and is contested again
  next turn if both units are still on it.

### 3.4 Determinism

- One `numpy.random.Generator` drives every random draw: map, battle starts, mixed strategies and
  first mover. It is passed explicitly and there is no global random state.
- The same seed and the same config must produce a **byte-identical** game log. P1 adds a test for
  this.

### 3.5 Encounter-frequency gate (decision 9, P1)

Agents differ only at encounters. If encounters are rare, every agent ties and the demo cannot show
the "measurable difference" the instructor asked for. With symmetric resources and a mirrored policy,
units may mostly stay on their own half.

P1 must therefore report the **mean number of encounters per game over 200 seeds**, with every
encounter played as E, and the distribution of V at encounters.

- Suggested target: at least 4 encounters per game.

**Result (measured 2026-11-13, `experiments/results/P1_encounter_frequency.csv`):**

- **Without regrowth the gate fails:** 1.3–2.7 encounters per game, and up to half of all games
  have none.
- **With regrowth after 8 turns, band placement and value-based targeting:** 19.2 encounters per
  game, every game has at least one, and all four values of V occur.
- See decisions D28–D30.

### 3.6 Stretch goal

Fog of war is a stretch goal only, and the first item cut.

---

## 4. Battle Mode (sequential layer; `config/battle.yaml`)

### 4.1 State

- 2 units per side, each with HP 2–3 and a position of Front or Back (both drawn by the seeded RNG).
- A unit can also be Withdrawn (left the battle alive) or dead (HP 0).
- The state also records each unit's defending flag, the side to move and the ply count.

The proposal describes "a mini-board with a few units". This is that mini-board: two ranks per side.

### 4.2 Moves

On its ply, the side to move picks exactly **one** unit and **one** action:

| Action | Legal when | Effect |
|---|---|---|
| STRIKE(target) | own unit Front, target Front and alive | target loses 1 HP, or 0 if it is Defending |
| DEFEND | own unit in battle | guard holds until this side's next ply |
| RETREAT | own unit in battle | Front → Back; Back → Withdrawn |
| ADVANCE | own unit Back | Back → Front |

- Branching factor ≤ 8: a Front unit facing two Front enemies has 2 strikes + DEFEND + RETREAT, and
  a side has 2 units.
- **Fixed move order.** Units in index order; within a unit: STRIKE(target 0), STRIKE(target 1),
  DEFEND, RETREAT, ADVANCE. Move generation always returns legal moves in this order.

### 4.3 Who moves first (decision 4)

- In the main game, the first mover is drawn by the seeded RNG (a fair coin).
- When we estimate win probabilities (§5), every start position is played twice, once with each side
  moving first. This guarantees p(d, d) is symmetric.

### 4.4 Terminal states and winner (decisions 2 and 3)

The battle ends when either of these happens:

- **A side has no units left in the battle.** All its units are dead or withdrawn, and that side
  **loses**. Withdrawn units survive but do not count. Without this rule, a side could withdraw at
  full HP and "win".
- **The 10-ply cap is reached.** The winner is the side with higher total HP *among units still in
  the battle*. An exact tie leaves the cell neutral, and both attackers still pay c.

**Simultaneous death is unreachable.** One action per ply deals at most 1 damage to one unit, so two
units can never die on the same ply, and both sides can never be eliminated at once. P4 adds a
property test asserting this instead of special-casing it.

### 4.5 Evaluation and terminal utility (decision 5)

Both are shared by minimax and alpha-beta. Scores are from the searching side's point of view.

- **Terminal states:** U = 1000·outcome + (my HP − their HP), where outcome ∈ {+1, 0, −1}. On a tie
  the HP difference is 0 by definition, so U = 0.
- **Depth cutoff, non-terminal state:** eval = 10·(my HP − their HP) + 2·(my Front units − their
  Front units).
  - The HP term measures material, the actual win condition, which is why it gets the large weight.
  - The Front term is positional: only Front units can strike, so units at the Front are potential
    damage next ply.
  - HP counts only units still in the battle, consistent with the win rule.
- |eval| ≤ 10·6 + 2·2 = 64 < 1000, so a proven win always outranks any heuristic score.
- The evaluation function is never used at terminal states. Otherwise the Front term could score an
  exact HP tie as a win.

### 4.6 Search: minimax and alpha-beta

- Depth is counted in plies from the root.
- **Root tie-breaking: the first maximum.** The best move changes only on a strict `>`.
- **Why alpha-beta then returns the same *move* as minimax, not just the same value:**
  - At the root, alpha always equals the exact best value found so far.
  - A later child whose true value is ≤ alpha may be pruned or fail low, but it returns only a bound
    that is ≤ alpha, so under strict `>` it can never displace the incumbent.
  - The first child whose true value exceeds alpha is searched with an open window, so it returns
    its exact value.
  - Hence both algorithms select the first child that achieves the maximum.
  - This argument requires identical move ordering, which §4.2 fixes.
- **Instrumentation, logged for every battle decision:**
  - nodes evaluated (every call to the recursive search, root included);
  - leaf evaluations;
  - nodes pruned (children skipped by cutoffs);
  - wall time.

### 4.7 Depth-5 feasibility estimate (feedback point 3)

Measured in P4 (E3):

- With branching factor b = 8, the full tree to depth 5 has Σ_{k=0..5} 8^k = **37,449 nodes**.
- The Knuth–Moore best case for alpha-beta [5] is b^⌈d/2⌉ + b^⌊d/2⌋ − 1 = 512 + 64 − 1 =
  **575 leaves**.
- At roughly 50k–200k nodes per second in Python, plain minimax at depth 5 takes about **0.2–0.75 s**
  per decision. This is an upper estimate, because many states have fewer than 8 legal moves.
- **Gate:** if measured plain minimax at depth 5 exceeds **2 s per decision** on our laptops, shrink
  the battle (fewer HP or units). **Depth 5 is never dropped.**

---

## 5. Coupling the two layers

- p(dᵢ, dⱼ) = P(a depth-dᵢ player beats a depth-dⱼ player), with ties counted separately.
- It is estimated from seeded Battle Mode games over a frozen set of random start positions, each
  played with both first-mover orders (§4.3). Estimates carry Wilson 95% intervals.
- The estimates replace p_row and p_col in the A-vs-A cell. When the depths differ, the encounter
  game becomes a general (asymmetric) bimatrix game.

**The indifference principle (classic viva material).**

- In a fully mixed equilibrium, *my* mixing probabilities are whatever makes *the opponent*
  indifferent among their actions. So my mix depends only on **the opponent's** payoffs, and the
  opponent's mix depends only on mine.
- Changing my battle-win probability p_row changes only my payoff matrix. It therefore moves **the
  opponent's** equilibrium mix, not mine.
- Solving the indifference equations in general gives the opponent's mix as a function of my win
  probability p:
  - y_A = (2V − c + 3d) / (V(5 − p))
  - y_D = (V(p + 1) − 3c + d(2p − 1)) / (V(5 − p))
  - y_E = 2(V(1 − p) + 2c − d(p + 1)) / (V(5 − p))
  - my equilibrium value = (2V − c + d(p − 2)) / (5 − p)
- At V = 6, c = 2, d = 1 these give (13/27, 3/27, 11/27) and 51/27 at p = ½.
- Since 2V − c + 3d > 0, raising p:
  - raises the opponent's A and D probabilities;
  - lowers the opponent's E probability;
  - raises my value.
- Counter-intuitively, then, **the player who is better at battle attacks *less* in equilibrium**. Its
  own mix is pinned by the *weaker* opponent's payoffs, and the weaker opponent attacks *more*.
- This holds while the equilibrium stays fully mixed: V > 2c, and the formula gives positive
  probabilities for **both** players. Each player's mix depends on the other's win probability, so
  both p and q must be in range.
  - At V = 6, c = 2, d = 1 that means both win probabilities above 1/8.
  - At V = 8 the condition holds for every p.
  - At V = 5 it needs both above 2/7. For example, V = 5 with q = 1/4 has no fully mixed
    equilibrium; `tests/test_predictions_analytic.py` checks this.
- The pre-registered form of this prediction is H3 in [PREDICTIONS.md](PREDICTIONS.md).

**Which p each experiment uses (decision 7).**

- E1 and E2 use p = ½ from the config. This isolates the effect of depth on the battles themselves.
- E4 uses the measured p(dᵢ, dⱼ), which implies both agents know each other's depth (common
  knowledge). That assumption is stated in the report.

---

## 6. Agents (`agents/`; `config/agents.yaml`)

Interface (sketch, finalized in P5). The proposal's agent loop is *observe → evaluate → choose →
receive payoff → update*. The original prompt had only `choose`; `observe` is added so learning agents
can update (§9 b).

```python
class Agent(Protocol):
    name: str
    battle_depth: int
    def choose(self, state: EncounterState, rng: np.random.Generator) -> Action: ...
    def observe(self, state: EncounterState, my_action: Action,
                opp_action: Action, payoff: Fraction) -> None: ...
```

`EncounterState` carries V, the bimatrix built from the config (from this agent's point of view),
the turn number, and both battle depths.

| Agent | Encounter policy |
|---|---|
| PureHeuristic | Attack iff V > T (T = 5), else Expand. "Above a threshold", as in the proposal (decision e). |
| FixedMixed | (A, D, E) = (0.40, 0.35, 0.25) |
| Equilibrium | Plays its Nash mixed strategy for the current matrix. With several equilibria: maximize own expected payoff, then break ties lexicographically. |
| ExpectedUtility | Best response to a model of the opponent. The model is the opponent's empirical action frequencies plus a Dirichlet(1,1,1) prior (fictitious play), **kept separately for each V** (decision 8), because PureHeuristic's play depends on V. Ties are broken A before D before E. |
| DepthLimited(d), d ∈ {1,3,5} | Equilibrium encounter policy, so **only** the battle depth differs |

**Two documented caveats.**

- **Mis-coordination.** Two Equilibrium agents choose independently, so in games with several
  equilibria they can pick strategies from *different* equilibria (the Battle of the Sexes fixture
  shows this). In our encounter matrices only the V = 2c points have multiple equilibria, and there
  both agents select (E,E).
- **Tie-breaking.** Against an opponent *known* to Expand at V = 4, A (V − c = 2) and E (V/2 = 2)
  tie, and the lexicographic rule would pick A, leaving the opponent 0 instead of 2.
  - With the Dirichlet(1,1,1) prior, the believed P(D) is always > 0, so E is strictly better and
    the tie never occurs in play.
  - See the addendum in PREDICTIONS.md.

---

## 7. Independent correctness checks

- **Own exact solver, checked against nashpy.** Our support-enumeration solver uses `Fraction`
  arithmetic and is cross-checked against nashpy on every matrix used anywhere.
  - The primary comparison is nashpy's `vertex_enumeration`. nashpy's `support_enumeration` is
    reported as secondary, because float rounding makes it drop true equilibria on some games (D41).
  - Degenerate games are detected exactly (polytope labels) and solved by exact vertex enumeration
    (D40).
- **Independent best-response verifier.** Written separately from the solver, it confirms that no
  player gains by a unilateral deviation.
- **Degenerate games** such as V = 4 are reported (what support enumeration returns and why), never
  crashed on.
- **Known-answer fixtures**, with expected results written into the test file *before* the
  implementation:
  - Prisoner's Dilemma;
  - Matching Pennies;
  - Battle of the Sexes (2 pure equilibria + 1 mixed);
  - Stag Hunt;
  - our V = 2, 4, 6 matrices.
- **Battle Mode:**
  - alpha-beta and minimax return the same value *and* move on ≥ 500 seeded positions at depths 1–5;
  - minimax agrees with brute-force full enumeration on tiny positions;
  - Hypothesis property tests: no illegal moves, HP never negative, termination guaranteed,
    simultaneous death unreachable.

---

## 8. Evaluation and live demo (summary; details in `config/experiments.yaml`)

### Experiments

| ID | What | Size |
|---|---|---|
| E1 | Round robin of the 4 encounter-policy agents, depth 3 | 200 seeds per pairing, each played with sides swapped |
| E2 | DepthLimited 1v3, 1v5, 3v5 | 200 seeds per pairing, sides swapped |
| E3 | Alpha-beta vs minimax nodes and time at depths 1–5 | frozen 500-position benchmark set |
| E4 | Coupling: measure p(dᵢ,dⱼ), rebuild the bimatrix, recompute equilibria, compare with H3 | as configured |
| E5 | ±25% sensitivity on V, c, d plus a V sweep | as configured |

**H2 vs E2 (decision 6).**

- H2 is measured at **battle level**: each depth against a fixed depth-1 opponent from frozen start
  positions.
- E2 is the **full-game** head-to-head.

**Encounter-level logging.** E1 also logs every encounter (V, both actions, both payoffs), because
the H4 predictions are per encounter and per V (see PREDICTIONS.md).

### Statistics

- Wilson 95% CIs for win rates, with ties reported separately.
- Bootstrap 95% CIs (resampling whole games) for mean score differences.
- "X beats Y" ⇔ the CI of the difference excludes 0.

### Reproducibility

- `make reproduce` (or `python -m resource_war.reproduce`, from P6) regenerates every table and plot.
- Every result file stores its config, seeds and git commit hash.
- Report tables are generated from the result CSVs.

### Demo

- **Primary:** Streamlit.
- **Fallback:** a rich CLI on the same engine.
- **Setup panel:** choose agents, battle depth and seed; load a payoff config; replay a seed.
- **During play:**
  - the board is redrawn every turn;
  - a Game Analysis card appears for each encounter;
  - for each battle, minimax and alpha-beta are shown side by side: chosen move, root value, nodes,
    time, % saved.
- **Configuration explorer:** sliders for V, c, d and p. The analysis updates live: A becomes
  dominated below 2c, there is a weak-dominance edge at 2c, and a cyclic equilibrium appears above it.
- **Results panel:** precomputed plots with CIs, plus a live 20-game mini-tournament.
- **Robustness:** every bad input gets a clear message, never a stack trace:
  - illegal moves;
  - bad or missing YAML;
  - NaN or non-numeric payoffs;
  - wrong matrix shapes;
  - out-of-range depth or seed;
  - custom matrices.

---

## 9. Changes since the proposal

Copied into the report. Track and topic are unchanged.

| # | Proposal | Now | Why |
|---|---|---|---|
| a | Role table with shared components | One owner + one reviewer per component (ROLES.md). The evaluation function moves from the agent library (Tushya/Jainam) to Battle Mode (Harshit). | Feedback point 4: "who is accountable for what is not clean". |
| b | Agent loop "observe … update" | `observe()` kept in the interface alongside `choose()` | Fictitious play needs it; the proposal's loop has the update step. |
| c | "Mini-board" for battles | Two-rank (Front/Back) mini-board with ADVANCE added | Keeps depth 5 feasible; ADVANCE makes Back units able to re-enter the fight. |
| d | "Win rate increases with lookahead depth, at least up to a point" | H2 pre-registers the stronger "non-decreasing at every depth" | A falsifiable prediction. Results are reported against both statements. |
| e | Pure heuristic attacks when V is "above a threshold" | V > T, T = 5 | Matches the proposal's wording; differs from ≥ only at V = 5, which the sweep produces. |
| f | Payoffs "illustrative; final numbers in Week 2" | Locked in `config/payoffs.yaml` with worked equilibria (§2) | Feedback point 3. Adds the defend cost d and the D-vs-D, D-vs-E and E-vs-A cells the proposal left open. |
| g | "Depth 5 feasible" asserted | Analytic estimate (§4.7) and a measured P4 gate | Feedback point 3. |
| h | Three demo questions | Plus H3 (coupling) | Feedback point 1 (no stated gap). |
| i | 8-week timeline from 2 Oct | Implementation began 13 Nov (repo empty until then) | Reported honestly in the reflection section. |
| j | Resources collected once | Used-up cells regrow after 8 turns; band placement; value-based targeting | P1 gate: without regrowth, encounters were too rare for agents to differ (D28–D30). |
| k | Equilibria by support enumeration (nashpy as a check) | Plus exact vertex enumeration and an exact degeneracy test; nashpy compared by vertex enumeration | Degenerate games (V = 2c) and nashpy rounding failures (D40, D41). |

---

## 10. References (from the approved proposal; add only references a teammate has read and checked)

[1] J. Nash, "Equilibrium points in n-person games," PNAS 36(1), 1950. https://doi.org/10.1073/pnas.36.1.48
[2] Y. Shoham and K. Leyton-Brown, *Multiagent Systems*, CUP, 2009. https://doi.org/10.1017/CBO9780511811654
[3] R. Porter, E. Nudelman, Y. Shoham, "Simple search methods for finding a Nash equilibrium," GEB 63(2), 2008. https://doi.org/10.1016/j.geb.2006.03.015
[4] V. Knight and J. Campbell, "Nashpy," JOSS 3(30), 904, 2018. https://doi.org/10.21105/joss.00904
[5] D. Knuth and R. Moore, "An analysis of alpha-beta pruning," AI 6(4), 1975. https://doi.org/10.1016/0004-3702(75)90019-3
[6] S. Russell and P. Norvig, *AI: A Modern Approach*, 4th ed., 2020. https://aima.cs.berkeley.edu/
[7] H. Simon, "A behavioral model of rational choice," QJE 69(1), 1955. https://doi.org/10.2307/1884852
[8] M. Lanctot et al., "OpenSpiel," arXiv:1908.09453, 2019. https://arxiv.org/abs/1908.09453
[9] O. Vinyals et al., "Grandmaster level in StarCraft II…," Nature 575, 2019. https://doi.org/10.1038/s41586-019-1724-z
