# Pre-registered predictions (H1–H4)

**Committed before any experiment code exists.** This file must be committed on its own, before the
first experiment commit, so the git history proves the order. After that, it is never rewritten.
Any change is appended as a dated **Addendum** at the bottom, with its reason. Every outcome is
reported honestly, whether or not the prediction holds.

Shared definitions:

- **Win rate** = wins / games. Ties are reported separately and never folded in.
- **Wilson** 95% intervals are used for proportions.
- **Bootstrap** 95% intervals (10,000 resamples of whole games, or whole start positions for
  battle-level data) are used for differences.
- "X beats Y" ⇔ the 95% CI of the difference excludes 0.

The exact fractions below were derived by hand and checked with exact rational arithmetic before
this file was written (see AI_USAGE.md). Each one is pinned by a unit test in the phase noted.

---

## H1 — Alpha-beta is exact and cheaper (E3, P4)

Setup: the frozen E3 benchmark (500 seeded positions, generated once and never regenerated), at
depths 1–5. Both searches use the same move ordering and first-maximum tie-breaking.

- **H1a (exactness).** On every position and every depth, alpha-beta returns the **same root value
  and the same root move** as plain minimax. *Refuted by a single mismatch.*
- **H1b (savings).** At depth 5, the mean ratio of nodes (alpha-beta / minimax) over the benchmark
  is **≤ 0.5**, i.e. at least 50% fewer nodes. *Refuted if the mean ratio is > 0.5.*
  - "Nodes evaluated" counts every call to the recursive search, root included (DESIGN §4.6).
  - The bootstrap CI over positions is reported alongside.

Context: the Knuth–Moore best case at b = 8, d = 5 is 575 leaves, against 32,768 for full minimax.
Our move ordering is fixed rather than tuned, so we expect much less than the best-case saving, but
we predict at least 50%.

## H2 — More lookahead never hurts in Battle Mode (battle level, P6)

Setup: each depth d ∈ {1, 3, 5} plays a **fixed depth-1 opponent** from 500 frozen start positions,
each played twice with the first mover swapped (1,000 battles per depth). w_d is its win rate.

- **Prediction:** w₁ ≤ w₃ ≤ w₅.
- **Refutation rule:** H2 is refuted if, for any pair d < d′, the 95% CI of w_d′ − w_d lies
  entirely below 0.
- **The proposal's weaker criterion** ("increases with depth, at least up to a point") is reported
  separately. It is met if the CI of w₃ − w₁ lies entirely above 0.
- Note: w₁ is the self-play baseline, (1 − tie rate)/2 by symmetry. The tie rate is reported for
  every depth.

## H3 — A lookahead advantage propagates into the encounter game (E4, P6)

Setup: V = 6, c = 2, d = 1. For each depth pair dᵢ > dⱼ, measure each player's win probability
against the other (p_deep and p_shallow; ties make up the remainder), from 500 frozen start
positions played in both first-mover orders.

**Analytic derivation (made before any run; DESIGN §5).** In the fully mixed equilibrium, each
player's mix is set by the *opponent's* indifference condition. Each player's mix is therefore a
function of the **opponent's** win probability r:

- f(r) = (39, 24r − 3, 54 − 42r) / (90 − 18r), over (A, D, E).
- A player whose own win probability is p has equilibrium value (p + 8)/(5 − p).
- Since dA/dr > 0, dD/dr > 0 and dE/dr < 0, a player attacks and defends **more** and expands
  **less** when its **opponent** is stronger.
- These formulas are valid while both win probabilities are > 1/8, so the support stays full.
- At p = ½ they give the symmetric equilibrium (13/27, 3/27, 11/27) with value 51/27.
- Worked example: p_deep = 3/5, p_shallow = 2/5 (no ties) gives
  - deeper player: (65/138, 11/138, 31/69), value 43/22 ≈ 1.955;
  - shallower player: (65/132, 19/132, 4/11), value 42/23 ≈ 1.826.

**Predictions.**

- **H3a (measured).** The deeper player wins more battles: the CI of p_deep − p_shallow lies above 0.
  This overlaps H2 and is a precondition for H3b–c.
- **H3b (direction of the shift).** In the coupled bimatrix built from the measured probabilities:
  - the **deeper** player's equilibrium value is higher than the shallower player's;
  - the **deeper** player attacks *less* and expands *more* than the shallower player;
  - each mix and value equals the closed form above, exactly.
  This part is analytic once H3a holds. E4 checks that the solver, the closed form and nashpy agree
  on the measured numbers.
- **H3c (realized payoffs).** Simulate 2,000 V = 6 encounters per depth pair. Both players play the
  coupled equilibrium, and every A-vs-A is resolved by actually playing Battle Mode at the two depths.
  The deeper player's mean realized encounter payoff exceeds the shallower player's.
  *Refuted if the bootstrap CI of the difference includes or lies below 0.*

Viva point: the advantage shows up in the *opponent's* behaviour. The stronger battler gains because
the weaker one is pushed into attacking and defending more, not because the stronger one attacks more.

## H4 — Equilibrium play does not exploit; modelling the opponent does (E1, P6)

Setup: the E1 round robin at battle depth 3, p = ½, analysed **per encounter and per V** from the
encounter log. The full-game outcome depends on how often each V comes up, so it is not predicted
(see H4d).

Exact expected payoff per encounter, for the first agent named against the second:

| V | Equilibrium vs FixedMixed | Equilibrium vs PureHeuristic | ExpectedUtility* vs FixedMixed | ExpectedUtility* vs PureHeuristic |
|---|---|---|---|---|
| 2 | 19/20 vs −1/10 | 1 vs 1 | 19/20 vs −1/10 | 1 vs 1 |
| 4 | 19/10 vs 19/20 | 2 vs 2 | 19/10 vs 19/20 | 2 vs 0 † |
| 6 | **893/540 vs 17/9** (1.654 vs 1.889) | **28/27 vs 17/9** (1.037 vs 1.889) | 57/20 vs 2 | 5 vs −2 |
| 8 | **449/180 vs 25/9** (2.494 vs 2.778) | **19/9 vs 25/9** (2.111 vs 2.778) | 19/5 vs 61/20 | 7 vs −2 |

\* ExpectedUtility with a converged opponent model, i.e. playing the exact best response. Against
FixedMixed the best response is E at every V. A Dirichlet(1,1,1) prior already implies E at
V = 6, so convergence matters little there.
† At V = 4, A and E tie against an Expander. The lexicographic tie-break picks A (DESIGN §6).

**Predictions.**

- **H4a (equilibrium does not exploit).** At V ∈ {6, 8}, the Equilibrium agent's mean per-encounter
  payoff is **lower** than its opponent's, against both FixedMixed and PureHeuristic. Equilibrium play
  makes the opponent indifferent (the opponent earns the equilibrium value whatever it does); it does
  not exploit the opponent's mistakes.
  - Test: the bootstrap CI of (opponent − Equilibrium) per encounter lies above 0 at V = 6 and at
    V = 8.
- **H4b.** At V ∈ {2, 4}, Equilibrium **ties** PureHeuristic exactly (both play E) and **beats**
  FixedMixed, which attacks a cell where attacking is dominated.
- **H4c (modelling the opponent pays).** ExpectedUtility outscores FixedMixed per encounter at every V
  (CI of the difference above 0, all encounters included, no burn-in). The opponent model is reset at
  the start of every game, so games stay independent.
- **H4d (exploratory, no direction predicted).** Full-game win rates and mean score differences in
  E1, with CIs. The per-V results above predict that the ranking depends on the mix of V values at
  encounters.

## E5 — Structural predictions for the sensitivity sweep (P6)

Derived in DESIGN §2.3 for c > d at p = ½:

- (E,E) is the unique equilibrium, found by iterated strict dominance, iff **V < 2c**.
- V = 2c is degenerate (A only weakly dominated; a continuum of equilibria).
- V > 2c gives the cyclic, fully mixed equilibrium, which is **Pareto-dominated by (E,E)**.
- **d never changes which of these regimes applies**; it only moves the mixing probabilities.

On the ±25% grid the boundary falls at V = 3 (c = 1.5), V = 4 (c = 2) and V = 5 (c = 2.5), giving 9
degenerate grid points. *Refuted by any grid point whose computed structure differs from this rule.*

---

## Addenda

### Addendum 1: 2026-11-13, before any experiment was run (P5)

The † footnote in the H4 table describes the idealised *converged* best response, where the model
puts probability exactly 0 on D. The real ExpectedUtility agent keeps Dirichlet(1,1,1) pseudo-counts,
so its believed P(D) is always > 0. Against A at V = 4, E beats A by 6·P(D) > 0. In play, the agent
therefore chooses **E** against an Expander at V = 4, and that cell is realized as (2, 2), not (2, 0).
`tests/test_agents.py::test_expected_utility_tie_break_at_v4` documents this. The A-before-E
tie-break only applies to a degenerate belief.

No prediction rule changes: H4c concerns ExpectedUtility vs FixedMixed, where the best response is E
at every V either way.
