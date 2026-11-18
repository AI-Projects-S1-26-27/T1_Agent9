# The agent library

Owner: Tushya Jain. Code in `src/resource_war/agents/`, tests in `tests/test_agents.py`.

Every agent answers one question at each encounter: Attack, Defend or Expand. It also carries a
Battle Mode search depth (1, 3 or 5). Movement and collection are fixed in the engine, so any
difference between two agents comes from those two things only.

| Agent | What it does | Why it is in the study |
|---|---|---|
| PureHeuristic | Attacks if the cell value V is above 5, otherwise expands | A simple rule with no game theory and no learning |
| FixedMixed | Always picks 40% Attack, 35% Defend, 25% Expand | Unpredictable, but ignores V and the opponent |
| Equilibrium | Plays its Nash equilibrium strategy for the current encounter matrix | Cannot be exploited, but does not exploit weak opponents either |
| ExpectedUtility | Counts what the opponent has done at each V and plays the best response | Learns, so it can punish predictable play (and be fooled while its counts are thin) |
| DepthLimited(d) | The Equilibrium policy, with Battle Mode depth d | Comparing d = 1, 3, 5 isolates the effect of lookahead |

## Things worth knowing

- **Interface.** `choose(state, rng)` returns 0, 1 or 2. `observe(...)` is called after each
  encounter; only ExpectedUtility uses it. (Decision D15.)
- **Sampling.** Mixed strategies are sampled with one `rng.random()` against cumulative
  probabilities, so a seeded game replays identically on every machine (D34). `sample_action`
  now rejects empty, negative or non-normalised distributions before drawing, which does not
  change the random stream for valid input.
- **Threshold.** PureHeuristic uses `V > 5`, not `>=`. The two differ only at V = 5, which the
  value sweep produces (D16).
- **Learning model.** ExpectedUtility starts from pseudo-counts (1, 1, 1), keeps separate counts
  for each V, and starts fresh in every game so games stay independent (D17, D50). The prior keeps
  P(Defend) above zero, which is why it plays Expand rather than Attack against an Expander at
  V = 4 (PREDICTIONS Addendum 1, D51).
- **Equilibrium choice.** When a matrix has several equilibria, the agent picks the one with the
  best own payoff, then the lexicographically smallest (D18). At V = 4 that is (Expand, Expand).
