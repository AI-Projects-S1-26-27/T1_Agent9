| Prediction | Pre-registered rule | Outcome | Evidence |
|---|---|---|---|
| H1a | alpha-beta returns minimax's value AND move on every benchmark position, depths 1-5 | supported | 2500 comparisons |
| H1b | mean node ratio (alpha-beta / minimax) at depth 5 <= 0.5 | supported | ratio 0.128 [0.119, 0.136] |
| Gate | plain minimax at depth 5 <= 2.0 s per decision | passed | max 0.373 s, mean 0.064 s |
| H2 | no significant DECREASE in battle win rate with depth (1 <= 3 <= 5) | refuted | proposal criterion (3 > 1 significant): met |
| H3a | deeper player wins more battles (CI of p_deep - p_shallow > 0) | not supported | 3v1: yes, 5v1: yes, 5v3: no |
| H3b | deeper player: higher value, attacks less, expands more; matches closed form; nashpy agrees | not supported | 3v1: yes, 5v1: yes, 5v3: no |
| H3c | deeper player's realized encounter payoff exceeds the shallower's (CI > 0) | not supported | 3v1: yes, 5v1: no, 5v3: no |
| H4a | at V = 6, 8 the Equilibrium agent earns LESS per encounter than FixedMixed and PureHeuristic | supported | vs FixedMixed V=6: yes, vs FixedMixed V=8: yes, vs PureHeuristic V=6: yes, vs PureHeuristic V=8: yes |
| H4b | at V = 2, 4 Equilibrium ties PureHeuristic exactly and beats FixedMixed | supported | vs PureHeuristic V=2: yes, vs PureHeuristic V=4: yes, vs FixedMixed V=2: yes, vs FixedMixed V=4: yes |
| H4c | ExpectedUtility outscores FixedMixed per encounter at every V | supported | V=2: yes, V=4: yes, V=6: yes, V=8: yes |
| E5 | structure rule (V < 2c pure (E,E); V = 2c degenerate; V > 2c cyclic and Pareto-dominated) holds at every grid point | supported | 0 mismatches over 108 grid points |
