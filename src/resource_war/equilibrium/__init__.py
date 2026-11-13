"""Mixed Nash equilibria: exact support enumeration, nashpy cross-check, best-response verifier.

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal
Implemented in phase P3.

The verifier is written independently of the solver: it only checks that no
player gains from a unilateral deviation, so a solver bug cannot hide itself.
"""
