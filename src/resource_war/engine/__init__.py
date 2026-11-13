"""Game engine: board, seeded turn loop, rules, shared movement policy.

Owner: Vaibhav Malik, Reviewer: Harshit Agrawal
Implemented in phase P1.

All randomness flows from one numpy Generator passed in explicitly, so a
(seed, config) pair always produces a byte-identical game log.
"""
