"""Estimator 3: two-state HMM decoded with Viterbi.

Minimise, over every possible H/A sequence s[1..n]:

    sum_t  -lambda * e[t] * (+1 if s[t] = H else -1)      +      penalty * (number of switches)

The first term rewards agreeing with the evidence; the second charges a fixed price per switch.
Frames with e = 0 (no ball, ball in flight, nobody close) cost the same either way, so the state
simply persists through them. The penalty is the one knob that sets the possession count:
raising it merges possessions, lowering it splits them.

Unlike the state machine this is a global optimum: it looks at the whole segment, so it can
place a switch where the new team really took the ball (no detection latency) and discard a
blip using evidence from BOTH sides of it. The price is that it is offline -- fine here, since
the whole match is available.
"""

import numpy as np

from .evidence import compute_evidence
from .postprocess import decode_all


def decode_segment(e, penalty, lam):
    n = len(e)
    if n == 0:
        return np.empty(0, dtype=np.int8)
    back = np.zeros((n, 2), dtype=np.int8)          # back[t, s] = best previous state if now in s
    cost_h = cost_a = 0.0
    for t in range(n):
        stay_h, move_h = cost_h, cost_a + penalty
        stay_a, move_a = cost_a, cost_h + penalty
        if stay_h <= move_h:
            new_h = stay_h
        else:
            new_h = move_h; back[t, 0] = 1
        if stay_a <= move_a:
            new_a = stay_a
            back[t, 1] = 1
        else:
            new_a = move_a
        step = lam * e[t]
        cost_h = new_h - step
        cost_a = new_a + step
    state = 0 if cost_h <= cost_a else 1
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[t, state]
    return path


def estimate(table, p, dead=None, evidence=None):
    e = compute_evidence(table, p) if evidence is None else evidence
    return decode_all(table, e, lambda seg: decode_segment(seg, p.vit_penalty, p.vit_lambda), p, dead)
