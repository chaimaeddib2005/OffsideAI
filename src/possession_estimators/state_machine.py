"""Estimator 2: a Schmitt trigger with a leaky evidence accumulator.

    acc <- decay * acc + e[t]            (e > 0 favours home)
    acc is clipped to [-hold*T, +hold*T]
    state H -> A when acc <= -T          state A -> H when acc >= +T

* Inertia. Evidence for the current team fills a reservoir (up to hold*T); the opponent has to
  drain it AND reach -T. That is "harder to switch than to stay" -- hysteresis.
* Margin. Evidence is already scaled by how clearly one team is closer, so a 0.2 m lead barely
  moves the accumulator.
* Duration. A switch needs several frames of confident evidence, so a one-touch challenge or a
  ball skimming past an opponent does not flip the state.
* Latency. With the defaults a genuine takeover is recognised after ~22 frames (0.9 s) of
  confident evidence. That delay is the price of not flipping on noise.
* Missing ball. e = 0: the accumulator just leaks towards zero and the state is held.
"""

import numpy as np

from . import config
from .evidence import compute_evidence
from .postprocess import decode_all

HOME, AWAY = config.HOME, config.AWAY


def decode_segment(e, switch, hold, decay, init):
    n = len(e)
    out = np.full(n, -1, dtype=np.int8)
    cap = hold * switch
    state, acc = -1, 0.0
    for t in range(n):
        acc = acc * decay + e[t]
        if acc > cap:
            acc = cap
        elif acc < -cap:
            acc = -cap
        if state == -1:
            if acc >= init:
                state = HOME
            elif acc <= -init:
                state = AWAY
        elif state == HOME:
            if acc <= -switch:
                state = AWAY
        else:
            if acc >= switch:
                state = HOME
        out[t] = state
    return out


def estimate(table, p, dead=None, evidence=None):
    e = compute_evidence(table, p) if evidence is None else evidence
    return decode_all(table, e,
                      lambda seg: decode_segment(seg, p.sm_switch, p.sm_hold, p.sm_decay, p.sm_init),
                      p, dead)
