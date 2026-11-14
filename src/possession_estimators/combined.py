"""Estimator 6 (the default for the submission): Viterbi on the AVERAGE of two evidence sources.

    e = 0.5 * (hand-written physics evidence  +  learned LightGBM evidence)

* hand-written (evidence.py): nearest-team margin x proximity x not-in-flight. Needs no data, so it
  cannot be wrong about the tracking format, but it has no memory of who had the ball before.
* learned (src/learned): trained on Metrica sample games, it also sees which team last and next
  controlled the ball, so it handles passes in flight better, but it was trained on cleaner
  tracking than BAR-ATH's.

Averaging hedges the second risk with the first. On held-out Metrica games (train on the other
game, see models/cross_game_results.md) it had the best mean spellF1 of all variants (0.764 vs 0.752 for the
hand-written evidence alone, 0.595 for the majority vote) and was never the worst in any condition; the margin over either source alone is small (~0.01, inside the
scorer's noise band), so the honest reason to prefer it is robustness, not a large gain.
"""

import numpy as np

from . import viterbi
from .evidence import compute_evidence
from .learned import _evidence as learned_evidence
from .postprocess import decode_all


def estimate(table, p, dead=None, evidence=None):
    e_hand = compute_evidence(table, p) if evidence is None else evidence
    e = 0.5 * (e_hand + learned_evidence(table))
    return decode_all(table, e,
                      lambda seg: viterbi.decode_segment(seg, p.combined_penalty, p.vit_lambda),
                      p, dead)
