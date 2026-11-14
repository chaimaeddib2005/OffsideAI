"""Estimator 5: Viterbi decoding of LEARNED evidence.

Same decoder as `viterbi.py` (global optimum, one switch penalty, dead balls end a possession), but
the per-frame evidence comes from the LightGBM model in models/possession_lgbm.txt instead of the
hand-written formula in evidence.py. The model is trained only on other games (see
src/learned/train.py and TECHNICAL_DOCUMENT.md).

The evidence is computed once per table and cached on the table object, so a --sweep does not
re-run the model for every penalty value.
"""

import numpy as np

from . import viterbi
from .postprocess import decode_all


def _evidence(table):
    cached = table.attrs.get("learned_evidence")
    if cached is None:
        from ..learned.model import learned_evidence
        cached = learned_evidence(table)
        table.attrs["learned_evidence"] = cached
    return cached


def estimate(table, p, dead=None, evidence=None):
    # `evidence` (the hand-written one passed in by run_estimators) is deliberately ignored.
    e = _evidence(table)
    return decode_all(table, e,
                      lambda seg: viterbi.decode_segment(seg, p.learned_penalty, p.vit_lambda),
                      p, dead)
