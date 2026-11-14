"""Score estimators on a HELD-OUT Metrica game, with the repo's own scorer (possession_metrics).

Ground truth here is the event-derived proxy (labels.py): H / A where determined, X where
undetermined or outside the two halves. X is never scored, so this is a relative comparison between
estimators on identical frames, not a prediction of the challenge score. Two conditions:
  clean   the ball as Metrica tracked it. Already missing on ~40% of frames, like BAR-ATH (59.3%
          observed), so this is the REALISTIC condition
  masked  the same game with extra synthetic dropouts (~37% of frames usable): a harsher stress
          test than the challenge, not a closer match
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..possession_estimators import majority, viterbi
from ..possession_estimators.config import Params
from ..possession_estimators.evidence import compute_evidence
from ..possession_estimators.postprocess import decode_all
from .model import learned_evidence, p_home

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402

LETTERS = np.array(["H", "A", "D"], dtype=object)


def gt_letters(labels, half):
    out = np.full(len(labels), "X", dtype=object)
    out[labels == 1] = "H"
    out[labels == 0] = "A"
    out[half == 0] = "X"
    return out


def score(codes, labels, half):
    gt = gt_letters(labels, half)
    pred = LETTERS[np.asarray(codes, dtype=int)]
    pred = np.where(half == 0, "D", pred)
    res = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
    scored = np.isin(gt, ("H", "A"))
    acc = float((pred[scored] == gt[scored]).mean())
    return {"spell_f1": float(res["spell_f1_mean"]), "n_pred_spells": int(res["n_pred_spells"]),
            "n_gt_spells": int(res["n_gt_spells"]), "frame_acc": acc}


def evaluate_game(table, labels, booster, penalties):
    """Returns {method_name: metrics}. Same frames and same scorer for every method."""
    p = Params()
    half = table["half"].to_numpy()
    out = {}
    e_hand = compute_evidence(table, p)
    out["majority (published-baseline style)"] = score(
        majority.estimate(table, p, dead=None, evidence=e_hand), labels, half)
    out["viterbi, hand-written evidence"] = score(
        viterbi.estimate(table, p, dead=None, evidence=e_hand), labels, half)
    ph = p_home(booster, table)
    raw = np.where(ph > 0.5, 0.0, 1.0)                   # 0 = H, 1 = A
    raw[np.isnan(ph)] = np.nan
    raw = pd.Series(raw).ffill().bfill().fillna(0).to_numpy()   # no ball: carry the last call
    out["learned model alone, no decoding"] = score(raw, labels, half)
    e_learn = learned_evidence(table, booster)
    for pen in penalties:
        codes = decode_all(table, e_learn,
                           lambda seg, pen=pen: viterbi.decode_segment(seg, pen, p.vit_lambda),
                           p, None)
        out["viterbi, learned evidence, penalty %g" % pen] = score(codes, labels, half)
    e_mix = 0.5 * (e_hand + e_learn)
    for pen in penalties:
        codes = decode_all(table, e_mix,
                           lambda seg, pen=pen: viterbi.decode_segment(seg, pen, p.vit_lambda),
                           p, None)
        out["viterbi, hand+learned evidence, penalty %g" % pen] = score(codes, labels, half)
    return out
