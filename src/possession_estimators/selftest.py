"""Self-test that needs NO match data.

    python -m src.possession_estimators.selftest

1. Unit tests on tiny arrays: hysteresis, Viterbi, short-run merging, dead-ball reset.
2. A simulated match with KNOWN possessions (carries, passes in flight, opponents pressing,
   tracker dropouts), scored with the real evaluation code. It proves the estimators behave as
   designed -- fewer false possessions than the vote, no extra delay -- NOT how well they do on
   the real match. Only the real tracking can tell you that.
"""

import os
import sys

import numpy as np
import pandas as pd

from . import config, majority, state_machine, vidal_codina, viterbi
from .config import Params
from .evidence import compute_evidence
from .postprocess import merge_short_runs, segments, decode_all
from .viterbi import decode_segment as vit_seg
from .state_machine import decode_segment as sm_seg

sys.path.insert(0, str(config.ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402

P = Params()


# ----------------------------------------------------------------------------- unit tests
def test_merge_short_runs():
    c = np.array([0] * 30 + [1] * 5 + [0] * 30 + [1] * 40, dtype=np.int8)
    out = merge_short_runs(c, 10)
    assert (out[:65] == 0).all() and (out[65:] == 1).all()
    assert np.array_equal(merge_short_runs(c, 1), c)
    edge = np.array([1] * 3 + [0] * 50, dtype=np.int8)
    assert (merge_short_runs(edge, 10) == 0).all()


def test_state_machine_hysteresis():
    # 100 frames of strong home evidence, 8 frames of strong away (a blip), then home again
    e = np.concatenate([np.ones(100), -np.ones(8), np.ones(100)])
    out = sm_seg(e, P.sm_switch, P.sm_hold, P.sm_decay, P.sm_init)
    assert (out[5:] == config.HOME).all(), "an 8-frame blip must not flip the state"
    # a real takeover: 60 frames of away evidence flips it, after a delay
    e = np.concatenate([np.ones(100), -np.ones(60)])
    out = sm_seg(e, P.sm_switch, P.sm_hold, P.sm_decay, P.sm_init)
    assert out[99] == config.HOME and out[-1] == config.AWAY
    flip = int(np.argmax(out == config.AWAY))
    assert 100 < flip < 140, "switch should come after ~20 frames, got {}".format(flip - 100)
    # no evidence: nothing is decided
    assert (sm_seg(np.zeros(50), 6, 1.5, 0.99, 4) == -1).all()


def test_viterbi():
    e = np.concatenate([np.ones(100), -np.ones(8), np.ones(100)])
    assert (vit_seg(e, 20.0, 1.0) == config.HOME).all(), "a short blip costs more than it gains"
    e = np.concatenate([np.ones(100), -np.ones(60), np.zeros(30)])
    out = vit_seg(e, 20.0, 1.0)
    assert out[99] == config.HOME and out[100] == config.AWAY, "switch lands exactly on the change"
    assert (out[160:] == config.AWAY).all(), "no evidence: the state persists"
    assert len(np.unique(vit_seg(np.zeros(40), 20.0, 1.0))) == 1


def test_dead_ball_resets():
    n = 400
    t = pd.DataFrame({"half": np.ones(n, dtype=np.int8)})
    e = np.concatenate([np.ones(150), np.zeros(100), -np.ones(150)])
    dead = np.zeros(n, dtype=bool); dead[150:250] = True
    codes = decode_all(t, e, lambda s: sm_seg(s, P.sm_switch, P.sm_hold, P.sm_decay, P.sm_init), P, dead)
    assert (codes[150:250] == config.DEAD).all()
    assert codes[149] == config.HOME and codes[251] == config.AWAY
    assert segments(np.ones(10), None) == [(0, 10)]
    h = np.array([0, 1, 1, 1, 0, 2, 2, 0]); assert segments(h) == [(1, 4), (5, 7)]


def test_vidal_codina():
    n = 140
    half = np.ones(n, dtype=np.int8)
    home_distance = np.full(n, 5.0)
    away_distance = np.full(n, 5.0)
    home_distance[:40] = 0.8
    away_distance[40:100] = 0.8

    x = np.arange(n, dtype=float)
    x[60:] = 60 - np.arange(n - 60, dtype=float)
    table = pd.DataFrame({
        "half": half,
        "ball_valid": np.ones(n, dtype=bool),
        "d_home": home_distance,
        "d_away": away_distance,
        "ball_x": x,
        "ball_y": np.zeros(n),
    })
    codes = vidal_codina.estimate(table, P)
    assert codes[39] == config.HOME and codes[40] == config.AWAY
    assert (codes[40:100] == config.AWAY).all()

    table["ball_x"] = np.arange(n, dtype=float)
    unchanged = vidal_codina.estimate(table, P)
    assert (unchanged == config.HOME).all(), "a team switch without a ball-motion change is rejected"

    home_distance = table["d_home"].to_numpy(copy=True)
    away_distance = table["d_away"].to_numpy(copy=True)
    home_distance[40:100] = 1.0
    away_distance[40:100] = 1.0
    table["d_home"] = home_distance
    table["d_away"] = away_distance
    duel = vidal_codina.estimate(table, P)
    assert (duel == config.HOME).all(), "a duel must not assert a new team control"


# ------------------------------------------------------------------------------ simulation
def simulate(n=30000, seed=1):
    """A match with known possession. Distances are generated the way tracking behaves:
    the possessing team's nearest player is close during a carry and far during a pass, while
    the opponent hovers at random distances and sometimes presses hard on a carry."""
    rng = np.random.default_rng(seed)
    truth = np.empty(n, dtype=np.int8)
    t, side = 0, int(rng.integers(0, 2))
    while t < n:
        dur = int(rng.gamma(4, 100)) + 60                   # mean ~ 460 frames = 18 s
        truth[t:t + dur] = side
        t += dur
        side = 1 - side

    d_poss, d_opp, speed = np.empty(n), np.empty(n), np.empty(n)
    t = 0
    while t < n:
        L = int(rng.integers(20, 70))                       # carry / hold
        sl = slice(t, min(t + L, n)); m = sl.stop - sl.start
        d_poss[sl] = np.abs(rng.normal(0.8, 0.5, m)) + 0.3
        d_opp[sl] = rng.uniform(1.5, 10, m)
        speed[sl] = rng.uniform(1, 7, m)
        if rng.random() < 0.12:                             # an opponent presses: misleading, 1-3 s
            d_opp[sl] = rng.uniform(0.2, 0.9, m)
            d_poss[sl] = rng.uniform(0.9, 1.6, m)
        t += L
        if t >= n:
            break
        F = int(rng.integers(15, 45))                       # pass in flight
        sl = slice(t, min(t + F, n)); m = sl.stop - sl.start
        d_poss[sl] = rng.uniform(3, 14, m)
        d_opp[sl] = rng.uniform(2.5, 14, m)
        speed[sl] = rng.uniform(13, 24, m)
        t += F

    home_has = truth == config.HOME
    d_home = np.where(home_has, d_poss, d_opp)
    d_away = np.where(home_has, d_opp, d_poss)

    valid = np.ones(n, dtype=bool)                          # tracker dropouts
    t = 0
    while t < n:
        if rng.random() < 0.004:
            L = int(rng.integers(15, 120)); valid[t:t + L] = False; t += L
        t += 1
    d_home[~valid] = np.nan; d_away[~valid] = np.nan; speed[~valid] = np.nan

    gap = np.zeros(n, dtype=np.int32)
    since = 0
    for i in range(n):
        since = 0 if valid[i] else since + 1
        gap[i] = since
    table = pd.DataFrame({
        "frame": np.arange(n), "half": np.ones(n, dtype=np.int8), "graded": np.ones(n, dtype=bool),
        "ball_valid": valid, "is_interp": np.zeros(n, dtype=bool), "ball_speed": speed,
        "d_home": d_home, "d_away": d_away, "margin": d_away - d_home, "ball_gap_len": gap,
        "near_team": np.where(~valid, -1, np.where(d_home <= d_away, 0, 1)).astype(np.int8)})
    return table, truth


def score(truth, codes):
    runs = pm.valid_runs(truth)
    s = pm.spell_scores(truth, codes, runs)
    return s["mean_f1"], s["n_pred"], s["n_gt"]


def test_simulated_match():
    table, truth = simulate()
    e = compute_evidence(table, P)
    results = {}
    for name, mod in (("majority", majority), ("state_machine", state_machine), ("viterbi", viterbi)):
        codes = mod.estimate(table, P, evidence=e)
        f1, n_pred, n_gt = score(truth, codes)
        frame_acc = float((codes == truth).mean())
        results[name] = (f1, n_pred, frame_acc)
        print("  {:<14} spellF1 {:.3f}   possessions {:>3} (true {})   frame accuracy {:.3f}".format(
            name, f1, n_pred, n_gt, frame_acc))
    # raw nearest-team vote with NO smoothing, for scale
    raw = np.where(table["near_team"].to_numpy() < 0, 2, table["near_team"].to_numpy()).astype(np.int8)
    print("  {:<14} spellF1 {:.3f}   possessions {:>3}".format("raw nearest", score(truth, raw)[0], score(truth, raw)[1]))

    maj = results["majority"]
    for name in ("state_machine", "viterbi"):
        f1, n_pred, _ = results[name]
        assert f1 > maj[0], "{} should beat the majority vote on spellF1".format(name)
        assert n_pred < maj[1] * 1.2 + 1, "{} should not invent possessions".format(name)


def main():
    for test in (test_merge_short_runs, test_state_machine_hysteresis, test_viterbi,
                 test_dead_ball_resets, test_vidal_codina, test_simulated_match):
        print("{} ...".format(test.__name__))
        test()
        print("  ok")
    print("\nall self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
