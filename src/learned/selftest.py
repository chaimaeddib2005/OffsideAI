"""Self-test for the learned part. Needs NO match data and NO Metrica data (uses models/*.txt).

    python -m src.learned.selftest

Checks: the feature function runs on a table built by the real data_preprocessing code, never looks across
halftime, is deterministic, matches the feature list stored with the model; the model loads and
gives probabilities; evidence is exactly 0 where the ball is unusable; the `learned` and
`combined` estimators return a valid label array.
"""
import json
import sys

import numpy as np

from ..data_preprocessing import clean, features as p1
from ..possession_estimators import combined, config, learned
from ..possession_estimators.config import Params
from . import model as M
from .features import FEATURES, model_features


def synthetic_table(seed=0, n_half=3000, perturb_half2=False):
    rng = np.random.default_rng(seed)
    n = 2 * n_half
    half = np.r_[np.ones(n_half), 2 * np.ones(n_half)].astype(np.int8)
    ball = np.cumsum(rng.normal(0, 0.4, (n, 2)), axis=0) + np.array([52.0, 34.0])
    ball[:, 0] = np.clip(ball[:, 0], 1, 104)
    ball[:, 1] = np.clip(ball[:, 1], 1, 67)
    players = ball[:, None, :] + rng.normal(0, 6.0, (n, 22, 2))
    if perturb_half2:                                   # change ONLY the second half
        players[n_half:] += rng.normal(0, 3.0, (n_half, 22, 2))
        ball[n_half:] += 5.0
    ball[400:520] = np.nan                              # a ball dropout inside the first half
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    blocks = [(1, 0, n_half - 1), (2, n_half, n - 1)]
    despiked, spike = clean.despike_ball(ball, blocks)
    filled, interp = clean.fill_short_gaps(despiked, blocks)
    return p1.build_table(players, ball, filled, spike, interp, team, half)


def test_features():
    t = synthetic_table()
    f = model_features(t)
    assert list(f.columns) == FEATURES and len(f) == len(t)
    assert np.array_equal(f.to_numpy(), model_features(t).to_numpy(), equal_nan=True), "not deterministic"
    # Nothing from the second half may influence the first. KNOWN, DOCUMENTED EXCEPTION: data_preprocessing's
    # ball_speed is a centred 5-frame mean over the whole array, so the last 2 frames of half 1 see
    # the first frames of half 2 when the halves are adjacent (Metrica, this synthetic table). On
    # BAR-ATH a 20,156-frame halftime gap sits between them, so it does not occur there. Those 2
    # frames (and the controller-context values derived from their speed) are excluded here.
    f2 = model_features(synthetic_table(perturb_half2=True))
    assert np.array_equal(f.to_numpy()[:2998], f2.to_numpy()[:2998], equal_nan=True), "leak across halftime"
    print("  features: ok ({} columns, deterministic, no halftime leak beyond data_preprocessing's 2-frame speed window)"
          .format(len(FEATURES)))


def test_model_and_evidence():
    report = json.loads((M.MODEL_PATH.parent / "training_report.json").read_text())
    assert report["features"] == FEATURES, "model was trained on a different feature list"
    assert "BAR-ATH" in report["never_used"]
    t = synthetic_table()
    booster = M.load()
    e = M.learned_evidence(t, booster)
    assert e.shape == (len(t),) and np.isfinite(e).all() and np.abs(e).max() <= 1.0
    no_ball = ~t["ball_valid"].to_numpy(dtype=bool)
    assert no_ball.any() and (e[no_ball] == 0).all(), "evidence must be 0 where there is no ball"
    print("  model + evidence: ok ({} frames without a ball, all e = 0)".format(int(no_ball.sum())))


def test_estimators():
    t = synthetic_table()
    p = Params()
    for name, mod in (("learned", learned), ("combined", combined)):
        t.attrs.clear()
        codes = mod.estimate(t, p)
        assert len(codes) == len(t) and set(np.unique(codes)) <= {config.HOME, config.AWAY, config.DEAD}
        again = mod.estimate(t, p)
        assert np.array_equal(codes, again), name + " is not deterministic"
        dead = np.zeros(len(t), dtype=bool); dead[1000:1100] = True
        with_dead = mod.estimate(t, p, dead=dead)
        assert (with_dead[1000:1100] == config.DEAD).all(), "dead frames must come out as D"
        print("  estimator {}: ok".format(name))


def main():
    for test in (test_features, test_model_and_evidence, test_estimators):
        print(test.__name__ + " ...")
        test()
    print("\nall learned self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
