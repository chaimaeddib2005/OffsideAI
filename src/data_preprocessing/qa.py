"""Sanity checks on the feature table. Run automatically by build_features.

    python -m src.data_preprocessing.qa            # checks outputs/features.parquet

Each check is PASS / WARN / FAIL. FAIL means something is wrong with the pipeline (alignment,
parsing, units) and nothing downstream should be trusted until it is fixed. WARN means "look at
this". The numbers quoted come from DATA.md.
"""

import sys

import numpy as np
import pandas as pd

from . import config

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def _check_rows(t):
    ok = len(t) == config.N_FRAMES and np.array_equal(t["frame"].to_numpy(), np.arange(config.N_FRAMES))
    return (PASS if ok else FAIL,
            "{} rows, frame = 0..{} (expected {}; ~164k would mean the broadcast file)".format(
                len(t), len(t) - 1, config.N_FRAMES))


def _check_graded(t):
    n = int(t["graded"].sum())
    return (PASS if n == config.EXPECT_GRADED_FRAMES else FAIL,
            "{} graded frames (expected {})".format(n, config.EXPECT_GRADED_FRAMES))


def _check_ball_visibility(t):
    all_n = int(t["ball_observed"].sum())
    gr_n = int(t.loc[t["graded"], "ball_observed"].sum())
    exact = all_n == config.EXPECT_BALL_OBSERVED_ALL and gr_n == config.EXPECT_BALL_OBSERVED_GRADED
    close = (abs(all_n - config.EXPECT_BALL_OBSERVED_ALL) < 0.005 * config.N_FRAMES
             and abs(gr_n - config.EXPECT_BALL_OBSERVED_GRADED) < 0.005 * config.EXPECT_GRADED_FRAMES)
    return (PASS if exact else WARN if close else FAIL,
            "raw ball seen on {} frames overall (expect {}), {} graded (expect {})".format(
                all_n, config.EXPECT_BALL_OBSERVED_ALL, gr_n, config.EXPECT_BALL_OBSERVED_GRADED))


def _check_tracked(t):
    g = t[t["graded"]]
    mean = float((g["n_home"].astype(int) + g["n_away"].astype(int)).mean())
    diff = abs(mean - config.EXPECT_MEAN_TRACKED)
    return (PASS if diff <= 0.1 else WARN if diff <= 0.5 else FAIL,
            "mean tracked objects per graded frame {:.2f} (expect ~{})".format(
                mean, config.EXPECT_MEAN_TRACKED))


def _check_kickoff(t):
    """At the start of each half the ball should be near the centre spot."""
    from games import half_video_blocks
    cx, cy = config.PITCH[0] / 2, config.PITCH[1] / 2
    notes, worst = [], 0.0
    for half, first, _ in half_video_blocks(config.GAME):
        win = t.iloc[first:first + 100]
        win = win[win["ball_valid"]]
        if win.empty:
            notes.append("half {}: no usable ball in first 4 s".format(half))
            worst = max(worst, 99)
            continue
        r = win.iloc[0]
        dist = float(np.hypot(r["ball_x"] - cx, r["ball_y"] - cy))
        worst = max(worst, dist)
        notes.append("half {}: ball {:.1f} m from centre spot at frame {}".format(half, dist, int(r["frame"])))
    return (PASS if worst <= 6 else WARN, "; ".join(notes) + " (expect < 6 m)")


def _check_baseline(t):
    """Rebuild the 50-frame baseline from our column and compare with the published one."""
    if not config.BASELINE_SMOOTH50.is_file():
        return WARN, "{} not found, skipped".format(config.BASELINE_SMOOTH50.name)
    from starter.baseline import smooth
    mine = smooth(t["near_team_raw"].to_numpy(), 50)
    ref = pd.read_csv(config.BASELINE_SMOOTH50)["label"].to_numpy()
    if len(ref) != len(mine):
        return FAIL, "length differs: {} vs {}".format(len(mine), len(ref))
    g = t["graded"].to_numpy()
    agree = float((mine[g] == ref[g]).mean()) * 100
    return (PASS if agree >= 99.5 else WARN if agree >= 95 else FAIL,
            "rebuilt baseline_smooth50 agrees with the published file on {:.2f}% of graded frames "
            "(expect ~100%: confirms alignment, team assignment and units)".format(agree))


def _check_cleaning(t):
    g = t[t["graded"]]
    interp = 100 * g["is_interp"].mean()
    spikes = int(g["is_spike"].sum())
    valid = 100 * g["ball_valid"].mean()
    return (PASS if interp <= 5 and spikes < 0.002 * len(g) else WARN,
            "interpolated {:.2f}% of graded frames, {} spikes removed, ball available on {:.1f}%".format(
                interp, spikes, valid))


CHECKS = [("row count / index", _check_rows), ("graded frames", _check_graded),
          ("ball visibility", _check_ball_visibility), ("players per frame", _check_tracked),
          ("kickoff position", _check_kickoff), ("baseline regression", _check_baseline),
          ("cleaning audit", _check_cleaning)]


def run(table):
    print("\nQA")
    print("-" * 78)
    failed = 0
    for name, fn in CHECKS:
        status, message = fn(table)
        failed += status == FAIL
        print("[{}] {:<20} {}".format(status, name, message))
    print("-" * 78)
    print("{} check(s) FAILED".format(failed) if failed else "no FAIL -- table is safe to hand over")
    return 1 if failed else 0


def _load(path=None):
    path = path or config.FEATURES_PATH
    if not path.is_file():
        path = path.with_name("features.csv.gz")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


if __name__ == "__main__":
    sys.exit(run(_load()))
