"""Self-test that needs NO match data. Builds a synthetic bundle in a temp folder (same file
formats as the real ATD) and checks the pipeline against starter/baseline.py.

    python -m src.data_preprocessing.selftest

What it proves: the parser reads the real text format, frame_id - 1 alignment is right, units
are metres, and `near_team_raw` is identical to what the published baseline computes from the
same file. It does NOT prove anything about the real data -- use qa.py for that.
"""

import os
import sys
import tempfile
import time

import numpy as np

from . import clean, config
from games import asset_paths, half_video_blocks


# ---------------------------------------------------------------- unit tests (tiny arrays)
def test_fill_short_gaps():
    ball = np.full((40, 2), np.nan)
    ball[[0, 5]] = [[0, 0], [5, 10]]            # gap of 4 -> filled
    ball[[10, 30]] = [[0, 0], [1, 1]]           # gap of 19 -> left alone
    out, filled = clean.fill_short_gaps(ball, [(1, 0, 39)], max_gap=12)
    assert filled[1:5].all() and not filled[11:30].any()
    assert np.allclose(out[1:5, 0], [1, 2, 3, 4]) and np.allclose(out[1:5, 1], [2, 4, 6, 8])
    assert np.isnan(out[11:30]).all()


def test_no_fill_across_halves():
    ball = np.full((40, 2), np.nan)
    ball[[10, 30]] = [[0, 0], [1, 1]]
    out, filled = clean.fill_short_gaps(ball, [(1, 0, 14), (2, 25, 39)], max_gap=50)
    assert not filled.any() and np.isnan(out[11:30]).all()


def test_despike():
    ball = np.zeros((20, 2)); ball[:, 0] = np.arange(20) * 0.1     # slow, steady ball
    ball[10] = [60.0, 30.0]                                         # one-frame teleport
    out, spike = clean.despike_ball(ball, [(1, 0, 19)])
    assert spike[10] and spike.sum() == 1 and np.isnan(out[10, 0])
    fast = np.zeros((20, 2)); fast[:, 0] = np.arange(20) * 1.0      # 25 m/s pass: legitimate
    _, spike = clean.despike_ball(fast, [(1, 0, 19)])
    assert not spike.any()


# ------------------------------------------------------------ integration test (synthetic ATD)
def make_bundle(root, n_frames, n_slots=46, seed=0):
    rng = np.random.default_rng(seed)
    folder = os.path.join(root, "tactical")
    os.makedirs(folder)
    paths = asset_paths(data_dir=root)

    half = n_slots // 2
    players_xml = "".join('<Player id="p{}" teamId="{}"/>'.format(s, "T1" if s < half else "T2")
                          for s in range(n_slots))
    with open(paths["atd_meta"], "w") as handle:
        handle.write('<main><Team id="T1"><Name>Home FC</Name></Team>'
                     '<Team id="T2"><Name>Away FC</Name></Team><Players>{}</Players></main>'
                     .format(players_xml))

    pos = rng.uniform(0.1, 0.9, (n_slots, 2))
    ball = np.array([0.5, 0.5])
    present = np.zeros(n_slots, bool); present[:11] = True; present[half:half + 11] = True
    ball_visible = True
    with open(paths["atd_raw"], "w") as handle:
        for i in range(n_frames):
            pos = np.clip(pos + rng.normal(0, 0.0015, pos.shape), 0, 1)
            ball = np.clip(ball + rng.normal(0, 0.003, 2), 0, 1)
            if rng.random() < (0.02 if ball_visible else 0.03):
                ball_visible = not ball_visible
            groups = []
            for s in range(n_slots):
                if present[s] and rng.random() > 0.01:
                    groups.append("{:.5f},{:.5f},1.0".format(*pos[s]))
                else:
                    groups.append("NaN,NaN,NaN" if s % 2 else "NaN")      # both missing styles
            if ball_visible:
                b = ball + (rng.normal(0, 0.5, 2) if rng.random() < 0.0005 else 0)  # rare spike
                b_txt = "{:.5f},{:.5f},3.0".format(*np.clip(b, -1, 2))
            else:
                b_txt = "NaN,NaN,NaN"
            handle.write("{}:{}:{}\n".format(i + 1, ";".join(groups), b_txt))
    return paths


def test_pipeline_matches_baseline():
    from starter import baseline
    from . import build_features as bf
    with tempfile.TemporaryDirectory() as tmp:
        n = config.N_FRAMES
        print("  generating synthetic ATD ({} frames) ...".format(n))
        t0 = time.time()
        paths = make_bundle(tmp, n)
        print("  done in {:.0f}s".format(time.time() - t0))

        table, team, _ = bf.build(data_dir=tmp, use_cache=False)
        assert len(table) == n and np.array_equal(table["frame"], np.arange(n))

        ref = baseline.nearest_player_labels(paths["atd_raw"], baseline.team_of_each_track(paths["atd_meta"]),
                                             n, config.PITCH)
        mine = table["near_team_raw"].to_numpy()
        assert np.array_equal(mine, ref), "near_team_raw differs from baseline on {} frames".format(
            int((mine != ref).sum()))

        blocks = half_video_blocks(config.GAME)
        assert (table["half"] == 0).sum() == n - sum(b - a + 1 for _, a, b in blocks)
        assert not table.loc[table["half"] == 0, "is_interp"].any()
        assert (table.loc[table["ball_valid"], "frames_since_ball"] == 0).all()
        both = table["margin"].notna()
        assert np.allclose(table.loc[both, "margin"], (table["d_away"] - table["d_home"])[both], atol=1e-3)
        print("  near_team_raw identical to starter/baseline.py on all {} frames".format(n))
        print("  interpolated {} frames, removed {} spikes".format(
            int(table["is_interp"].sum()), int(table["is_spike"].sum())))


def main():
    for test in (test_fill_short_gaps, test_no_fill_across_halves, test_despike,
                 test_pipeline_matches_baseline):
        print("{} ...".format(test.__name__))
        test()
        print("  ok")
    print("\nall self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
