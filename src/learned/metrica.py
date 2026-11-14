"""Load Metrica Sports sample games and express them in the P1 feature-table schema.

Adapted from saideepojha/football-possession-estimator (MIT) `data.py`. The difference: instead of
building a private feature set, each game is pushed through THIS repository's own
`data_preprocessing` cleaning + `build_table`, so a Metrica game and the BAR-ATH match end up in
the same table (same column names, same units, same ball cleaning rules).

Metrica tracking is normalised to [0, 1]; we scale to a 105 x 68 m pitch. 25 Hz, like BAR-ATH.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data_preprocessing import clean, features as p1_features

PITCH = (105.0, 68.0)
HOME, AWAY = 0, 1


def _read_tracking(path):
    raw = pd.read_csv(path, header=None, skiprows=3)
    meta = raw.iloc[:, :3].copy()
    meta.columns = ["period", "frame", "time"]
    xy = raw.iloc[:, 3:].to_numpy(dtype=float)
    n = xy.shape[1] // 2
    xy = xy.reshape(len(raw), n, 2)
    return meta, xy[:, :-1, :], xy[:, -1, :]          # meta, players, ball


def load_raw(data_dir, game):
    """Sample_Game_{1,2}: returns dict(players, ball, team, period, events). Coordinates 0-1."""
    d = Path(data_dir) / "Sample_Game_{}".format(game)
    stem = "Sample_Game_{}".format(game)
    meta, home, ball = _read_tracking(d / "{}_RawTrackingData_Home_Team.csv".format(stem))
    _, away, _ = _read_tracking(d / "{}_RawTrackingData_Away_Team.csv".format(stem))
    events = pd.read_csv(d / "{}_RawEventsData.csv".format(stem))
    players = np.concatenate([home, away], axis=1)
    team = np.array([HOME] * home.shape[1] + [AWAY] * away.shape[1], dtype=np.int8)
    return {"players": players, "ball": ball, "team": team,
            "period": meta["period"].to_numpy(), "frame": meta["frame"].to_numpy(),
            "events": events, "name": stem}


def blocks_from_period(period):
    """[(half, first, last)] inclusive indices, the format data_preprocessing.clean expects."""
    out = []
    for h in (1, 2):
        idx = np.flatnonzero(period == h)
        if len(idx):
            out.append((h, int(idx[0]), int(idx[-1])))
    return out


def mask_ball(ball, blocks, rng, target_frac=0.40, median_gap_s=2.0, fps=25):
    """SYNTHETIC extra ball dropouts, used as a stress test and as training augmentation.

    Metrica's ball is already missing on ~40% of graded frames as tracked (usable on 60.9% / 59.1%
    of them in games 1 / 2), almost exactly BAR-ATH's 59.3%. So the as-tracked tables are the
    realistic condition; this adds more dropout on top (leaving ~37% usable), which is HARSHER than
    the challenge. Gap lengths are log-normal (mostly short, a few long). The pattern is an
    assumption, not a measurement; TECHNICAL_DOCUMENT.md says so.
    """
    out = ball.copy()
    for _, first, last in blocks:
        n = last - first + 1
        missing = 0
        mu, sigma = np.log(median_gap_s * fps), 1.0
        t = first
        # alternate visible stretches and gaps so that ~target_frac of the half is missing
        mean_gap = float(np.exp(mu + sigma ** 2 / 2))
        mean_vis = mean_gap * (1 - target_frac) / target_frac
        while t <= last:
            vis = int(rng.exponential(mean_vis)) + 1
            t += vis
            if t > last:
                break
            gap = int(rng.lognormal(mu, sigma)) + 1
            out[t:min(t + gap, last + 1)] = np.nan
            missing += min(gap, last + 1 - t)
            t += gap
    return out


def build_table(raw, ball_dropout_seed=None):
    """P1-schema feature table for one Metrica game. With `ball_dropout_seed`, synthetic ball
    dropouts are applied first (training augmentation / stress test)."""
    players_m = raw["players"] * np.asarray(PITCH)
    ball_m = raw["ball"] * np.asarray(PITCH)
    blocks = blocks_from_period(raw["period"])
    if ball_dropout_seed is not None:
        ball_m = mask_ball(ball_m, blocks, np.random.default_rng(ball_dropout_seed))
    despiked, is_spike = clean.despike_ball(ball_m, blocks)
    ball_clean, is_interp = clean.fill_short_gaps(despiked, blocks)
    half = np.where(np.isin(raw["period"], (1, 2)), raw["period"], 0).astype(np.int8)
    return p1_features.build_table(players_m, ball_m, ball_clean, is_spike, is_interp,
                                   raw["team"], half)
