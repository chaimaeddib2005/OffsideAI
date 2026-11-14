"""Features for the learned model. They are computed from the P1 feature table ONLY.

That is the whole point: the same function runs on a Metrica training game and on the BAR-ATH
match, because both are expressed in the P1 schema. Anything the challenge table lacks (the
second-nearest player, ball height, player identities, absolute pitch side) is deliberately not
used. ball_x / ball_y are also left out: a model that sees where the ball is can learn which end
the home team attacks in the training game, which is not transferable.

Columns used from the table: half, ball_valid, is_interp, ball_speed, d_home, d_away, margin,
n_home_near, n_away_near.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FPS = 25
CTRL_RADIUS = 2.5      # m: a player this close to a slow ball is "in control"
CTRL_MAX_SPEED = 12.0  # m/s: above this the ball is in flight, nobody controls it
AGE_CAP_S = 10.0       # context older than this is as good as unknown

FEATURES = [
    "d_home", "d_away", "margin", "nearest", "ball_speed",
    "n_home_near", "n_away_near", "is_interp",
    "home_closing", "away_closing", "margin_mean_25", "margin_mean_75",
    "last_ctrl_team", "last_ctrl_age", "next_ctrl_team", "next_ctrl_age",
]


def _ctrl_context(sign, ctrl):
    """Inside ONE half. sign = +1 home / -1 away per frame; ctrl = bool 'someone is in control'.
    Returns team and age (s) of the previous and the next controlling frame (NaN if none)."""
    n = len(ctrl)
    idx = np.arange(n)
    last_i = np.maximum.accumulate(np.where(ctrl, idx, -1))
    next_i = np.minimum.accumulate(np.where(ctrl, idx, n)[::-1])[::-1]
    has_last, has_next = last_i >= 0, next_i < n
    last_team = np.where(has_last, sign[np.clip(last_i, 0, n - 1)], np.nan)
    next_team = np.where(has_next, sign[np.clip(next_i, 0, n - 1)], np.nan)
    last_age = np.where(has_last, np.minimum((idx - last_i) / FPS, AGE_CAP_S), np.nan)
    next_age = np.where(has_next, np.minimum((next_i - idx) / FPS, AGE_CAP_S), np.nan)
    return last_team, last_age, next_team, next_age


def model_features(table):
    """DataFrame (len(table) x len(FEATURES)) float32, NaN where unknown."""
    n = len(table)
    half = table["half"].to_numpy()
    d_home = table["d_home"].to_numpy(dtype=float)
    d_away = table["d_away"].to_numpy(dtype=float)
    margin = table["margin"].to_numpy(dtype=float)
    speed = table["ball_speed"].to_numpy(dtype=float)
    valid = table["ball_valid"].to_numpy(dtype=bool)

    f = pd.DataFrame({
        "d_home": d_home, "d_away": d_away, "margin": margin,
        "nearest": np.fmin(d_home, d_away),
        "ball_speed": speed,
        "n_home_near": table["n_home_near"].to_numpy(dtype=float),
        "n_away_near": table["n_away_near"].to_numpy(dtype=float),
        "is_interp": table["is_interp"].to_numpy(dtype=float),
    })
    for col in ("home_closing", "away_closing", "margin_mean_25", "margin_mean_75",
                "last_ctrl_team", "last_ctrl_age", "next_ctrl_team", "next_ctrl_age"):
        f[col] = np.nan

    nearest = f["nearest"].to_numpy()
    ctrl = valid & (nearest <= CTRL_RADIUS) & (np.nan_to_num(speed, nan=99.0) <= CTRL_MAX_SPEED)
    sign = np.where(np.nan_to_num(margin, nan=0.0) >= 0, 1.0, -1.0)

    for h in (1, 2):
        sel = np.flatnonzero(half == h)
        if len(sel) == 0:
            continue
        a, b = sel[0], sel[-1] + 1
        sl = slice(a, b)
        f.loc[f.index[sl], "home_closing"] = pd.Series(d_home[sl]).diff().to_numpy() * FPS
        f.loc[f.index[sl], "away_closing"] = pd.Series(d_away[sl]).diff().to_numpy() * FPS
        for w in (25, 75):
            f.loc[f.index[sl], "margin_mean_%d" % w] = (
                pd.Series(margin[sl]).rolling(w, center=True, min_periods=1).mean().to_numpy())
        lt, la, nt, na = _ctrl_context(sign[sl], ctrl[sl])
        f.loc[f.index[sl], "last_ctrl_team"] = lt
        f.loc[f.index[sl], "last_ctrl_age"] = la
        f.loc[f.index[sl], "next_ctrl_team"] = nt
        f.loc[f.index[sl], "next_ctrl_age"] = na
    return f[FEATURES].astype(np.float32)
