"""Per-frame facts about the ball and the players around it. Collective features only:
the 46 track slots are tracklets, not people, so nothing here follows an individual player.
"""

import numpy as np
import pandas as pd

from . import config

HOME, AWAY, NOBODY = config.HOME, config.AWAY, config.NOBODY


def distance_matrix(players_m, ball_m):
    """(n_frames, n_slots) distance in metres from the ball to every slot. NaN if either is missing."""
    return np.hypot(players_m[..., 0] - ball_m[:, None, 0],
                    players_m[..., 1] - ball_m[:, None, 1])


def nearest_team(d, team):
    """Team code of the single closest tracked slot (0 home, 1 away, -1 nobody).

    Ties go to the lowest slot index, exactly as in starter/baseline.py, so that a baseline
    rebuilt from this column matches the published one.
    """
    dd = np.where(np.isnan(d), np.inf, d)
    j = dd.argmin(axis=1)
    best = dd[np.arange(len(dd)), j]
    out = team[j].astype(np.int8)
    out[np.isinf(best)] = NOBODY
    return out


def team_min_distance(d, team, which):
    dd = np.where(np.isnan(d[:, team == which]), np.inf, d[:, team == which])
    best = dd.min(axis=1)
    best[np.isinf(best)] = np.nan
    return best


def gap_columns(missing):
    """For a boolean 'ball unavailable' series:
    frames_since_ball = 1, 2, 3 ... inside each gap (0 where the ball is available)
    ball_gap_len      = total length of the gap this frame sits in (0 where available)
    """
    since = np.zeros(len(missing), dtype=np.int32)
    length = np.zeros(len(missing), dtype=np.int32)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        since[s:e] = np.arange(1, e - s + 1)
        length[s:e] = e - s
    return since, length


def ball_speed(ball_m, fps=config.FPS, window=config.SPEED_SMOOTH):
    """Ball speed in m/s: step length * fps, then a centred rolling mean. NaN without a ball."""
    step = np.hypot(np.diff(ball_m[:, 0]), np.diff(ball_m[:, 1])) * fps
    speed = np.concatenate(([np.nan], step))
    speed = pd.Series(speed).rolling(window, center=True, min_periods=1).mean().to_numpy(copy=True)
    speed[np.isnan(ball_m[:, 0])] = np.nan
    return speed


def build_table(players_m, ball_raw_m, ball_clean_m, is_spike, is_interp, team, half):
    """Assemble the one-row-per-frame feature table (see FEATURES.md for every column)."""
    n = len(ball_clean_m)
    d_clean = distance_matrix(players_m, ball_clean_m)
    d_raw = distance_matrix(players_m, ball_raw_m)

    d_home = team_min_distance(d_clean, team, HOME)
    d_away = team_min_distance(d_clean, team, AWAY)
    valid = ~np.isnan(ball_clean_m[:, 0])
    since, gap_len = gap_columns(~valid)

    tracked = ~np.isnan(players_m[..., 0])                      # (n, slots)
    near = d_clean <= config.NEAR_RADIUS                        # NaN compares False

    table = pd.DataFrame({
        "frame": np.arange(n, dtype=np.int32),
        "half": half.astype(np.int8),
        "graded": (half > 0),
        # ball, after cleaning
        "ball_x": ball_clean_m[:, 0].astype(np.float32),
        "ball_y": ball_clean_m[:, 1].astype(np.float32),
        "ball_observed": ~np.isnan(ball_raw_m[:, 0]),
        "ball_valid": valid,
        "is_interp": is_interp,
        "is_spike": is_spike,
        "frames_since_ball": since,
        "ball_gap_len": gap_len,
        "ball_speed": ball_speed(ball_clean_m).astype(np.float32),
        # who is close
        "d_home": d_home.astype(np.float32),
        "d_away": d_away.astype(np.float32),
        "margin": (d_away - d_home).astype(np.float32),         # > 0: home closer
        "near_team": nearest_team(d_clean, team),
        "near_team_raw": nearest_team(d_raw, team),             # raw ball: for the baseline check
        "n_home_near": (near[:, team == HOME]).sum(axis=1).astype(np.int8),
        "n_away_near": (near[:, team == AWAY]).sum(axis=1).astype(np.int8),
        # tracking quality
        "n_home": tracked[:, team == HOME].sum(axis=1).astype(np.int8),
        "n_away": tracked[:, team == AWAY].sum(axis=1).astype(np.int8),
    })
    return table
