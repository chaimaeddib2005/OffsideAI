"""Evidence layer: turn each frame into one number e in [-1, 1].

    e > 0  the frame says HOME has the ball       e < 0  it says AWAY
    |e|    how much to trust that statement       e = 0  no information (carry on)

e = side * confidence, where

  side        = tanh(margin / margin_scale)         margin = d_away - d_home: who is closer, and by
                                                    how much. A 3 m lead is near-certain, 0.2 m is not.
  proximity   = exp(-(nearest_distance / R)^2)      a player 8 m from the ball is not "in control"
  not_flying  = 1 -> 0 as ball speed goes 6 -> 18 m/s   a ball in flight belongs to nobody yet
  interp      x interp_weight on interpolated frames
  no ball     e = 0

Why this beats "nearest player wins": during a pass the nearest player is often an opponent
standing in the lane, but proximity and speed push that vote towards zero.
"""

import numpy as np

from .config import Params


def compute_evidence(table, p=None):
    p = p or Params()
    d_home = table["d_home"].to_numpy(dtype=float)
    d_away = table["d_away"].to_numpy(dtype=float)
    margin = table["margin"].to_numpy(dtype=float)
    speed = table["ball_speed"].to_numpy(dtype=float)
    valid = table["ball_valid"].to_numpy(dtype=bool)

    side = np.tanh(margin / p.margin_scale)
    nearest = np.fmin(d_home, d_away)                       # NaN only if both are missing
    proximity = np.exp(-(nearest / p.control_radius) ** 2)
    not_flying = np.clip((p.speed_hi - speed) / (p.speed_hi - p.speed_lo), 0.0, 1.0)
    not_flying = np.where(np.isnan(speed), 1.0, not_flying)  # unknown speed: do not punish

    confidence = proximity * not_flying
    confidence = np.where(table["is_interp"].to_numpy(dtype=bool), confidence * p.interp_weight, confidence)

    e = side * confidence
    e = np.where(valid & np.isfinite(e), e, 0.0)
    return e.astype(np.float64)
