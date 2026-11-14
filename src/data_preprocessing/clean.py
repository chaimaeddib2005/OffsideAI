"""Metres, despiking, and short-gap filling for the ball. Players are only rescaled.

Everything here works per HALF. Nothing is ever interpolated across halftime, and nothing is
touched outside the two playing halves (those frames are not graded anyway).
"""

import numpy as np

from . import config


def to_metres(players, ball, pitch=config.PITCH):
    """0-1 -> metres. The pitch is not square, so each axis gets its own scale."""
    scale = np.asarray(pitch, dtype=float)
    return players * scale, ball * scale


def despike_ball(ball_m, blocks, max_speed=config.MAX_BALL_SPEED, fps=config.FPS,
                 passes=config.DESPIKE_PASSES):
    """Remove isolated impossible jumps.

    An observation is a spike when the speed needed to reach it from the previous observed
    sample AND to leave it for the next observed sample both exceed `max_speed`. A single bad
    frame trips both; a genuine fast pass trips neither. A ball that is simply re-acquired
    somewhere new after a gap trips only one (the gap makes the implied speed small anyway).

    Returns (cleaned copy, is_spike bool array).
    """
    out = ball_m.copy()
    spike = np.zeros(len(out), dtype=bool)
    for _, first, last in blocks:
        for _ in range(passes):
            seg = out[first:last + 1]                      # a view: edits land in `out`
            idx = np.flatnonzero(~np.isnan(seg[:, 0]))
            if len(idx) < 3:
                break
            pos = seg[idx]
            seconds = np.diff(idx) / fps
            speed = np.hypot(*np.diff(pos, axis=0).T) / seconds     # len(idx) - 1
            bad = np.zeros(len(idx), dtype=bool)
            bad[1:-1] = (speed[:-1] > max_speed) & (speed[1:] > max_speed)
            if not bad.any():
                break
            seg[idx[bad]] = np.nan
            spike[first + idx[bad]] = True
    return out, spike


def fill_short_gaps(ball_m, blocks, max_gap=config.MAX_GAP_FRAMES):
    """Linearly interpolate ball gaps of at most `max_gap` frames, inside a half, only when the
    ball is observed on BOTH sides. Longer gaps are left as NaN: guessing a ball position for
    several seconds invents data.

    Returns (filled copy, is_interp bool array).
    """
    out = ball_m.copy()
    filled = np.zeros(len(out), dtype=bool)
    for _, first, last in blocks:
        seg = out[first:last + 1]
        idx = np.flatnonzero(~np.isnan(seg[:, 0]))
        if len(idx) < 2:
            continue
        gaps = np.diff(idx) - 1
        for k in np.flatnonzero((gaps > 0) & (gaps <= max_gap)):
            s, e = idx[k], idx[k + 1]
            t = np.arange(s + 1, e)
            for c in range(2):
                seg[t, c] = np.interp(t, [s, e], [seg[s, c], seg[e, c]])
            filled[first + s + 1:first + e] = True
    return out, filled
