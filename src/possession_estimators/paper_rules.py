"""Team-level adaptation of the possession rules in Vidal-Codina et al. (2022).

The paper uses individual player control zones and checks ball trajectory/speed changes
within a player's control sequence. The cached project features retain only each team's
nearest-player distance, so this implementation uses team-level control episodes instead.
"""

import numpy as np

from . import config
from .postprocess import merge_short_runs, segments


def _control_teams(table, p):
    d_home = table["d_home"].to_numpy(dtype=float)
    d_away = table["d_away"].to_numpy(dtype=float)
    valid = table["ball_valid"].to_numpy(dtype=bool)

    home_near = valid & np.isfinite(d_home) & (d_home <= p.paper_pz_radius)
    away_near = valid & np.isfinite(d_away) & (d_away <= p.paper_pz_radius)
    duel = (
        valid
        & np.isfinite(d_home)
        & np.isfinite(d_away)
        & (d_home <= p.paper_duel_radius)
        & (d_away <= p.paper_duel_radius)
    )

    control = np.full(len(table), -1, dtype=np.int8)
    control[home_near & ~away_near & ~duel] = config.HOME
    control[away_near & ~home_near & ~duel] = config.AWAY
    return control


def _episodes(control, start, end):
    """Return contiguous team-control episodes as half-open (team, start, stop) tuples."""
    episodes = []
    frame = start
    while frame < end:
        team = int(control[frame])
        if team < 0:
            frame += 1
            continue
        stop = frame + 1
        while stop < end and control[stop] == team:
            stop += 1
        episodes.append((team, frame, stop))
        frame = stop
    return episodes


def _motion_vector(x, y, valid, frame, direction, window):
    """Estimate a local ball-motion vector over `window` frames."""
    other = frame + direction * window
    if other < 0 or frame < 0 or other >= len(x):
        return None
    if not valid[frame] or not valid[other]:
        return None
    if not (np.isfinite(x[frame]) and np.isfinite(y[frame])
            and np.isfinite(x[other]) and np.isfinite(y[other])):
        return None
    if direction < 0:
        vector = np.array([x[frame] - x[other], y[frame] - y[other]], dtype=float)
    else:
        vector = np.array([x[other] - x[frame], y[other] - y[frame]], dtype=float)
    return vector * (config.FPS / window)


def _contact_change(table, episode_start, episode_stop, p):
    """Check for a meaningful direction or speed change across a control episode."""
    x = table["ball_x"].to_numpy(dtype=float)
    y = table["ball_y"].to_numpy(dtype=float)
    valid = table["ball_valid"].to_numpy(dtype=bool)
    window = p.paper_motion_window_frames

    incoming = _motion_vector(x, y, valid, episode_start, -1, window)
    outgoing = _motion_vector(x, y, valid, episode_stop - 1, 1, window)
    if incoming is None or outgoing is None:
        return False

    incoming_speed = float(np.linalg.norm(incoming))
    outgoing_speed = float(np.linalg.norm(outgoing))
    speed_changed = abs(incoming_speed - outgoing_speed) >= p.paper_speed_delta
    direction_changed = False
    if incoming_speed > 0 and outgoing_speed > 0:
        cosine = float(np.dot(incoming, outgoing) / (incoming_speed * outgoing_speed))
        direction_changed = cosine <= p.paper_direction_cosine
    return speed_changed or direction_changed


def _decode_segment(table, control, start, stop, carry, p):
    episodes = _episodes(control, start, stop)
    if not episodes:
        return np.full(stop - start, carry, dtype=np.int8), carry

    owner = episodes[0][0]
    labels = np.full(stop - start, owner, dtype=np.int8)
    for team, episode_start, episode_stop in episodes:
        if team != owner and _contact_change(table, episode_start, episode_stop, p):
            owner = team
        labels[episode_start - start:stop - start] = owner

    min_len = int(round(p.min_run_seconds * config.FPS))
    labels = merge_short_runs(labels, min_len)
    return labels, int(labels[-1])


def estimate(table, p, dead=None, evidence=None):
    """Estimate H/A using team control zones and contact-change confirmation.

    `evidence` is accepted to match the estimator interface; this rule-based method does not use it.
    """
    del evidence
    n = len(table)
    codes = np.full(n, config.DEAD, dtype=np.int8)
    control = _control_teams(table, p)
    half = table["half"].to_numpy()
    carry = config.HOME
    for start, stop in segments(half, dead):
        decoded, carry = _decode_segment(table, control, start, stop, carry, p)
        codes[start:stop] = decoded
    return codes
