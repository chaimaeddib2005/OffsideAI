"""Read the raw ATD text file into numpy arrays. The ONLY module that touches the raw data.

Raw line:  <frame_id>:<x,y,speed>;<x,y,speed>;...:<x,y,speed>
                      (46 player slots, ';'-separated)   (ball)
Missing observations are "NaN". Coordinates are normalised 0-1 (NOT metres).

The one alignment rule lives here and nowhere else:  video_frame = frame_id - 1.
"""

import os
import time

import numpy as np

from . import config
from games import require, source_frames
from starter.baseline import team_of_each_track


def _parse_players(field, n_slots):
    """(n_slots, 2) x,y for one line, or None if the line does not have the expected shape."""
    values = np.fromstring(field.replace(";", ","), sep=",")
    if values.size == 3 * n_slots:                       # fast path: every slot is "x,y,s"
        return values.reshape(n_slots, 3)[:, :2]
    out = np.full((n_slots, 2), np.nan)                  # slow path: a slot may be a bare "NaN"
    groups = field.split(";")
    if len(groups) != n_slots:
        return None
    for slot, group in enumerate(groups):
        if group[:1] != "N":
            xy = group.split(",")
            out[slot] = (float(xy[0]), float(xy[1]))
    return out


def _parse_file(raw_path, n_frames, n_slots):
    players = np.full((n_frames, n_slots, 2), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    seen = np.zeros(n_frames, dtype=bool)
    bad_lines = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1                        # <- the alignment rule
            if not 0 <= i < n_frames:
                bad_lines += 1
                continue
            xy = _parse_players(parts[1], n_slots)
            if xy is None:
                bad_lines += 1
                continue
            players[i] = xy
            b = np.fromstring(parts[2], sep=",")
            if b.size >= 2:
                ball[i] = b[:2]
            seen[i] = True
    if bad_lines:
        print("  warning: {} malformed/out-of-range lines skipped".format(bad_lines))
    return players, ball, seen


def load_atd(game=config.GAME, data_dir=None, use_cache=True, cache_dir=None):
    """Return (players, ball, team_of_slot), all in NORMALISED 0-1 coordinates.

    players : (n_frames, n_slots, 2) float64, NaN where a slot is not tracked
    ball    : (n_frames, 2)          float64, NaN where the ball is not seen
    team    : (n_slots,)             int8,    0 = home (Barcelona), 1 = away (Athletic)

    Parsing takes about a minute, so the result is cached as an .npz next to the outputs and
    reused while it is newer than the raw file.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    team = team_of_each_track(meta_path)
    n_slots = len(team)

    cache_dir = cache_dir or config.CACHE_DIR
    cache = os.path.join(str(cache_dir), "{}_atd_parsed.npz".format(game))
    if use_cache and os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(raw_path):
        data = np.load(cache)
        if data["players"].shape == (n_frames, n_slots, 2):
            print("  using cached parse: {}".format(os.path.relpath(cache, str(config.ROOT))))
            return data["players"], data["ball"], team

    start = time.time()
    print("  parsing {} ...".format(os.path.basename(raw_path)))
    players, ball, seen = _parse_file(raw_path, n_frames, n_slots)
    missing = int((~seen).sum())
    print("  parsed {} frames in {:.0f}s ({} frames absent from the file)".format(
        n_frames, time.time() - start, missing))

    if use_cache:
        os.makedirs(str(cache_dir), exist_ok=True)
        np.savez(cache, players=players, ball=ball)
    return players, ball, team
