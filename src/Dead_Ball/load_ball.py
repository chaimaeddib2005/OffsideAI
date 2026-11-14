"""Load  features and the raw-ATD player-speed signal.

 feature cache intentionally drops per-player data, so player speeds are streamed from
the raw ATD file in one pass. Only the mean over tracked players is kept, plus the number of
finite speed observations. A frame with fewer than PLAYER_MIN_OBS observations is marked unknown.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


REQUIRED_FEATURES = (
    "frame", "half", "graded", "ball_x", "ball_y", "ball_observed", "ball_valid",
    "is_interp", "is_spike", "frames_since_ball", "ball_gap_len", "ball_speed",
)


def read_features(path=C.FEATURES_PATH):
    path = Path(path)
    if not path.is_file():
        alt = path.with_name("features.csv.gz")
        if alt.is_file():
            path = alt
        else:
            raise IOError("features table not found: {}\nBuild it first:  "
                          "python -m src.data_preprocessing.build_features".format(path))
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = [name for name in REQUIRED_FEATURES if name not in df.columns]
    if missing:
        raise ValueError("features file is missing columns: {}".format(missing))
    if len(df) != C.N_FRAMES:
        raise ValueError("features has {} rows, expected {}".format(len(df), C.N_FRAMES))
    if not np.array_equal(df["frame"].to_numpy(), np.arange(C.N_FRAMES)):
        raise ValueError("features frame column is not 0..{} in order".format(C.N_FRAMES - 1))
    return df


def count_player_slots(meta_path=C.ATD_META_PATH):
    root = ET.parse(meta_path).getroot()
    slots = len(root.findall(".//Player"))
    if slots <= 0:
        raise ValueError("no Player entries found in {}".format(meta_path))
    return slots


def load_player_speeds(raw_path=C.ATD_RAW_PATH, meta_path=C.ATD_META_PATH,
                       n_frames=C.N_FRAMES):
    """Return a frame-indexed table of mean player speed and observation count.

    The raw file uses frame_id starting at 1. The tactical video frame is frame_id - 1.
    """
    n_slots = count_player_slots(meta_path)
    mean_speed = np.full(n_frames, np.nan, dtype=np.float64)
    n_speed = np.zeros(n_frames, dtype=np.int16)
    seen = np.zeros(n_frames, dtype=bool)

    with open(raw_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            head, _, tail = line.partition(":")
            try:
                frame_id = int(head)
            except ValueError:
                continue
            index = frame_id - 1
            if index < 0 or index >= n_frames:
                continue
            seen[index] = True

            field, _, _ = tail.partition(":")
            values = np.fromstring(field.replace(";", ","), sep=",")
            if values.size == n_slots * 3:
                speeds = values.reshape(n_slots, 3)[:, 2]
            else:
                groups = field.split(";")
                speeds = np.full(n_slots, np.nan, dtype=np.float64)
                for slot, group in enumerate(groups[:n_slots]):
                    if not group or group[0] in ("N", "n"):
                        continue
                    parts = group.split(",")
                    if len(parts) >= 3:
                        try:
                            speeds[slot] = float(parts[2])
                        except ValueError:
                            pass

            finite = speeds[np.isfinite(speeds)]
            n_speed[index] = finite.size
            if finite.size >= C.PLAYER_MIN_OBS:
                mean_speed[index] = finite.mean()

    if not seen.all():
        missing = int((~seen).sum())
        raise ValueError("raw ATD is missing {} frame ids".format(missing))

    return pd.DataFrame({
        "frame": np.arange(n_frames, dtype=np.int32),
        "player_speed_mean": mean_speed.astype(np.float32),
        "player_speed_count": n_speed,
    })


def build_signals(features, player_speeds):
    """Pack the signals used by the rule engine into one frame-indexed dictionary."""
    n = C.N_FRAMES
    if len(features) != n or len(player_speeds) != n:
        raise ValueError("feature and player-speed tables must both have {} rows".format(n))

    ball_speed = features["ball_speed"].to_numpy(dtype=np.float64)
    ball_valid = features["ball_valid"].to_numpy(dtype=bool)
    ball_x = features["ball_x"].to_numpy(dtype=np.float64)
    ball_y = features["ball_y"].to_numpy(dtype=np.float64)
    graded = features["graded"].to_numpy(dtype=bool)
    d_home = features["d_home"].to_numpy(dtype=np.float64)
    d_away = features["d_away"].to_numpy(dtype=np.float64)
    margin = features["margin"].to_numpy(dtype=np.float64)
    opponent_distance = np.where(margin >= 0, d_away, d_home)

    finite_speed = np.isfinite(ball_speed)
    trusted = (ball_valid & finite_speed & (ball_speed >= 0.0)
               & (ball_speed <= C.BALL_TRUST_MAX_SPEED))
    safe_speed = np.where(trusted, ball_speed, 0.0)

    player_speed = player_speeds["player_speed_mean"].to_numpy(dtype=np.float64)
    player_count = player_speeds["player_speed_count"].to_numpy(dtype=np.int16)

    out_of_pitch = np.zeros(n, dtype=bool)
    position_ok = ball_valid & np.isfinite(ball_x) & np.isfinite(ball_y)
    tol = C.PITCH_TOLERANCE_M
    out_of_pitch[position_ok] = (
        (ball_x[position_ok] < -tol)
        | (ball_x[position_ok] > C.PITCH_LENGTH + tol)
        | (ball_y[position_ok] < -tol)
        | (ball_y[position_ok] > C.PITCH_WIDTH + tol)
    )

    player_finite = np.isfinite(player_speed)
    player_active = (player_finite & (player_speed >= C.PLAYER_ACTIVE_SPEED))
    ball_alive = trusted & (safe_speed >= C.BALL_ALIVE_SPEED)
    strong_ball = trusted & (safe_speed >= C.BALL_STRONG_SPEED)

    return {
        "frame": np.arange(n, dtype=np.int32),
        "half": features["half"].to_numpy(dtype=np.int8),
        "graded": graded,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_speed": ball_speed,
        "ball_valid": ball_valid,
        "ball_observed": features["ball_observed"].to_numpy(dtype=bool),
        "is_interp": features["is_interp"].to_numpy(dtype=bool),
        "is_spike": features["is_spike"].to_numpy(dtype=bool),
        "ball_gap_len": features["ball_gap_len"].to_numpy(dtype=np.int32),
        "frames_since_ball": features["frames_since_ball"].to_numpy(dtype=np.int32),
        "safe_ball_speed": safe_speed,
        "trusted": trusted,
        "player_speed": player_speed,
        "player_speed_count": player_count,
        "player_finite": player_finite,
        "player_active": player_active,
        "ball_alive": ball_alive,
        "strong_ball": strong_ball,
        "out_of_pitch": out_of_pitch,
        "opponent_distance": opponent_distance,
    }
