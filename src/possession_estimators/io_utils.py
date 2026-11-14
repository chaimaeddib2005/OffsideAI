"""Reading the feature table, the optional dead-ball mask, and writing submissions."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

REQUIRED = ["frame", "half", "graded", "ball_valid", "is_interp", "ball_speed",
            "d_home", "d_away", "margin", "ball_gap_len", "near_team"]


def load_features(path=None):
    path = path or config.FEATURES_PATH
    path = Path(path)
    if not path.is_file():
        alt = path.with_name("features.csv.gz")
        if alt.is_file():
            path = alt
        else:
            raise IOError("features table not found: {}\nBuild it first:  "
                          "python -m src.data_preprocessing.build_features".format(path))
    table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in table.columns]
    if missing:
        raise ValueError("features table lacks columns {} -- rebuild it with data_preprocessing".format(missing))
    if len(table) != config.N_FRAMES or not np.array_equal(table["frame"].to_numpy(), np.arange(len(table))):
        raise ValueError("features table must have {} rows with frame = 0..{}".format(
            config.N_FRAMES, config.N_FRAMES - 1))
    return table


def load_dead_mask(path, n_frames):
    """Boolean array, True where the ball is dead. Accepts
      * dead_ball mask: a .csv / .csv.gz / .parquet with columns frame,label where label is D / N / X
        (D = dead ball; N and X are not dead, X frames are handled by the `graded` flag anyway),
      * a file with an `is_dead` column,
      * a bare boolean .npy array.
    """
    path = str(path)
    if path.endswith(".npy"):
        mask = np.load(path).astype(bool)
    else:
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        if "frame" in df.columns:
            df = df.sort_values("frame")
            if not np.array_equal(df["frame"].to_numpy(), np.arange(len(df))):
                raise ValueError("dead mask frames must be exactly 0..{} (no gaps, no duplicates)".format(len(df) - 1))
        if "is_dead" in df.columns:
            mask = df["is_dead"].to_numpy().astype(bool)
        elif "label" in df.columns:
            mask = (df["label"].astype(str).str.strip().str.upper() == "D").to_numpy()
        else:
            raise ValueError("dead mask needs a 'label' (D/N/X) or an 'is_dead' column, found {}".format(list(df.columns)))
    if len(mask) != n_frames:
        raise ValueError("dead mask has {} rows, expected {}".format(len(mask), n_frames))
    return mask


def to_letters(codes, graded=None, outside="D"):
    letters = np.array(config.LETTERS)[codes]
    if graded is not None and outside != "D":
        letters = np.where(graded, letters, outside)
    return letters


def write_labels(path, letters):
    folder = os.path.dirname(str(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(str(path), "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(letters))
    return path


def possession_runs(codes, graded):
    """Count maximal runs of H or A inside graded frames; a D or a change of side ends one."""
    c = np.where(graded, codes, config.DEAD)
    start = (c != config.DEAD) & np.concatenate(([True], c[1:] != c[:-1]))
    n_runs = int(start.sum())
    alive = int((c != config.DEAD).sum())
    mean_s = alive / max(n_runs, 1) / config.FPS
    return n_runs, mean_s
