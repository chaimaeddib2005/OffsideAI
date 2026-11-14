"""Train / load the LightGBM possession model and turn it into evidence e in [-1, 1]."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FEATURES, model_features

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "possession_lgbm.txt"

PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 15,
    "min_data_in_leaf": 200,           # frames are 25 Hz and highly autocorrelated: be conservative
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "seed": 0, "bagging_seed": 0, "feature_fraction_seed": 0,
    "deterministic": True, "force_row_wise": True, "num_threads": 1, "verbosity": -1,
}
N_ROUNDS = 200


def train(feature_frames, label_arrays, valid_masks):
    """feature_frames: list of DataFrames; label_arrays: 1 home / 0 away / NaN; valid_masks:
    bool arrays (ball usable). Returns a lightgbm.Booster."""
    import lightgbm as lgb
    X = np.concatenate([f.to_numpy()[(m & ~np.isnan(y))]
                        for f, y, m in zip(feature_frames, label_arrays, valid_masks)])
    y = np.concatenate([y[(m & ~np.isnan(y))] for y, m in zip(label_arrays, valid_masks)])
    return lgb.train(PARAMS, lgb.Dataset(X, label=y, feature_name=FEATURES), N_ROUNDS)


def load(path=None):
    import lightgbm as lgb
    path = Path(path or MODEL_PATH)
    if not path.is_file():
        raise IOError("trained model not found: {}\nTrain it:  python -m src.learned.train "
                      "--metrica-dir <sample-data>/data".format(path))
    return lgb.Booster(model_file=str(path))


def p_home(booster, table):
    """P(home has the ball) per frame, NaN where there is no usable ball."""
    X = model_features(table)
    ok = table["ball_valid"].to_numpy(dtype=bool) & np.isfinite(X["nearest"].to_numpy())
    out = np.full(len(table), np.nan)
    if ok.any():
        out[ok] = booster.predict(X.to_numpy()[ok])
    return out


def learned_evidence(table, booster=None):
    """e = 2p - 1, 0 where the ball is unusable (the decoder then simply carries the state on)."""
    booster = booster or load()
    p = p_home(booster, table)
    e = np.where(np.isnan(p), 0.0, 2.0 * p - 1.0)
    return e.astype(np.float64)
