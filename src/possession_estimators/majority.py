"""Estimator 1 (reference): nearest team, then a centred majority vote.

Same as the published baseline but computed from the cleaned feature table. It has no notion of
confidence, flight or inertia, which is exactly why the other two exist. Kept so every comparison
has the floor next to it. D appears where no ball was seen anywhere in the window.
"""

import numpy as np

from starter.baseline import smooth

from . import config


def estimate(table, p, dead=None, evidence=None):
    letters = smooth(table["near_team"].to_numpy(), p.majority_window)
    codes = np.where(letters == "H", config.HOME, np.where(letters == "A", config.AWAY, config.DEAD)).astype(np.int8)
    codes = np.where(table["graded"].to_numpy(), codes, config.DEAD).astype(np.int8)
    if dead is not None:
        codes[dead] = config.DEAD
    return codes
