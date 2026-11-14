"""Proxy possession labels for the TRAINING games, derived by an algorithm from event data.

From saideepojha/football-possession-estimator (MIT) `labels.py`.
PASS / SHOT / RECOVERY / SET PIECE start possession for that event's team and it is carried
forward until the next event; BALL LOST / BALL OUT / CHALLENGE / FAULT RECEIVED make it
undetermined (NaN, excluded from training).

RULES.md: "Data of this kind about other games is fine." These events are about the Metrica
sample games, never about BAR-ATH. The labels are a proxy (events are approximate), see
TECHNICAL_DOCUMENT.md.
"""
from __future__ import annotations

import numpy as np

STARTS = {"PASS", "SHOT", "RECOVERY", "SET PIECE"}
BREAKS = {"BALL LOST", "BALL OUT", "CHALLENGE", "FAULT RECEIVED"}


def possession_labels(events, frames):
    """Array (T,): 1 = home, 0 = away, NaN = undetermined."""
    ev = events.sort_values("Start Frame", kind="stable")
    ev_frames = ev["Start Frame"].to_numpy()
    ev_types = ev["Type"].to_numpy()
    ev_teams = ev["Team"].to_numpy()
    labels = np.full(len(frames), np.nan)
    current, j = np.nan, 0
    for i, f in enumerate(frames):
        while j < len(ev) and ev_frames[j] <= f:
            if ev_types[j] in STARTS:
                current = 1.0 if ev_teams[j] == "Home" else 0.0
            elif ev_types[j] in BREAKS:
                current = np.nan
            j += 1
        labels[i] = current
    return labels
