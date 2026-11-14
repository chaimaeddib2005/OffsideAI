"""Adapter around the repository's Smart Tagging loader.

Smart Tagging is used as event hints: a set-piece tag identifies a likely restart nearby, and a
GOALS tag identifies a long stoppage. Tag intervals are never copied across as dead-ball labels.
"""

import numpy as np
import pandas as pd

from starter.load_smart_tagging import load as _load_smart_tagging

from . import config as C


CODE_SET_PIECES = "SET PIECES"
CODE_GOALS = "GOALS"


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def time_base_check():
    from starter.load_smart_tagging import check_time_base
    return check_time_base(_load_smart_tagging())


def load_events(fps=C.FPS):
    raw = _load_smart_tagging()
    rows = []
    for event_id, event in enumerate(raw):
        tags = event.get("tags") or {}
        code = event.get("code")
        code_norm = (code or "").strip().upper()
        event_type = tags.get("Type")
        type_key = (event_type or "").strip().lower()
        start_s = _safe_float(event.get("start_s"))
        end_s = _safe_float(event.get("end_s"))
        start_frame = event.get("start_frame")
        end_frame = event.get("end_frame")
        if start_frame is None or end_frame is None:
            start_frame = int(round(start_s * fps)) if np.isfinite(start_s) else -1
            end_frame = int(round(end_s * fps)) if np.isfinite(end_s) else -1
        valid_timing = (
            np.isfinite(start_s) and np.isfinite(end_s)
            and 0 <= start_frame < end_frame <= C.N_FRAMES
        )
        restart_type = C.RESTART_TYPE_MAP.get(type_key)
        rows.append({
            "event_id": event_id,
            "code": code,
            "code_norm": code_norm,
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "Team": tags.get("Team"),
            "Half": tags.get("Half"),
            "Type": event_type,
            "type_key": type_key,
            "valid_timing": bool(valid_timing),
            "is_set_piece": code_norm == CODE_SET_PIECES,
            "is_goal": code_norm == CODE_GOALS,
            "restart_type": restart_type,
        })
    df = pd.DataFrame(rows)
    df["is_restart"] = df["is_set_piece"] & df["restart_type"].notna()
    return df


def load_restarts(events=None):
    events = load_events() if events is None else events
    return events[events["is_restart"] & events["valid_timing"]].copy()


def load_goals(events=None):
    events = load_events() if events is None else events
    return events[events["is_goal"] & events["valid_timing"]].copy()


def link_goals_to_kickoffs(goals, restarts):
    """Associate each goal with the first Kick Off (After Goal) starting at/after the goal."""
    kickoffs = restarts[restarts["restart_type"] == "KICKOFF_AFTER_GOAL"].sort_values(
        "start_frame")
    links = []
    for _, goal in goals.sort_values("start_frame").iterrows():
        candidates = kickoffs[
            (kickoffs["start_frame"] >= goal["start_frame"])
            & (kickoffs["start_frame"] <= goal["end_frame"] + 60 * C.FPS)
        ]
        if len(candidates):
            kickoff = candidates.iloc[0]
            links.append({
                "goal_event_id": int(goal["event_id"]),
                "goal_start_frame": int(goal["start_frame"]),
                "goal_end_frame": int(goal["end_frame"]),
                "kickoff_event_id": int(kickoff["event_id"]),
                "kickoff_start_frame": int(kickoff["start_frame"]),
                "kickoff_end_frame": int(kickoff["end_frame"]),
            })
        else:
            links.append({
                "goal_event_id": int(goal["event_id"]),
                "goal_start_frame": int(goal["start_frame"]),
                "goal_end_frame": int(goal["end_frame"]),
                "kickoff_event_id": -1,
                "kickoff_start_frame": -1,
                "kickoff_end_frame": -1,
            })
    return pd.DataFrame(links)


def attach_goal_links(restarts, goals, links):
    restarts = restarts.copy()
    restarts["goal_event_id"] = -1
    for _, link in links.iterrows():
        mask = restarts["event_id"] == link["kickoff_event_id"]
        restarts.loc[mask, "goal_event_id"] = int(link["goal_event_id"])
    return restarts


def count_table(events):
    if len(events) == 0:
        return pd.DataFrame(columns=["code", "type", "count"])
    table = (events.assign(type=events["Type"].fillna(""))
             .groupby(["code_norm", "type"], dropna=False)
             .size().reset_index(name="count")
             .sort_values(["code_norm", "count"], ascending=[True, False]))
    table.columns = ["code", "type", "count"]
    return table
