"""Dead-ball rule engine.

The engine combines three kinds of evidence:

* tagged set pieces, used only to locate a restart and a nearby dead stretch;
* tagged goals, treated as long stoppages that end at the following kickoff;
* tracking-derived low-activity intervals: long missing-ball gaps and sustained out-of-pitch
  excursions.

Tag intervals are never copied as D labels. Every D interval has an explicit start, end, and
evidence record in the diagnostics.
"""

import numpy as np
import pandas as pd

from . import config as C
from . import smoothing


def runs_from_mask(mask, min_len=1):
    runs = []
    start = None
    for index, value in enumerate(mask):
        if value:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= min_len:
                runs.append((start, index - 1))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask) - 1))
    return runs


def merge_runs(runs, max_gap=0):
    merged = []
    for start, end in sorted(runs):
        if not merged:
            merged.append([start, end])
        elif start - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def interval_mask(intervals, n=C.N_FRAMES):
    mask = np.zeros(n, dtype=bool)
    for start, end in intervals:
        if start <= end:
            mask[start:end + 1] = True
    return mask


def dead_before_counts(active):
    dead_before = np.zeros(len(active), dtype=np.int32)
    run = 0
    for index, value in enumerate(active):
        dead_before[index] = run
        run = 0 if value else run + 1
    return dead_before


def last_active_indices(active):
    last = np.full(len(active), -1, dtype=np.int32)
    current = -1
    for index, value in enumerate(active):
        if value:
            current = index
        last[index] = current
    return last


def event_half_range(event):
    label = str(event.get("Half") or "").lower()
    if "1st" in label:
        return C.HALF_RANGES[1]
    if "2nd" in label:
        return C.HALF_RANGES[2]
    for frame in (int(event["start_frame"]), int(event["end_frame"]) - 1):
        half = C.half_of_frame(frame)
        if half:
            return C.HALF_RANGES[half]
    midpoint = (int(event["start_frame"]) + int(event["end_frame"])) // 2
    return min(C.HALF_RANGES.values(),
               key=lambda pair: abs(pair[0] - midpoint))


def _clip_to_range(start, end, half_range):
    low, high = half_range
    start = max(int(start), low)
    end = min(int(end), high)
    if start > end:
        return None
    return start, end


def _profile(event):
    return C.RESTART_PROFILES.get(event["restart_type"],
                                  {"pre_pad_s": 6.0, "post_search_s": 15.0})


def find_restart(event, activity, dead_before):
    """Find the first credible restart at or after the tag start."""
    tag_start = int(event["start_frame"])
    tag_end = int(event["end_frame"])
    half_range = event_half_range(event)
    half_start, half_end = half_range
    profile = _profile(event)
    restart_type = event["restart_type"]

    if restart_type == "KICKOFF_START":
        low = max(C.FIXED_DEAD_END + 1, half_start) if half_start == C.SECOND_HALF_START \
            else half_start
        high = min(half_end, tag_end + int(round(profile["post_search_s"] * C.FPS)))
    else:
        low = max(tag_start, half_start)
        high = min(half_end, tag_end + int(round(profile["post_search_s"] * C.FPS)))

    candidates = []
    for start in sorted(set(activity["combined_starts"]) | set(activity["strong_starts"])):
        if start < low or start > high:
            continue
        if restart_type != "KICKOFF_START" \
                and dead_before[start] < C.RESTART_MIN_STOPPAGE_FRAMES:
            continue
        candidates.append(start)

    if candidates:
        restart_frame = candidates[0]
        in_combined = restart_frame in activity["combined_start_set"]
        in_strong = restart_frame in activity["strong_start_set"]
        evidence = "combined+strong" if in_combined and in_strong else (
            "combined" if in_combined else "strong")
        fallback = False
    else:
        if restart_type == "KICKOFF_START":
            restart_frame = (C.FIXED_DEAD_END + 1 if half_start == C.SECOND_HALF_START
                             else C.FIRST_HALF_START)
        else:
            restart_frame = min(tag_end, half_end)
        evidence = "none"
        fallback = True

    restart_frame = int(min(max(restart_frame, half_start), half_end))
    return {
        "restart_type": restart_type,
        "tag_start": tag_start,
        "tag_end": tag_end,
        "search_low": int(low),
        "search_high": int(high),
        "candidate_count": len(candidates),
        "restart_frame": restart_frame,
        "delay_s": (restart_frame - tag_start) / C.FPS,
        "evidence": evidence,
        "fallback": fallback,
        "accepted": not fallback,
    }


def estimate_dead_start(event, restart_frame, active, last_active_at):
    """Find the start of the dead stretch that ends at the restart.

    The backward search is capped by a type-specific pad before the tag start, so a long missing
    ball gap from an earlier phase cannot stretch D across the whole gap.
    """
    tag_start = int(event["start_frame"])
    half_range = event_half_range(event)
    half_start, _ = half_range
    profile = _profile(event)
    pad_low = max(half_start, tag_start - int(round(profile["pre_pad_s"] * C.FPS)))
    search_high = int(restart_frame) - C.RESTART_MIN_STOPPAGE_FRAMES
    search_low = max(half_start, int(restart_frame) - C.RESTART_BACKTRACK_MAX_FRAMES)

    last_active = -1
    if search_high >= search_low:
        last_active = int(last_active_at[search_high])
        if last_active < search_low:
            last_active = -1

    if last_active >= 0:
        dead_start = max(pad_low, last_active + 1)
    else:
        dead_start = pad_low

    if dead_start >= restart_frame:
        dead_start = max(pad_low, restart_frame - C.MIN_D_RUN_FRAMES)
    dead_start = int(min(max(dead_start, half_start), restart_frame))

    return {
        "dead_start": dead_start,
        "dead_end": int(restart_frame) - 1,
        "pad_low": int(pad_low),
        "last_active_frame": int(last_active),
        "backtrack_used": bool(last_active >= 0 and dead_start == last_active + 1),
    }


def build_event_intervals(events, goals, restarts, links, signals, activity):
    dead_before = dead_before_counts(activity["active"])
    last_active_at = last_active_indices(activity["active"])

    restart_results = {}
    restart_rows = []
    for _, event in restarts.sort_values("start_frame").iterrows():
        result = find_restart(event, activity, dead_before)
        restart_results[int(event["event_id"])] = result
        dead = estimate_dead_start(event, result["restart_frame"],
                                   activity["active"], last_active_at)
        if event["restart_type"] == "KICKOFF_START":
            dead["dead_start"] = (C.FIRST_HALF_START
                                  if event_half_range(event)[0] == C.FIRST_HALF_START
                                  else C.FIXED_DEAD_START)
            dead["dead_end"] = result["restart_frame"] - 1
        restart_rows.append({
            "event_id": int(event["event_id"]),
            "restart_type": event["restart_type"],
            "team": event.get("Team"),
            "half": event.get("Half"),
            "tag_start": int(event["start_frame"]),
            "tag_end": int(event["end_frame"]),
            "restart_frame": result["restart_frame"],
            "delay_s": result["delay_s"],
            "accepted": result["accepted"],
            "fallback": result["fallback"],
            "evidence": result["evidence"],
            "candidate_count": result["candidate_count"],
            "dead_start": dead["dead_start"],
            "dead_end": dead["dead_end"],
            "dead_duration_s": (dead["dead_end"] - dead["dead_start"] + 1) / C.FPS
            if dead["dead_end"] >= dead["dead_start"] else 0.0,
            "last_active_frame": dead["last_active_frame"],
            "backtrack_used": dead["backtrack_used"],
        })

    interval_rows = []
    for _, event in restarts.sort_values("start_frame").iterrows():
        event_id = int(event["event_id"])
        result = restart_results[event_id]
        restart_type = event["restart_type"]
        half_range = event_half_range(event)

        if restart_type == "KICKOFF_AFTER_GOAL" and int(event.get("goal_event_id", -1)) >= 0:
            continue

        if restart_type == "KICKOFF_START":
            dead_start = C.FIRST_HALF_START if half_range[0] == C.FIRST_HALF_START \
                else C.FIXED_DEAD_START
            dead_end = result["restart_frame"] - 1
        else:
            dead = estimate_dead_start(event, result["restart_frame"],
                                       activity["active"], last_active_at)
            dead_start = dead["dead_start"]
            dead_end = dead["dead_end"]

        clipped = _clip_to_range(dead_start, dead_end, half_range)
        if clipped is None:
            continue
        start, end = clipped
        interval_rows.append({
            "start": start,
            "end": end,
            "source": "set_piece",
            "restart_type": restart_type,
            "event_id": event_id,
            "team": event.get("Team"),
            "restart_frame": result["restart_frame"],
            "evidence": result["evidence"],
            "fallback": result["fallback"],
            "duration_s": (end - start + 1) / C.FPS,
        })

    goal_rows = []
    for _, goal in goals.sort_values("start_frame").iterrows():
        goal_id = int(goal["event_id"])
        half_range = event_half_range(goal)
        link = links[links["goal_event_id"] == goal_id]
        kickoff_id = int(link.iloc[0]["kickoff_event_id"]) if len(link) else -1
        cluster = "goal"
        if kickoff_id >= 0 and kickoff_id in restart_results:
            dead_end = restart_results[kickoff_id]["restart_frame"] - 1
            cluster = "goal+kickoff"
        else:
            dead_end = int(goal["end_frame"]) - 1

        dead_start = int(goal["start_frame"]) - int(round(C.GOAL_PRE_PAD_S * C.FPS))
        dead_end = int(dead_end) + int(round(C.GOAL_END_PAD_S * C.FPS))
        clipped = _clip_to_range(dead_start, dead_end, half_range)
        if clipped is None:
            continue
        start, end = clipped
        goal_rows.append({
            "start": start,
            "end": end,
            "source": "goal",
            "restart_type": "GOAL",
            "event_id": goal_id,
            "team": goal.get("Team"),
            "kickoff_event_id": kickoff_id,
            "cluster": cluster,
            "restart_frame": dead_end + 1,
            "evidence": "goal",
            "fallback": kickoff_id < 0,
            "duration_s": (end - start + 1) / C.FPS,
        })

    restart_diagnostics = pd.DataFrame(restart_rows)
    goal_diagnostics = pd.DataFrame(goal_rows)
    return restart_results, restart_diagnostics, goal_diagnostics, interval_rows + goal_rows


def _rolling_mean(values, window, min_periods):
    return pd.Series(values).rolling(
        window, center=True, min_periods=min_periods).mean().to_numpy()


def detect_tracking_intervals(signals):
    graded = signals["graded"]
    player_speed = signals["player_speed"]
    rows = []
    intervals = []

    gap_mask = graded & (~signals["ball_valid"]) & (
        signals["ball_gap_len"] >= C.GAP_MIN_FRAMES)
    for start, end in runs_from_mask(gap_mask):
        length = end - start + 1
        if length < C.GAP_MIN_FRAMES:
            continue
        window = player_speed[start:end + 1]
        finite = window[np.isfinite(window)]
        if finite.size < max(3, int(0.25 * length)):
            continue
        mean_speed = float(finite.mean())
        if mean_speed >= C.GAP_MAX_MEAN_PLAYER_SPEED:
            continue
        intervals.append((start, end))
        rows.append({
            "start": start,
            "end": end,
            "source": "low_activity_gap",
            "duration_s": length / C.FPS,
            "mean_player_speed": mean_speed,
            "player_speed_count": int(finite.size),
            "evidence": "long missing ball + low player speed",
        })

    oop_mask = graded & signals["out_of_pitch"]
    for raw_start, raw_end in runs_from_mask(oop_mask):
        half = C.half_of_frame(raw_start)
        if not half:
            continue
        half_range = C.HALF_RANGES[half]
        raw_length = raw_end - raw_start + 1
        pad = C.OOP_PAD_FRAMES
        expanded = _clip_to_range(raw_start - pad, raw_end + pad, half_range)
        if expanded is None:
            continue
        start, end = expanded
        window = player_speed[start:end + 1]
        finite = window[np.isfinite(window)]
        mean_speed = float(finite.mean()) if finite.size else np.nan
        expanded_length = end - start + 1
        oop_fraction = raw_length / expanded_length
        long_enough = raw_length >= C.OOP_MIN_FRAMES
        very_long = raw_length >= 2 * C.OOP_MIN_FRAMES
        low_activity = finite.size > 0 and mean_speed < C.OOP_MAX_MEAN_PLAYER_SPEED
        high_fraction = oop_fraction >= C.OOP_MIN_FRACTION
        if not long_enough:
            continue
        if not (low_activity or very_long or high_fraction):
            continue
        intervals.append((start, end))
        rows.append({
            "start": start,
            "end": end,
            "source": "out_of_pitch",
            "duration_s": expanded_length / C.FPS,
            "mean_player_speed": mean_speed,
            "player_speed_count": int(finite.size),
            "out_of_pitch_fraction": oop_fraction,
            "evidence": "sustained out-of-pitch ball",
        })

    window = C.STATIONARY_WINDOW_FRAMES
    min_obs = max(1, int(0.8 * window))
    ball_for_stationary = np.where(signals["trusted"],
                                   signals["safe_ball_speed"], np.nan)
    ball_mean = _rolling_mean(ball_for_stationary, window, min_obs)
    player_mean = _rolling_mean(player_speed, window, min_obs)
    valid_fraction = _rolling_mean(signals["ball_valid"].astype(float), window, window)
    strong_fraction = _rolling_mean(signals["strong_ball"].astype(float), window, window)
    stationary_mask = (
        graded
        & (valid_fraction >= C.STATIONARY_MIN_VALID_FRACTION)
        & (ball_mean < C.STATIONARY_MAX_BALL_SPEED)
        & (player_mean < C.STATIONARY_MAX_PLAYER_SPEED)
        & (strong_fraction < 1e-9)
    )
    for start, end in merge_runs(runs_from_mask(stationary_mask),
                                 C.STATIONARY_MERGE_GAP_FRAMES):
        length = end - start + 1
        if length < C.STATIONARY_MIN_DURATION_FRAMES:
            continue
        window_ball = signals["safe_ball_speed"][start:end + 1]
        window_player = player_speed[start:end + 1]
        ball_finite = window_ball[np.isfinite(window_ball)]
        player_finite = window_player[np.isfinite(window_player)]
        intervals.append((start, end))
        rows.append({
            "start": start,
            "end": end,
            "source": "stationary_ball",
            "duration_s": length / C.FPS,
            "mean_player_speed": float(player_finite.mean()) if player_finite.size else np.nan,
            "player_speed_count": int(player_finite.size),
            "mean_ball_speed": float(ball_finite.mean()) if ball_finite.size else np.nan,
            "evidence": "long low-activity window with a slow visible ball",
        })

    return pd.DataFrame(rows), intervals


def compute_activity(signals):
    combined = signals["ball_alive"] & signals["player_active"]
    strong = signals["strong_ball"]
    active = combined | strong
    combined_runs = runs_from_mask(combined, C.RESTART_COMBINED_MIN_FRAMES)
    strong_runs = runs_from_mask(strong, C.RESTART_STRONG_BALL_MIN_FRAMES)
    return {
        "combined": combined,
        "strong": strong,
        "active": active,
        "combined_runs": combined_runs,
        "strong_runs": strong_runs,
        "combined_starts": [start for start, _ in combined_runs],
        "strong_starts": [start for start, _ in strong_runs],
        "combined_start_set": set(start for start, _ in combined_runs),
        "strong_start_set": set(start for start, _ in strong_runs),
    }


class C2InvariantError(RuntimeError):
    pass


def c2_seed_mask(signals):
    """Frames where the ball is visible, slow, and more than 25 m from the nearest opponent."""
    raw = (
        signals["ball_valid"]
        & (signals["opponent_distance"] > C.C2_SEED_MAX_OPPONENT_DISTANCE_M)
        & (np.nan_to_num(signals["ball_speed"], nan=99.0) < C.C2_SEED_MAX_BALL_SPEED)
    )
    return interval_mask(runs_from_mask(raw, C.C2_SEED_MIN_FRAMES),
                         n=len(raw))


def expand_c2_seed(seed_mask, labels, signals):
    """Grow C2 seeds only through low-activity N frames, at most 10 s in each direction."""
    n = len(labels)
    expanded = seed_mask.copy()
    for start, end in runs_from_mask(seed_mask):
        frame = start - 1
        steps = 0
        while frame >= 0 and steps < C.C2_EXPAND_MAX_FRAMES:
            if (labels[frame] != "N" or not signals["graded"][frame]
                    or not signals["ball_valid"][frame]
                    or not np.isfinite(signals["ball_speed"][frame])
                    or signals["ball_speed"][frame] >= C.C2_EXPAND_MAX_BALL_SPEED
                    or signals["strong_ball"][frame]
                    or not np.isfinite(signals["player_speed"][frame])
                    or signals["player_speed"][frame] >= C.C2_EXPAND_MAX_PLAYER_SPEED):
                break
            expanded[frame] = True
            frame -= 1
            steps += 1

        frame = end + 1
        steps = 0
        while frame < n and steps < C.C2_EXPAND_MAX_FRAMES:
            if (labels[frame] != "N" or not signals["graded"][frame]
                    or not signals["ball_valid"][frame]
                    or not np.isfinite(signals["ball_speed"][frame])
                    or signals["ball_speed"][frame] >= C.C2_EXPAND_MAX_BALL_SPEED
                    or signals["strong_ball"][frame]
                    or not np.isfinite(signals["player_speed"][frame])
                    or signals["player_speed"][frame] >= C.C2_EXPAND_MAX_PLAYER_SPEED):
                break
            expanded[frame] = True
            frame += 1
            steps += 1
    return expanded


def _assert_c2_does_not_touch_existing_d(base_dead, promoted):
    candidate = base_dead | promoted
    for start, end in runs_from_mask(candidate):
        if not promoted[start:end + 1].any():
            continue
        touches = bool(base_dead[start:end + 1].any())
        touches = touches or bool(start > 0 and base_dead[start - 1])
        touches = touches or bool(end + 1 < len(base_dead) and base_dead[end + 1])
        if touches:
            raise C2InvariantError(
                "C2 expansion would extend or merge existing D frames {}-{}".format(start, end))


def apply_c2_exp10(labels, signals):
    """Promote eligible N frames with C2_exp10; X and existing D labels are never changed."""
    labels = np.asarray(labels)
    if len(labels) != len(signals["graded"]):
        raise ValueError("labels and signals must have the same length")
    base_dead = labels == "D"
    seed = c2_seed_mask(signals)
    expanded = expand_c2_seed(seed, labels, signals)
    promoted = expanded & (labels == "N")
    _assert_c2_does_not_touch_existing_d(base_dead, promoted)

    out = labels.copy()
    out[promoted] = "D"
    if np.any(out[labels == "X"] != "X"):
        raise C2InvariantError("C2 changed an X frame")
    if np.any(out[base_dead] != "D"):
        raise C2InvariantError("C2 changed an existing D frame")

    candidate_runs = runs_from_mask(out == "D")
    original_runs = runs_from_mask(base_dead)
    interval_rows = []
    for start, end in candidate_runs:
        if not promoted[start:end + 1].any():
            continue
        overlap = [(low, high) for low, high in original_runs
                   if low <= end and start <= high]
        if not overlap:
            kind = "new"
        elif len(overlap) > 1:
            kind = "merged"
        else:
            low, high = overlap[0]
            kind = "extended" if start < low or end > high else "unchanged"
        interval_rows.append({
            "start": int(start),
            "end": int(end),
            "frames": int(end - start + 1),
            "promoted_frames": int(promoted[start:end + 1].sum()),
            "kind": kind,
        })

    halves = signals["half"][promoted]
    stats = {
        "config": {
            "seed_max_opponent_distance_m": C.C2_SEED_MAX_OPPONENT_DISTANCE_M,
            "seed_max_ball_speed_mps": C.C2_SEED_MAX_BALL_SPEED,
            "seed_min_duration_s": C.C2_SEED_MIN_DURATION_S,
            "expand_duration_s": C.C2_EXPAND_DURATION_S,
            "expand_max_ball_speed_mps": C.C2_EXPAND_MAX_BALL_SPEED,
            "expand_max_player_speed_mps": C.C2_EXPAND_MAX_PLAYER_SPEED,
        },
        "seed_frames": int(seed.sum()),
        "seed_runs": [[int(start), int(end)] for start, end in runs_from_mask(seed)],
        "promoted_N_to_D": int(promoted.sum()),
        "promoted_by_half": {
            "1": int((halves == 1).sum()),
            "2": int((halves == 2).sum()),
            "0": int((halves == 0).sum()),
        },
        "D_frames_before": int(base_dead.sum()),
        "D_frames_after": int((out == "D").sum()),
        "net_D_increase": int((out == "D").sum() - base_dead.sum()),
        "new_or_changed_intervals": len(interval_rows),
        "new_intervals": int(sum(row["kind"] == "new" for row in interval_rows)),
        "extended_intervals": int(sum(row["kind"] == "extended" for row in interval_rows)),
        "merged_intervals": int(sum(row["kind"] == "merged" for row in interval_rows)),
        "existing_D_runs_extended": 0,
        "existing_D_runs_merged": 0,
        "intervals": interval_rows,
        "guard_passed": True,
    }
    return out, stats


def build_label_array(signals, events, goals, restarts, links):
    activity = compute_activity(signals)
    restart_results, restart_diag, goal_diag, event_intervals = build_event_intervals(
        events, goals, restarts, links, signals, activity)
    tracking_diag, tracking_intervals = detect_tracking_intervals(signals)

    all_intervals = []
    for row in event_intervals:
        clipped = _clip_to_range(row["start"], row["end"],
                                 C.HALF_RANGES[C.half_of_frame(row["start"]) or 1])
        if clipped:
            all_intervals.append(clipped)
    for start, end in tracking_intervals:
        half = C.half_of_frame(start)
        if half:
            clipped = _clip_to_range(start, end, C.HALF_RANGES[half])
            if clipped:
                all_intervals.append(clipped)

    all_intervals = merge_runs(all_intervals, max_gap=0)

    labels = np.full(C.N_FRAMES, "X", dtype="<U1")
    labels[signals["graded"]] = "N"
    labels[C.FIXED_DEAD_START:C.FIXED_DEAD_END + 1] = "D"
    labels[interval_mask(all_intervals)] = "D"

    protected = np.zeros(C.N_FRAMES, dtype=bool)
    protected[C.FIXED_DEAD_START:C.FIXED_DEAD_END + 1] = True
    labels, smooth_stats = smoothing.smooth_labels(
        labels, signals, activity, protected)

    labels[C.FIXED_DEAD_START:C.FIXED_DEAD_END + 1] = "D"
    labels[~signals["graded"]] = "X"

    labels, c2_stats = apply_c2_exp10(labels, signals)

    final_intervals = runs_from_mask(labels == "D", 1)
    report = {
        "labels": labels,
        "activity": activity,
        "restart_results": restart_results,
        "restart_diagnostics": restart_diag,
        "goal_diagnostics": goal_diag,
        "tracking_diagnostics": tracking_diag,
        "event_intervals": event_intervals,
        "tracking_intervals": tracking_intervals,
        "final_intervals": final_intervals,
        "smoothing": smooth_stats,
        "c2": c2_stats,
    }
    return report
