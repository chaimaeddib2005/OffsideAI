"""Conservative smoothing for the per-frame dead-ball mask.

Short D islands are removed because a stoppage shorter than MIN_D_RUN_S is rarely a real dead
ball. Nearby D islands are merged only when the gap between them has no strong activity evidence,
so a genuine restart is not swallowed by smoothing.
"""

import numpy as np

from . import config as C


def _runs(mask):
    runs = []
    start = None
    for index, value in enumerate(mask):
        if value:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _d_runs(labels):
    return _runs(labels == "D")


def _same_graded_range(start, end):
    for low, high in C.GRADED_RANGES:
        if low <= start and end <= high:
            return True
    return False


def _remove_short(labels, protected, min_frames):
    out = labels.copy()
    removed = 0
    for start, end in _d_runs(out):
        if end - start + 1 >= min_frames:
            continue
        if protected[start:end + 1].any():
            continue
        out[start:end + 1] = "N"
        removed += 1
    return out, removed


def smooth_labels(labels, signals, activity, protected):
    out = labels.copy()
    active = activity["combined"] | activity["strong"]

    out, removed = _remove_short(out, protected, C.MIN_D_RUN_FRAMES)

    merged = 0
    for _ in range(20):
        runs = _d_runs(out)
        changed = False
        for index in range(len(runs) - 1):
            _, end = runs[index]
            next_start, _ = runs[index + 1]
            gap_start = end + 1
            gap_end = next_start - 1
            if gap_start > gap_end:
                continue
            gap_length = gap_end - gap_start + 1
            if gap_length > C.MERGE_GAP_FRAMES:
                continue
            if not _same_graded_range(gap_start, gap_end):
                continue
            gap_active = active[gap_start:gap_end + 1]
            if activity["strong"][gap_start:gap_end + 1].any():
                continue
            blocking_runs = [end - start + 1 for start, end in _runs(gap_active)
                             if end - start + 1 >= C.RESTART_COMBINED_MIN_FRAMES]
            if blocking_runs:
                continue
            out[gap_start:gap_end + 1] = "D"
            merged += 1
            changed = True
            break
        if not changed:
            break

    out, removed_after = _remove_short(out, protected, C.MIN_D_RUN_FRAMES)
    removed += removed_after

    out[protected] = "D"
    out[~signals["graded"]] = "X"

    d_runs = _d_runs(out)
    lengths = np.array([end - start + 1 for start, end in d_runs], dtype=np.int32)
    stats = {
        "removed_short_runs": int(removed),
        "merged_gaps": int(merged),
        "d_frames": int((out == "D").sum()),
        "n_runs": int(len(d_runs)),
        "run_length_min": int(lengths.min()) if lengths.size else 0,
        "run_length_median": float(np.median(lengths)) if lengths.size else 0.0,
        "run_length_max": int(lengths.max()) if lengths.size else 0,
        "run_length_mean_s": float(lengths.mean() / C.FPS) if lengths.size else 0.0,
    }
    return out, stats
