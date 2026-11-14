"""Output validation, reproducibility checks, and diagnostic reports."""

import hashlib
import json

import numpy as np
import pandas as pd

from . import config as C


def runs_from_labels(labels, label):
    runs = []
    start = None
    for index, value in enumerate(labels):
        if value == label:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(labels) - 1))
    return runs


def label_stats(labels):
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    stats = {str(key): int(value) for key, value in zip(unique, counts)}
    graded = int((labels != "X").sum())
    d_runs = runs_from_labels(labels, "D")
    lengths = np.array([end - start + 1 for start, end in d_runs], dtype=np.int32)
    return {
        "n_frames": int(total),
        "counts": stats,
        "share": {key: value / total for key, value in stats.items()},
        "graded_frames": graded,
        "d_runs": len(d_runs),
        "d_run_length_min": int(lengths.min()) if lengths.size else 0,
        "d_run_length_median": float(np.median(lengths)) if lengths.size else 0.0,
        "d_run_length_max": int(lengths.max()) if lengths.size else 0,
        "d_run_length_mean_s": float(lengths.mean() / C.FPS) if lengths.size else 0.0,
    }


def validate_labels(labels):
    errors = []
    if len(labels) != C.N_FRAMES:
        errors.append("expected {} frames, got {}".format(C.N_FRAMES, len(labels)))
    if labels.dtype.kind not in ("U", "S"):
        errors.append("labels must be a string array")
    allowed = set(C.LABELS)
    observed = set(np.unique(labels).tolist())
    if not observed.issubset(allowed):
        errors.append("unexpected labels: {}".format(sorted(observed - allowed)))

    graded = C.graded_mask()
    if np.any(labels[~graded] != "X"):
        errors.append("non-graded frames must all be X")
    if np.any(~np.isin(labels[graded], ["D", "N"])):
        errors.append("graded frames must be D or N")

    fixed = labels[C.FIXED_DEAD_START:C.FIXED_DEAD_END + 1]
    if not np.all(fixed == "D"):
        errors.append("fixed second-half dead block contains a non-D frame")
    if len(fixed) != C.FIXED_DEAD_FRAMES:
        errors.append("fixed block length is not {}".format(C.FIXED_DEAD_FRAMES))

    stats = label_stats(labels)
    if stats["counts"].get("D", 0) == 0:
        errors.append("no D frames")
    if stats["counts"].get("N", 0) == 0:
        errors.append("no N frames")

    return {
        "ok": not errors,
        "errors": errors,
        "stats": stats,
    }


def write_output(labels, path=C.OUTPUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "frame": np.arange(C.N_FRAMES, dtype=np.int32),
        "label": labels,
    })
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def write_dead_mask(labels, path=C.DEAD_MASK_PATH):
    """frame,is_dead file for possession_estimators (--dead). is_dead is True only where label == D."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "frame": np.arange(C.N_FRAMES, dtype=np.int32),
        "is_dead": np.asarray(labels) == "D",
    })
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def validate_output_file(path=C.OUTPUT_PATH):
    df = pd.read_csv(path)
    errors = []
    if list(df.columns) != ["frame", "label"]:
        errors.append("header must be exactly frame,label; got {}".format(list(df.columns)))
    if len(df) != C.N_FRAMES:
        errors.append("expected {} rows, got {}".format(C.N_FRAMES, len(df)))
    if len(df) and not np.array_equal(df["frame"].to_numpy(), np.arange(C.N_FRAMES)):
        errors.append("frame column is not 0..{} in order".format(C.N_FRAMES - 1))
    if len(df) and not set(df["label"].unique()).issubset(set(C.LABELS)):
        errors.append("unexpected labels in file")
    if len(df) == C.N_FRAMES:
        labels = df["label"].to_numpy(dtype="<U1")
        check = validate_labels(labels)
        errors.extend(check["errors"])
    return {"ok": not errors, "errors": errors}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def labels_sha256(labels):
    return hashlib.sha256(labels.tobytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def final_run_table(labels):
    rows = []
    for start, end in runs_from_labels(labels, "D"):
        rows.append({
            "start_frame": int(start),
            "end_frame": int(end),
            "duration_s": (end - start + 1) / C.FPS,
            "start_s": start / C.FPS,
            "end_s": end / C.FPS,
        })
    return pd.DataFrame(rows)


def write_tables(report, out_dir=C.DIAGNOSTICS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    report["restart_diagnostics"].to_csv(out_dir / "restart_diagnostics.csv", index=False)
    report["goal_diagnostics"].to_csv(out_dir / "goal_diagnostics.csv", index=False)
    report["tracking_diagnostics"].to_csv(out_dir / "tracking_diagnostics.csv", index=False)
    final_run_table(report["labels"]).to_csv(out_dir / "final_d_runs.csv", index=False)


def _shade(ax, labels, low, high):
    for start, end in runs_from_labels(labels[low:high + 1], "D"):
        ax.axvspan(low + start, low + end, color="#d62728", alpha=0.16, linewidth=0)


def _plot_window(signals, labels, low, high, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    low = max(0, int(low))
    high = min(C.N_FRAMES - 1, int(high))
    frame = np.arange(low, high + 1)
    ball_speed = signals["safe_ball_speed"][low:high + 1]
    player_speed = signals["player_speed"][low:high + 1]
    raw_ball_speed = signals["ball_speed"][low:high + 1]
    ball_x = signals["ball_x"][low:high + 1]
    ball_y = signals["ball_y"][low:high + 1]
    d_mask = (labels[low:high + 1] == "D").astype(float)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    ax = axes[0]
    ax.plot(frame, raw_ball_speed, color="#7f7f7f", linewidth=0.7, label="ball speed (raw)")
    ax.plot(frame, ball_speed, color="#1f77b4", linewidth=1.0, label="trusted ball speed")
    ax.plot(frame, player_speed, color="#2ca02c", linewidth=1.0, label="mean player speed")
    ax.axhline(C.BALL_ALIVE_SPEED, color="#1f77b4", linestyle=":", linewidth=0.8)
    ax.axhline(C.PLAYER_ACTIVE_SPEED, color="#2ca02c", linestyle=":", linewidth=0.8)
    ax.axhline(C.BALL_STRONG_SPEED, color="#ff7f0e", linestyle=":", linewidth=0.8)
    ax.set_ylabel("m/s")
    ax.set_ylim(0, 45)
    ax.legend(loc="upper right", fontsize=8)
    _shade(ax, labels, low, high)

    ax = axes[1]
    ax.plot(frame, ball_x, color="#1f77b4", linewidth=0.9, label="ball x")
    ax.plot(frame, ball_y, color="#ff7f0e", linewidth=0.9, label="ball y")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(C.PITCH_LENGTH, color="black", linewidth=0.6)
    ax.axhline(C.PITCH_WIDTH, color="black", linewidth=0.6)
    ax.set_ylabel("metres")
    ax.legend(loc="upper right", fontsize=8)
    _shade(ax, labels, low, high)

    ax = axes[2]
    ax.fill_between(frame, 0, d_mask, step="post", color="#d62728", alpha=0.55)
    ax.set_ylabel("D mask")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["N", "D"])
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("video frame")
    _shade(ax, labels, low, high)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_overview(signals, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = np.arange(C.N_FRAMES)
    minutes = frame / C.FPS / 60.0
    d_mask = (labels == "D").astype(float)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(minutes, signals["safe_ball_speed"], color="#1f77b4", linewidth=0.5)
    axes[0].plot(minutes, signals["player_speed"], color="#2ca02c", linewidth=0.5)
    axes[0].set_ylabel("m/s")
    axes[0].set_ylim(0, 40)
    axes[0].fill_between(minutes, 0, 40, where=d_mask.astype(bool),
                         color="#d62728", alpha=0.16, linewidth=0)

    axes[1].plot(minutes, signals["ball_x"], color="#1f77b4", linewidth=0.5)
    axes[1].plot(minutes, signals["ball_y"], color="#ff7f0e", linewidth=0.5)
    axes[1].set_ylabel("ball x/y (m)")
    axes[1].fill_between(minutes, 0, 1, where=d_mask.astype(bool),
                         transform=axes[1].get_xaxis_transform(),
                         color="#d62728", alpha=0.08, linewidth=0)

    axes[2].fill_between(minutes, 0, d_mask, step="post", color="#d62728", alpha=0.55)
    axes[2].set_ylabel("D mask")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["N", "D"])
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_xlabel("tactical video minutes")
    fig.suptitle("Dead-ball mask overview")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_plots(report, out_dir=C.DIAGNOSTICS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    signals = report["signals"]
    labels = report["labels"]
    restart_diag = report["restart_diagnostics"]
    goal_diag = report["goal_diagnostics"]
    selected = []

    for restart_type in C.RESTART_TYPES:
        subset = restart_diag[restart_diag["restart_type"] == restart_type]
        if len(subset) == 0:
            continue
        accepted = subset[subset["accepted"]]
        row = (accepted if len(accepted) else subset).iloc[0]
        selected.append(("restart", restart_type, row))

    half2 = restart_diag[(restart_diag["restart_type"] == "KICKOFF_START")
                         & (restart_diag["half"] == "2nd Half")]
    if len(half2):
        selected.append(("restart", "KICKOFF_START_2ND", half2.iloc[0]))

    if len(goal_diag):
        selected.append(("goal", "GOAL", goal_diag.sort_values("duration_s",
                                                               ascending=False).iloc[0]))

    tracking_diag = report["tracking_diagnostics"]
    if len(tracking_diag):
        for source in ("stationary_ball", "out_of_pitch", "low_activity_gap"):
            subset = tracking_diag[tracking_diag["source"] == source]
            if len(subset) == 0:
                continue
            if source == "low_activity_gap":
                median = subset["duration_s"].median()
                row = subset.iloc[(subset["duration_s"] - median).abs().argmin()]
            else:
                row = subset.sort_values("duration_s", ascending=False).iloc[0]
            selected.append(("tracking", source, row))

    seen = set()
    for kind, name, row in selected:
        key = (kind, name, int(row.get("event_id", -1)), int(row.get("start", -1)))
        if key in seen:
            continue
        seen.add(key)
        if kind == "goal":
            low = int(row["start"]) - 5 * C.FPS
            high = int(row["end"]) + 5 * C.FPS
            title = "goal event {}  frames {}-{}".format(
                int(row["event_id"]), int(row["start"]), int(row["end"]))
        elif kind == "tracking":
            low = int(row["start"]) - 5 * C.FPS
            high = int(row["end"]) + 5 * C.FPS
            title = "{}  frames {}-{}  evidence {}".format(
                name, int(row["start"]), int(row["end"]), row["evidence"])
        else:
            low = min(int(row["tag_start"]), int(row["dead_start"])) - 5 * C.FPS
            high = max(int(row["tag_end"]), int(row["restart_frame"])) + 5 * C.FPS
            title = "{} event {}  tag {}-{}  restart {}  evidence {}".format(
                name, int(row["event_id"]), int(row["tag_start"]), int(row["tag_end"]),
                int(row["restart_frame"]), row["evidence"])
        path = out_dir / "event_{}_{:04d}.png".format(
            name.lower().replace(" ", "_"), int(row.get("event_id", 0)))
        _plot_window(signals, labels, low, high, title, path)

    _plot_overview(signals, labels, out_dir / "overview.png")
    return {
        "n_event_plots": len(seen),
        "overview": str(out_dir / "overview.png"),
    }


def write_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
