"""Build  dead-ball mask.

Run from the repository root:

    python -m src.Dead_Ball.build_p3

Outputs:
  outputs/p3_dead_mask.csv              frame,label for all 199,319 tactical frames
  outputs/dead_mask.csv                 frame,is_dead  (input of possession_estimators --dead)
  outputs/p3_diagnostics/*              run report, event tables, diagnostic plots
"""

import argparse
from pathlib import Path

import numpy as np

from . import config as C
from . import load_ball, load_tagging, rules, validation


def _restart_summary(diag):
    if len(diag) == 0:
        return {}
    rows = {}
    for restart_type, group in diag.groupby("restart_type"):
        rows[restart_type] = {
            "n": int(len(group)),
            "accepted": int(group["accepted"].sum()),
            "fallback": int(group["fallback"].sum()),
            "delay_s_median": float(group["delay_s"].median()),
            "delay_s_min": float(group["delay_s"].min()),
            "delay_s_max": float(group["delay_s"].max()),
        }
    return rows


def _goal_summary(goal_diag, links):
    if len(goal_diag) == 0:
        return {}
    return {
        "n_goals": int(len(goal_diag)),
        "n_linked_kickoffs": int((links["kickoff_event_id"] >= 0).sum()),
        "d_frames_from_goals": int(sum(
            int(row["end"]) - int(row["start"]) + 1 for _, row in goal_diag.iterrows())),
        "durations_s": [float(value) for value in goal_diag["duration_s"].tolist()],
    }


def _tracking_summary(tracking_diag):
    if len(tracking_diag) == 0:
        return {"n_intervals": 0}
    rows = {}
    for source, group in tracking_diag.groupby("source"):
        rows[source] = {
            "n": int(len(group)),
            "frames": int(sum(int(row["end"]) - int(row["start"]) + 1
                              for _, row in group.iterrows())),
            "duration_s_median": float(group["duration_s"].median()),
            "duration_s_max": float(group["duration_s"].max()),
        }
    return rows


def build(features_path=C.FEATURES_PATH, raw_atd_path=C.ATD_RAW_PATH,
          output_path=C.OUTPUT_PATH, diagnostics=True, repro_check=True,
          dead_mask_path=C.DEAD_MASK_PATH):
    C.check_match_constants()
    time_base = load_tagging.time_base_check()

    features = load_ball.read_features(features_path)
    player_speeds = load_ball.load_player_speeds(raw_atd_path)
    signals = load_ball.build_signals(features, player_speeds)

    events = load_tagging.load_events()
    restarts = load_tagging.load_restarts(events)
    goals = load_tagging.load_goals(events)
    links = load_tagging.link_goals_to_kickoffs(goals, restarts)
    restarts = load_tagging.attach_goal_links(restarts, goals, links)

    report = rules.build_label_array(signals, events, goals, restarts, links)
    labels = report["labels"]

    if repro_check:
        report_second = rules.build_label_array(signals, events, goals, restarts, links)
        if not np.array_equal(labels, report_second["labels"]):
            raise RuntimeError("dead-ball rule engine is not deterministic")
        labels_hash = validation.labels_sha256(labels)
    else:
        labels_hash = validation.labels_sha256(labels)

    label_check = validation.validate_labels(labels)
    if not label_check["ok"]:
        raise RuntimeError("label validation failed: {}".format(label_check["errors"]))

    report["signals"] = signals
    if diagnostics:
        validation.write_tables(report)
        plots = validation.make_plots(report)
    else:
        plots = {}

    output_path = validation.write_output(labels, output_path)
    validation.write_dead_mask(labels, dead_mask_path)
    file_check = validation.validate_output_file(output_path)
    if not file_check["ok"]:
        raise RuntimeError("output validation failed: {}".format(file_check["errors"]))

    restart_diag = report["restart_diagnostics"]
    goal_diag = report["goal_diagnostics"]
    tracking_diag = report["tracking_diagnostics"]
    final_stats = label_check["stats"]

    payload = {
        "config": {
            "fps": C.FPS,
            "n_frames": C.N_FRAMES,
            "pitch": [C.PITCH_LENGTH, C.PITCH_WIDTH],
            "graded_ranges": [list(pair) for pair in C.GRADED_RANGES],
            "fixed_dead_block": [C.FIXED_DEAD_START, C.FIXED_DEAD_END,
                                 C.FIXED_DEAD_FRAMES],
            "thresholds": {
                "ball_trust_max_speed": C.BALL_TRUST_MAX_SPEED,
                "pitch_tolerance_m": C.PITCH_TOLERANCE_M,
                "ball_alive_speed": C.BALL_ALIVE_SPEED,
                "ball_strong_speed": C.BALL_STRONG_SPEED,
                "player_active_speed": C.PLAYER_ACTIVE_SPEED,
                "player_min_obs": C.PLAYER_MIN_OBS,
                "restart_combined_min_frames": C.RESTART_COMBINED_MIN_FRAMES,
                "restart_strong_ball_min_frames": C.RESTART_STRONG_BALL_MIN_FRAMES,
                "restart_min_stoppage_s": C.RESTART_MIN_STOPPAGE_S,
                "max_backward_search_s": C.MAX_BACKWARD_SEARCH_S,
                "min_d_run_s": C.MIN_D_RUN_S,
                "merge_gap_s": C.MERGE_GAP_S,
                "gap_min_s": C.GAP_MIN_S,
                "gap_max_mean_player_speed": C.GAP_MAX_MEAN_PLAYER_SPEED,
                "oop_min_s": C.OOP_MIN_S,
                "oop_pad_s": C.OOP_PAD_S,
                "oop_max_mean_player_speed": C.OOP_MAX_MEAN_PLAYER_SPEED,
                "stationary_window_s": C.STATIONARY_WINDOW_S,
                "stationary_max_ball_speed": C.STATIONARY_MAX_BALL_SPEED,
                "stationary_max_player_speed": C.STATIONARY_MAX_PLAYER_SPEED,
                "stationary_min_duration_s": C.STATIONARY_MIN_DURATION_S,
                "c2_seed_max_opponent_distance_m": C.C2_SEED_MAX_OPPONENT_DISTANCE_M,
                "c2_seed_max_ball_speed": C.C2_SEED_MAX_BALL_SPEED,
                "c2_seed_min_duration_s": C.C2_SEED_MIN_DURATION_S,
                "c2_expand_duration_s": C.C2_EXPAND_DURATION_S,
                "c2_expand_max_ball_speed": C.C2_EXPAND_MAX_BALL_SPEED,
                "c2_expand_max_player_speed": C.C2_EXPAND_MAX_PLAYER_SPEED,
            },
        },
        "smart_tagging": {
            "n_events": int(len(events)),
            "n_set_pieces": int(events["is_set_piece"].sum()),
            "n_goals": int(events["is_goal"].sum()),
            "n_restarts": int(len(restarts)),
            "time_base": time_base,
            "counts": load_tagging.count_table(events).to_dict(orient="records"),
        },
        "restarts": _restart_summary(restart_diag),
        "goals": _goal_summary(goal_diag, links),
        "tracking": _tracking_summary(tracking_diag),
        "final": {
            "label_stats": final_stats,
            "smoothing": report["smoothing"],
            "c2": report["c2"],
            "n_final_d_runs": len(report["final_intervals"]),
        },
        "validation": {
            "labels": label_check,
            "output_file": file_check,
        },
        "reproducibility": {
            "deterministic_rebuild": bool(repro_check),
            "labels_sha256": labels_hash,
            "output_sha256": validation.file_sha256(output_path),
        },
        "diagnostics": plots,
        "output": str(output_path),
    }

    if diagnostics:
        validation.write_json(payload, C.DIAGNOSTICS_DIR / "report.json")

    return payload


def _print_summary(payload):
    stats = payload["final"]["label_stats"]
    counts = stats["counts"]
    share = stats["share"]
    print("frames: {}  D {:,} ({:.1f}%)  N {:,} ({:.1f}%)  X {:,} ({:.1f}%)".format(
        stats["n_frames"], counts.get("D", 0), 100 * share.get("D", 0),
        counts.get("N", 0), 100 * share.get("N", 0),
        counts.get("X", 0), 100 * share.get("X", 0)))
    print("D runs: {}  median {:.1f}s  max {:.1f}s".format(
        stats["d_runs"], stats["d_run_length_median"] / C.FPS,
        stats["d_run_length_max"] / C.FPS))
    c2 = payload["final"]["c2"]
    print("C2: {} N->D  new intervals {}".format(
        c2["promoted_N_to_D"], c2["new_or_changed_intervals"]))
    print("restarts: {}  goals: {}".format(
        payload["smart_tagging"]["n_restarts"],
        payload["smart_tagging"]["n_goals"]))
    print("time base agrees: {}".format(
        payload["smart_tagging"]["time_base"].get("agrees")))
    print("validation: {}".format(
        "OK" if payload["validation"]["labels"]["ok"]
        and payload["validation"]["output_file"]["ok"] else "FAILED"))
    print("output: {}".format(payload["output"]))
    print("sha256: {}".format(payload["reproducibility"]["output_sha256"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=C.FEATURES_PATH)
    parser.add_argument("--raw-atd", type=Path, default=C.ATD_RAW_PATH)
    parser.add_argument("--out", type=Path, default=C.OUTPUT_PATH)
    parser.add_argument("--dead-out", type=Path, default=C.DEAD_MASK_PATH)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--no-repro-check", action="store_true")
    args = parser.parse_args(argv)

    payload = build(
        features_path=args.features,
        raw_atd_path=args.raw_atd,
        output_path=args.out,
        dead_mask_path=args.dead_out,
        diagnostics=not args.no_diagnostics,
        repro_check=not args.no_repro_check,
    )
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
