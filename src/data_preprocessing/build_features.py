"""THE one command. Raw tracking in, one clean per-frame table out.

    python -m src.data_preprocessing.build_features
    python -m src.data_preprocessing.build_features --max-gap 8 --no-cache --skip-qa

Writes   outputs/features.parquet   (or features.csv.gz if pyarrow is not installed)
         outputs/features_meta.json (parameters + counts, for reproducibility)
Then runs the QA checks and exits non-zero if one of them FAILS.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import clean, config, features, load_atd as loader
from games import half_video_blocks


def half_column(n_frames, blocks):
    half = np.zeros(n_frames, dtype=np.int8)               # 0 = not graded (pre/half/post-match)
    for h, first, last in blocks:
        half[first:last + 1] = h
    return half


def save_table(table, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table.to_parquet(out_path, index=False)
        return out_path
    except ImportError:
        fallback = out_path.with_name("features.csv.gz")
        print("  pyarrow not installed -> writing {} instead (pip install pyarrow for parquet)"
              .format(fallback.name))
        table.to_csv(fallback, index=False)
        return fallback


def build(game=config.GAME, data_dir=None, max_gap=config.MAX_GAP_FRAMES,
          max_speed=config.MAX_BALL_SPEED, use_cache=True):
    """Run the whole pipeline and return (table, team, params). Writes nothing."""
    t0 = time.time()
    print("[1/4] loading tracking")
    players, ball, team = loader.load_atd(game, data_dir, use_cache=use_cache)
    n = len(ball)
    blocks = half_video_blocks(game)

    print("[2/4] converting to metres")
    players_m, ball_raw_m = clean.to_metres(players, ball)

    print("[3/4] cleaning the ball (despike > {} m/s, fill gaps <= {} frames)".format(max_speed, max_gap))
    despiked, is_spike = clean.despike_ball(ball_raw_m, blocks, max_speed=max_speed)
    ball_clean_m, is_interp = clean.fill_short_gaps(despiked, blocks, max_gap=max_gap)

    print("[4/4] building features")
    table = features.build_table(players_m, ball_raw_m, ball_clean_m, is_spike, is_interp,
                                 team, half_column(n, blocks))
    params = {"game": game, "max_gap_frames": max_gap, "max_ball_speed": max_speed,
              "near_radius_m": config.NEAR_RADIUS, "speed_smooth_frames": config.SPEED_SMOOTH,
              "pitch_m": list(config.PITCH), "n_rows": int(len(table)),
              "half_blocks": [list(b) for b in blocks], "seconds": round(time.time() - t0, 1)}
    return table, team, params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None, help="override MTAH_DATA_DIR / ./data")
    parser.add_argument("--out", default=str(config.FEATURES_PATH))
    parser.add_argument("--max-gap", type=int, default=config.MAX_GAP_FRAMES)
    parser.add_argument("--max-speed", type=float, default=config.MAX_BALL_SPEED)
    parser.add_argument("--no-cache", action="store_true", help="re-parse the raw ATD")
    parser.add_argument("--skip-qa", action="store_true")
    args = parser.parse_args(argv)

    table, team, params = build(data_dir=args.data_dir, max_gap=args.max_gap,
                                max_speed=args.max_speed, use_cache=not args.no_cache)
    path = save_table(table, args.out)
    graded = table["graded"]
    params.update({"ball_observed_pct_all": round(100 * table["ball_observed"].mean(), 2),
                   "ball_observed_pct_graded": round(100 * table.loc[graded, "ball_observed"].mean(), 2),
                   "output": str(path)})
    config.META_PATH.write_text(json.dumps(params, indent=2))
    print("\nwrote {} rows x {} columns -> {}".format(len(table), table.shape[1], path))
    print("ball observed: {:.1f}% of all frames, {:.1f}% of graded frames".format(
        params["ball_observed_pct_all"], params["ball_observed_pct_graded"]))

    if args.skip_qa:
        return 0
    from . import qa
    return qa.run(table)


if __name__ == "__main__":
    sys.exit(main())
