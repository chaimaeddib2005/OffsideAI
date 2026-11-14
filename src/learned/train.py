"""Train the learned evidence model on Metrica Sample Games 1 and 2 and validate it across games.

    git clone https://github.com/metrica-sports/sample-data
    python -m src.learned.train --metrica-dir sample-data/data

What it does
  1. Loads both games and expresses them in the data_preprocessing feature-table schema (metrica.py).
  2. Builds proxy possession labels from the event data (labels.py).
  3. Cross-game validation: train on one game, test on the other, in both directions and in two
     conditions (ball as tracked, which already resembles BAR-ATH's 59% availability / extra synthetic dropouts as a stress test). Prints one table, compares the learned
     evidence against the hand-written evidence and the majority baseline, and picks the Viterbi
     penalty from a FLAT region of that table.
  4. Trains the final model on both games and writes models/possession_lgbm.txt and
     models/training_report.json.

Nothing from BAR-ATH is read here: not its tracking, not its tagging, not the public ground truth.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from . import metrica
from .features import FEATURES, model_features
from .heldout import evaluate_game
from .labels import possession_labels
from . import model as M

PENALTIES = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
TRAIN_DROPOUT_SEEDS = (100, 300)      # two synthetic-dropout copies of each training game
TEST_DROPOUT_SEED = 777               # a different dropout pattern for the held-out game


def build_all(data_dir):
    raw = {g: metrica.load_raw(data_dir, g) for g in (1, 2)}
    labels = {g: possession_labels(raw[g]["events"], raw[g]["frame"]) for g in raw}
    tables = {}
    for g in raw:
        tables[(g, "clean")] = metrica.build_table(raw[g])
        for s in TRAIN_DROPOUT_SEEDS:
            tables[(g, "masked%d" % s)] = metrica.build_table(raw[g], ball_dropout_seed=s + g)
        tables[(g, "test_masked")] = metrica.build_table(raw[g], ball_dropout_seed=TEST_DROPOUT_SEED + g)
    return raw, labels, tables


def training_set(games, tables, labels):
    feats, ys, ms = [], [], []
    for g in games:
        for variant in ["clean"] + ["masked%d" % s for s in TRAIN_DROPOUT_SEEDS]:
            t = tables[(g, variant)]
            feats.append(model_features(t))
            ys.append(labels[g])
            ms.append(t["ball_valid"].to_numpy(dtype=bool))
    return feats, ys, ms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrica-dir", required=True, help="the sample-data/data folder")
    ap.add_argument("--out", default=str(M.MODEL_PATH))
    args = ap.parse_args(argv)
    t0 = time.time()

    raw, labels, tables = build_all(args.metrica_dir)
    n_lab = {g: int((~np.isnan(labels[g])).sum()) for g in labels}
    print("labelled frames (proxy): game 1 = {:,}  game 2 = {:,}".format(n_lab[1], n_lab[2]))

    # ---- cross-game validation ----------------------------------------------------------------
    results = {}        # (test_game, condition) -> {method: metrics}
    for test in (1, 2):
        train_g = 2 if test == 1 else 1
        booster = M.train(*training_set([train_g], tables, labels))
        for cond, key in (("clean", "clean"), ("masked", "test_masked")):
            results[(test, cond)] = evaluate_game(tables[(test, key)], labels[test], booster, PENALTIES)

    methods = list(next(iter(results.values())).keys())
    lines = ["| method | " + " | ".join("G%d %s spellF1 / acc / n spells" % k for k in results) + " |",
             "|---|" + "---|" * len(results)]
    for m in methods:
        cells = []
        for k in results:
            r = results[k][m]
            cells.append("{:.3f} / {:.3f} / {}".format(r["spell_f1"], r["frame_acc"], r["n_pred_spells"]))
        lines.append("| {} | {} |".format(m, " | ".join(cells)))
    gt_n = ", ".join("G%d %s: %d" % (k[0], k[1], results[k][methods[0]]["n_gt_spells"]) for k in results)
    table_md = "\n".join(lines) + "\n\nProxy-GT spells (the scorer's count for the same frames): " + gt_n
    print("\nHeld-out results (train on the OTHER game; proxy labels; relative comparison only)\n")
    print(table_md)

    # ---- choose the penalties from a flat region ----------------------------------------------
    # Pre-declared rule: among the grid values whose mean cross-game spellF1 is within 0.02 of the
    # best, take the geometric middle (nearest grid value). A sharp optimum is an artefact.
    def pick(prefix):
        mean_f1 = {pen: float(np.mean([results[k]["%s%g" % (prefix, pen)]["spell_f1"] for k in results]))
                   for pen in PENALTIES}
        best = max(mean_f1.values())
        flat = [pen for pen in PENALTIES if mean_f1[pen] >= best - 0.02]
        mid = float(np.exp(np.mean(np.log(flat))))
        chosen = min(PENALTIES, key=lambda x: abs(np.log(x) - np.log(mid)))
        return mean_f1, flat, chosen

    f1_learned, flat_learned, chosen_learned = pick("viterbi, learned evidence, penalty ")
    f1_combined, flat_combined, chosen_combined = pick("viterbi, hand+learned evidence, penalty ")
    hand = float(np.mean([results[k]["viterbi, hand-written evidence"]["spell_f1"] for k in results]))
    major = float(np.mean([results[k]["majority (published-baseline style)"]["spell_f1"] for k in results]))
    print("\nmean spellF1 over the 4 held-out conditions: majority {:.3f} | hand-written Viterbi {:.3f} | "
          "learned {:.3f} | hand+learned {:.3f}".format(major, hand, max(f1_learned.values()),
                                                       max(f1_combined.values())))
    print("learned  flat region {} -> penalty {:g}".format(flat_learned, chosen_learned))
    print("combined flat region {} -> penalty {:g}".format(flat_combined, chosen_combined))
    mean_f1, flat, chosen = f1_combined, flat_combined, chosen_combined

    # ---- final model on both games ------------------------------------------------------------
    booster = M.train(*training_set([1, 2], tables, labels))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out))
    imp = dict(sorted(zip(FEATURES, booster.feature_importance("gain").round(1).tolist()),
                      key=lambda kv: -kv[1]))
    report = {
        "trained_on": ["Metrica Sample_Game_1", "Metrica Sample_Game_2"],
        "never_used": "BAR-ATH tracking, tagging, homography, public or private ground truth",
        "labels": "event-derived proxy (src/learned/labels.py): determined frames only",
        "labelled_frames": n_lab,
        "features": FEATURES, "lightgbm_params": M.PARAMS, "rounds": M.N_ROUNDS,
        "augmentation": "plus 2 copies of each game with extra synthetic ball dropouts (log-normal gaps; ~37% of frames usable, harsher than BAR-ATH). Metrica as tracked already has the ball on ~60% of graded frames (BAR-ATH: 59.3%)",
        "cross_game_results": {"%d_%s" % k: v for k, v in results.items()},
        "mean_spellF1_by_penalty": {"learned": {str(k): v for k, v in f1_learned.items()},
                                    "hand+learned": {str(k): v for k, v in f1_combined.items()}},
        "mean_spellF1_majority": major, "mean_spellF1_hand_viterbi": hand,
        "flat_region": {"learned": flat_learned, "hand+learned": flat_combined},
        "recommended_penalty": {"learned": chosen_learned, "combined": chosen_combined},
        "feature_importance_gain": imp,
        "env": {"python": platform.python_version(), "numpy": np.__version__,
                "lightgbm": __import__("lightgbm").__version__},
        "seconds": round(time.time() - t0, 1),
    }
    (out.parent / "training_report.json").write_text(json.dumps(report, indent=2))
    (out.parent / "cross_game_results.md").write_text(table_md + "\n")
    print("\nmodel -> {}\nreport -> {}\nfeature importance (gain): {}".format(
        out, out.parent / "training_report.json", imp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
