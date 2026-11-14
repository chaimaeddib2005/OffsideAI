"""Run the possession estimators on outputs/features.parquet and write submission CSVs.

    python -m src.possession_estimators.run_estimators                       # default estimators
    python -m src.possession_estimators.run_estimators --estimators viterbi
    python -m src.possession_estimators.run_estimators --vit_penalty 30 --min_run_seconds 1.5
    python -m src.possession_estimators.run_estimators --dead outputs/dead_mask.parquet
    python -m src.possession_estimators.run_estimators --sweep               # robustness, no labels needed

Then score:   python evaluation/run_evaluation.py --results outputs/possession_estimates
Validate:     python evaluation/validate_submission.py outputs/possession_estimates/viterbi.csv
"""

import argparse
import dataclasses
import os
import sys
import time

import numpy as np

from . import combined, config, io_utils, learned, majority, state_machine, vidal_codina, viterbi
from .config import Params
from .evidence import compute_evidence

ESTIMATORS = {"majority": majority, "state_machine": state_machine, "viterbi": viterbi,
              "vidal_codina": vidal_codina, "learned": learned, "combined": combined}

# main knob per estimator, and the values tried by --sweep
SWEEPS = {"majority": ("majority_window", [25, 50, 75, 125]),
          "state_machine": ("sm_switch", [3.0, 4.5, 6.0, 9.0, 12.0, 18.0]),
          "viterbi": ("vit_penalty", [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]),
          "vidal_codina": ("vc_pz_radius", [1.0, 1.5, 2.0, 2.5]),
          "learned": ("learned_penalty", [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]),
          "combined": ("combined_penalty", [5.0, 10.0, 20.0, 40.0, 80.0, 160.0])}


def run_one(name, table, p, dead, evidence):
    return ESTIMATORS[name].estimate(table, p, dead=dead, evidence=evidence)


def add_param_args(parser):
    for f in dataclasses.fields(Params):
        parser.add_argument("--" + f.name, type=type(f.default), default=None,
                            help="default {}".format(f.default))


def params_from_args(args):
    p = Params()
    for f in dataclasses.fields(Params):
        v = getattr(args, f.name)
        if v is not None:
            setattr(p, f.name, v)
    return p


def dead_mask(table, p, path):
    mask = io_utils.load_dead_mask(path, len(table)) if path else np.zeros(len(table), dtype=bool)
    if p.dead_gap_seconds > 0:
        mask |= table["ball_gap_len"].to_numpy() >= p.dead_gap_seconds * config.FPS
    return mask if mask.any() else None


def summary(name, codes, graded):
    runs, mean_s = io_utils.possession_runs(codes, graded)
    g = codes[graded]
    share = float((g == config.HOME).sum()) / max(int((g != config.DEAD).sum()), 1)
    return "{:<28} possessions {:>4}  mean {:>5.1f} s  home share {:>5.1f}%  D {:>5.1f}%".format(
        name, runs, mean_s, 100 * share, 100 * float((g == config.DEAD).mean()))


def sweep(table, p, dead, evidence, names):
    graded = table["graded"].to_numpy()
    print("\nSweep: how stable is the output when the main knob moves? (no ground truth used)")
    print("'differs' = share of graded frames that change versus the default setting.\n")
    for name in names:
        knob, values = SWEEPS[name]
        default = run_one(name, table, p, dead, evidence)
        print("{}  (knob: {}, default {})".format(name, knob, getattr(p, knob)))
        for v in values:
            q = dataclasses.replace(p, **{knob: type(getattr(p, knob))(v)})
            codes = run_one(name, table, q, dead, evidence)
            differs = 100 * float((codes[graded] != default[graded]).mean())
            print("  {:>8}  {}  differs {:>5.2f}%".format(v, summary("", codes, graded).strip(), differs))
        print()
    print("Pick a value from a FLAT region (possession count and 'differs' both change slowly), not "
          "from a sharp optimum. Real football: roughly 15 to 20 s per possession.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", default=None)
    parser.add_argument("--out", default=str(config.ESTIMATES_DIR))
    parser.add_argument("--estimators", nargs="+", default=list(ESTIMATORS),
                        choices=list(ESTIMATORS))
    parser.add_argument("--dead", default=None,
                        help="dead_ball mask: frame,label CSV with D/N/X")
    parser.add_argument("--outside", default="D", choices=["D", "X"],
                        help="label for frames outside the two halves (never scored). D matches the baseline")
    parser.add_argument("--sweep", action="store_true", help="print a parameter-stability table and exit")
    add_param_args(parser)
    args = parser.parse_args(argv)

    p = params_from_args(args)
    table = io_utils.load_features(args.features)
    graded = table["graded"].to_numpy()
    dead = dead_mask(table, p, args.dead)
    n_dead = 0 if dead is None else int((dead & graded).sum())
    print("{} frames, {} graded, dead-ball frames inside graded play: {} ({:.1f}%)".format(
        len(table), int(graded.sum()), n_dead, 100.0 * n_dead / max(int(graded.sum()), 1)))
    evidence = compute_evidence(table, p)
    print("evidence: ball usable on {:.1f}% of graded frames, mean |e| {:.3f}\n".format(
        100 * float((evidence[graded] != 0).mean()), float(np.abs(evidence[graded]).mean())))

    if args.sweep:
        sweep(table, p, dead, evidence, args.estimators)
        return 0

    for name in args.estimators:
        t0 = time.time()
        codes = run_one(name, table, p, dead, evidence)
        path = os.path.join(args.out, name + ".csv")
        io_utils.write_labels(path, io_utils.to_letters(codes, graded, args.outside))
        print(summary(name, codes, graded) + "   ({:.1f}s)".format(time.time() - t0))
    print("\nwritten to {}".format(os.path.relpath(args.out, str(config.ROOT))))
    print("score:  python evaluation/run_evaluation.py --results {}".format(
        os.path.relpath(args.out, str(config.ROOT))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
