"""ONE command: the provided match data in, a validated <team>.csv out.

    python make_submission.py --team MyTeam
    python make_submission.py --team MyTeam --data-dir /path/to/mtah2027      # if not in ./data

Stages (each one is also runnable on its own, see src/*/README.md):
  P1  src.data_preprocessing.build_features   raw ATD tracking  -> outputs/features.parquet
  P3  src.Dead_Ball.build_p3                  features + Smart Tagging + player speeds (new detector)
                                                                 -> outputs/dead_mask.csv
  P2  src.possession_estimators.run_estimators  features + dead mask -> H / A / D per frame
  then: copy to submission/<team>.csv, run the organisers' validator on that exact file, and print
        the public-5-minute check (a plumbing check ONLY: never tune on it, see RULES.md).

The default estimator is hand-written Viterbi with switch penalty 10. Everything is deterministic:
the same data and the same repository give the same file (RULES.md, rule 3).
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def banner(text):
    print("\n" + "=" * 100 + "\n" + text + "\n" + "=" * 100, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", required=True, help="team name; the file is submission/<team>.csv")
    ap.add_argument("--data-dir", default=None, help="folder holding the unpacked data bundle (default ./data)")
    ap.add_argument("--estimator", default="viterbi",
                    choices=["combined", "learned", "viterbi", "state_machine", "paper_rules", "majority"])
    ap.add_argument("--vit-penalty", type=float, default=None,
                    help="Viterbi switch penalty (only applies with --estimator viterbi)")
    ap.add_argument("--no-dead-mask", action="store_true", help="skip P3 (no D frames except outside the halves)")
    ap.add_argument("--reuse-features", action="store_true", help="do not rebuild outputs/features.parquet")
    args = ap.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9_\-]+", args.team):
        ap.error("--team may only contain letters, digits, '_' and '-' (it becomes the file name)")
    if args.vit_penalty is not None and args.estimator != "viterbi":
        ap.error("--vit-penalty can only be used with --estimator viterbi")

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else ROOT / "data"
    os.environ["MTAH_DATA_DIR"] = str(data_dir)      # read by games.py and starter/load_*.py
    sys.path.insert(0, str(ROOT))
    atd = data_dir / "tactical" / "BAR-ATH_tactical_atd.txt"
    if not atd.is_file():
        sys.exit("data not found: {}\nDownload the bundle (see DATA.md), unpack it into ./data or pass "
                 "--data-dir. The data is licensed for the competition and is never part of this "
                 "repository.".format(atd))

    from src.data_preprocessing import build_features
    from src.Dead_Ball import build_p3
    from src.possession_estimators import run_estimators

    out = ROOT / "outputs"
    features = out / "features.parquet"
    if not args.reuse_features or not (features.is_file() or (out / "features.csv.gz").is_file()):
        banner("P1  cleaning the tracking -> per-frame feature table")
        if build_features.main(["--data-dir", str(data_dir)]) != 0:
            sys.exit("P1 failed (its QA checks did not pass); see the messages above")

    dead_args = []
    if not args.no_dead_mask:
        banner("P3  dead-ball mask")
        build_p3.main(["--raw-atd", str(atd)])
        dead_args = ["--dead", str(out / "dead_mask.csv")]

    banner("P2  possession labels with the '{}' estimator".format(args.estimator))
    est_dir = out / "possession_estimates"
    estimator_args = ["--estimators", args.estimator, "--out", str(est_dir)] + dead_args
    vit_penalty = args.vit_penalty
    if args.estimator == "viterbi" and vit_penalty is None:
        vit_penalty = 10.0
    if vit_penalty is not None:
        estimator_args += ["--vit_penalty", str(vit_penalty)]
    if run_estimators.main(estimator_args) != 0:
        sys.exit("P2 failed")

    sub_dir = ROOT / "submission"
    sub_dir.mkdir(exist_ok=True)
    final = sub_dir / (args.team + ".csv")
    shutil.copyfile(est_dir / (args.estimator + ".csv"), final)

    banner("validating the exact file that will be uploaded")
    rc = subprocess.call([sys.executable, str(ROOT / "evaluation" / "validate_submission.py"), str(final)])
    if rc != 0:
        sys.exit("VALIDATION FAILED: do not upload this file")

    banner("plumbing check on the public 5 minutes (NOT a tuning set; 17 possessions cannot rank anything)")
    check_dir = out / "public_check"
    shutil.rmtree(check_dir, ignore_errors=True)
    check_dir.mkdir(parents=True)
    shutil.copyfile(final, check_dir / final.name)
    subprocess.call([sys.executable, str(ROOT / "evaluation" / "run_evaluation.py"), "--results", str(check_dir)])

    doc = ROOT / "TECHNICAL_DOCUMENT.md"
    if doc.is_file():
        shutil.copyfile(doc, sub_dir / (args.team + ".md"))

    banner("DONE")
    print("possession file : {}".format(final))
    print("sha256          : {}".format(sha256(final)))
    print("technical doc   : {} (fill in every <<...>> before uploading)".format(sub_dir / (args.team + ".md")))
    print("Upload the .csv and the document with your PUBLIC repository URL. Re-run validate_submission.py on "
          "the exact file you upload; there is no leaderboard feedback.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
