# What was changed relative to the two source repositories

Sources: `chaimaeddib2005/OffsideAI` (the base repository), and
`saideepojha/football-possession-estimator` (learned model idea + proxy labels).

## Added
- `src/learned/` : Metrica loader that emits the `data_preprocessing` feature-table schema, proxy labels, features computed from
  that table only, LightGBM training with cross-game validation, held-out evaluation with the repo's own scorer,
  and a no-data self-test.
- `src/possession_estimators/learned.py`, `combined.py`: two experimental estimators (Viterbi on learned evidence;
  Viterbi on the average of hand-written and learned evidence).
- `models/` : the trained model, `training_report.json`, `cross_game_results.md`.
- `make_submission.py` : data_preprocessing -> dead_ball -> possession estimation -> validate, in one command.
- `TECHNICAL_DOCUMENT.md` (the third required deliverable, with `<<placeholders>>` to fill), this file, `THIRD_PARTY.md`.

## Fixed
- The revised dead-ball detector in `src/Dead_Ball/` replaces the earlier `src/p3_deadball/`
  implementation. Submission generation now uses the revised detector and writes its mask to the
  standard `outputs/dead_mask.csv` path consumed by P2.
- The submission default is hand-written Viterbi with switch penalty 15; the learned and combined estimators remain
  available for experiments.
- Removed `more_than_a_hack_just_code/`, a stale older copy of `starter/` and `src/` (it lacked `vidal_codina.py`,
  `tuning_checks.py` and `dead_report.py`, and its `config`/`rules`/`run_estimators` had drifted from `src/`).
- Dead-ball README said it reads `outputs/features.csv.gz`; the code reads `features.parquet` with a `.csv.gz` fallback.
- `requirements.txt` now pins versions and includes lightgbm and matplotlib.
- Organiser README moved to `README_CHALLENGE.md`; `README.md` now describes this solution.
- `.gitignore`: `submission/*` added.

## Not changed
Tracking-cleanup logic; the organisers' `evaluation/`.
