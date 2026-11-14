# models/

* `possession_lgbm.txt`: LightGBM model, 200 rounds, 16 features computed from the P1 feature table (list in
  `training_report.json`). Predicts P(home has the ball) for a frame with a usable ball.
* `training_report.json`: training data, parameters, cross-game results, chosen penalties, feature importance,
  library versions.
* `cross_game_results.md`: the held-out comparison table.

**Training data: Metrica Sports public sample data, Sample Games 1 and 2 (tracking + event data), only.** Possession
labels are derived by an algorithm from the event data (`src/learned/labels.py`); 78,386 and 80,655 labelled frames.
Nothing from BAR-ATH was used: not its tracking, tagging, homography, nor its public or private ground truth.
No labels were produced by hand.

Retrain: `python -m src.learned.train --metrica-dir <sample-data>/data` (deterministic: same file every time).
`make_submission.py` uses the committed model file, so the submission does not depend on the Metrica download.
