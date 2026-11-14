# More Than a Hack 2027: football possession

This repository contains our solution for the More Than a Hack football possession challenge.
The task is to produce one label per tactical-video frame for the BAR–ATH match:

- `H`: Barcelona in possession
- `A`: Athletic Club in possession
- `D`: ball is not in play
- `X`: ungraded frame / no prediction

Our selected approach is a hand-written physics evidence model decoded with Viterbi, combined with
a rule-based dead-ball mask. The default Viterbi switch penalty is **15**. The state machine,
nearest-player baseline, the Vidal-Codina et al. possession-method adaptation, and trained
LightGBM estimators remain available for reproducible comparisons, but are not the selected
submission method.

The original challenge materials are in [README_CHALLENGE.md](README_CHALLENGE.md). See
[TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md) for the submission methodology and
[THIRD_PARTY.md](THIRD_PARTY.md) for code and data provenance.

## Selected pipeline

The end-to-end command runs three components:

1. **Tracking cleanup and feature generation**
   ([src/data_preprocessing/](src/data_preprocessing/)):
   converts tracking coordinates to metres, removes isolated implausible ball jumps, interpolates
   only short ball gaps, and computes ball/player proximity and speed features.
2. **Dead-ball detection** ([src/Dead_Ball/](src/Dead_Ball/), the `dead_ball` component): creates
   a frame-level mask
   using Smart Tagging set-piece/goal events as hints, then checks tracking activity around likely
   restarts. Additional rules identify stoppages from long ball gaps with low player activity,
   out-of-pitch ball positions, and stationary-ball intervals. A conservative slow/isolated-ball
   rule adds eligible intervals without extending or merging existing dead-ball runs.
3. **Possession estimation** ([src/possession_estimators/](src/possession_estimators/)):
   computes signed hand-written evidence for which team controls a nearby, relatively slow ball.
   The Viterbi decoder finds the whole-match sequence that balances this evidence against a cost of
   15 for each possession switch. Dead-mask frames are `D` and split decoding into independent
   live-play segments.

The dead-ball mask is not itself a possession prediction. Its intermediate labels are `D` (detected
stoppage), `N` (live-play candidate), and `X` (ungraded frame); possession estimation converts live-play frames to
`H` or `A`. This separation lets the possession decoder restart after a detected stoppage rather
than carrying the previous team's possession through it.

### Run the components separately

Run these commands from the repository root, in order, after unpacking the challenge bundle into
`data/`. The feature table is created first; dead-ball detection consumes that table and the raw
tracking; possession estimation consumes both generated outputs.

```bash
# 1. Clean tracking and write outputs/features.parquet
python3 -m src.data_preprocessing.build_features --data-dir ./data

# 2. Write outputs/p3_dead_mask.csv and outputs/dead_mask.csv
python3 -m src.Dead_Ball.build_p3 \
  --features outputs/features.parquet \
  --raw-atd data/tactical/BAR-ATH_tactical_atd.txt \
  --out outputs/p3_dead_mask.csv \
  --dead-out outputs/dead_mask.csv

# 3. Write possession labels using the selected estimator and penalty
python3 -m src.possession_estimators.run_estimators \
  --estimators viterbi --vit_penalty 15 \
  --dead outputs/dead_mask.csv \
  --out outputs/possession_estimates
```

If the challenge bundle is elsewhere, set its root for both commands. For example:

```bash
export MTAH_DATA_DIR=/path/to/data
python3 -m src.data_preprocessing.build_features --data-dir "$MTAH_DATA_DIR"
python3 -m src.Dead_Ball.build_p3 \
  --features outputs/features.parquet \
  --raw-atd "$MTAH_DATA_DIR/tactical/BAR-ATH_tactical_atd.txt" \
  --out outputs/p3_dead_mask.csv \
  --dead-out outputs/dead_mask.csv
```

`MTAH_DATA_DIR` is needed by the dead-ball component to load Smart Tagging events. If preprocessing
reports that it wrote `features.csv.gz` instead of Parquet, use that path for the `--features`
argument. The full pipeline command below configures the data directory automatically.

### Why Viterbi with penalty 15

Possession is a persistent state, not an independent nearest-player decision on every frame.
Evidence is stronger when one team's nearest player is clearly closer to a slow ball and weaker
when the ball is distant, interpolated, or in flight. Viterbi uses evidence from both before and
after a possible handover, avoiding the flicker of frame-by-frame assignment and the delay of a
strictly causal rule.

We use a fixed switch penalty of **15** as a compromise between frequent switches at lower
penalties and longer merged spells at higher penalties. The cross-game Metrica comparisons below
show that penalties 10–20 are a relatively flat region, with small differences on approximate,
event-derived labels. Penalty 15 is the documented submission setting, not a claim that it is
universally optimal. It was not selected using the BAR–ATH public-window score.

## Other implemented estimators

All estimators consume the same feature table and write per-frame label CSVs:

| Estimator | Summary |
|---|---|
| `viterbi` | Selected hand-written evidence plus global sequence decoding; default penalty 15. |
| `state_machine` | Causal leaky evidence accumulator with hysteresis; switches after sustained opposing evidence. |
| `majority` | Nearest-team baseline with a centred majority filter. |
| `vidal_codina` | Team-level adaptation of Vidal-Codina et al.'s possession-control method; not a full reproduction. |
| `learned` | Viterbi on LightGBM evidence trained on Metrica event-derived proxy labels. |
| `combined` | Viterbi on the average of hand-written and learned evidence. |

Run one or several estimators after `data_preprocessing` and `dead_ball` have produced their files:

```bash
python3 -m src.possession_estimators.run_estimators \
  --estimators viterbi state_machine majority vidal_codina learned combined \
  --dead outputs/dead_mask.csv \
  --out outputs/possession_estimates
```

For a parameter-stability check that does not need labels:

```bash
python3 -m src.possession_estimators.run_estimators \
  --estimators viterbi --dead outputs/dead_mask.csv --sweep
```

The CLI also supports individual parameter overrides such as `--vit_penalty 15`. For more
implementation details, assumptions, and limitations, see
[src/possession_estimators/README.md](src/possession_estimators/README.md).

## Training the experimental learned estimator

The selected hand-written Viterbi method does **not** require training. The optional `learned` and
`combined` estimators use the included LightGBM training pipeline. To regenerate the model and
cross-game report, obtain Metrica Sports' public sample data separately and run:

```bash
python3 -m src.learned.train --metrica-dir sample-data/data
```

Training labels are generated from Metrica event data; they are approximate proxy labels, not
manually corrected possession ground truth. The training report uses cross-game evaluation
(training on one sample game and testing on the other). These two games are too few to establish
general performance, and the learned model is not used by the selected submission. Metrica data is
not included in this repository; see [models/README.md](models/README.md) and
[models/cross_game_results.md](models/cross_game_results.md).

## Experiments and score interpretation

The following hand-written Viterbi results were measured against the Metrica event-derived proxy
labels. `Masked` adds seeded synthetic ball dropouts as a stress test. SpellF1 is averaged over
the four game/condition combinations:

| Penalty | G1 clean | G1 masked | G2 clean | G2 masked | Mean spellF1 |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.776 | 0.709 | 0.761 | 0.692 | 0.734 |
| 10 | 0.814 | 0.724 | 0.807 | 0.705 | 0.763 |
| **15 (selected)** | 0.791 | 0.700 | 0.818 | 0.716 | 0.756 |
| 20 | 0.781 | 0.707 | 0.813 | 0.707 | 0.752 |

These are small-sample proxy-label experiments, not challenge scores or proof that penalty 15 is
best. The public BAR–ATH window contains only 17 reference possessions, so its score has high
uncertainty. Use it only to check that the pipeline and output format work; do not tune the method
or penalty on it. The private full-match labels are not available in this repository.

## Install, test, and evaluate

Use Python 3.12 and install the pinned dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m src.data_preprocessing.selftest
python3 -m src.possession_estimators.selftest
python3 -m src.learned.selftest
```

To evaluate generated predictions when the corresponding ground truth is available to you:

```bash
python3 evaluation/run_evaluation.py \
  --gt public \
  --results outputs/possession_estimates \
  --out outputs/viterbi_score
```

The evaluation script reports frame metrics, possession-spell F1 at several IoU thresholds,
transition metrics, and diagnostics. The `public` ground truth covers only the short public
window; it is not suitable for ranking or tuning configurations. For output-format checks, run the
submission validator directly:

```bash
python3 evaluation/validate_submission.py submission/OffsideAI.csv
```

## Generate the competition submission

Obtain the gated competition bundle and unpack it under `data/` as described in
[DATA.md](DATA.md). From the repository root, run:

```bash
python3 make_submission.py --team OffsideAI
```

This uses the selected Viterbi estimator with penalty 15 and the dead-ball mask by default.
To set the penalty explicitly, use `--vit-penalty 15`. To store the gated data elsewhere, add
`--data-dir /path/to/data`. The command runs preprocessing, dead-ball detection, possession
estimation, and validation, then writes `submission/OffsideAI.csv` and `submission/OffsideAI.md`, a copy of the technical
document. Fill in the team, member, and repository placeholders in that document and validate the
exact CSV you intend to upload. Do not commit or redistribute the gated match data.

## Repository map

| Path | Purpose |
|---|---|
| [make_submission.py](make_submission.py) | End-to-end preprocessing, prediction, and validation |
| [src/data_preprocessing/](src/data_preprocessing/) | Tracking cleanup and per-frame features |
| [src/Dead_Ball/](src/Dead_Ball/) | `dead_ball` detector and mask |
| [src/possession_estimators/](src/possession_estimators/) | Evidence, estimators, and sequence decoders |
| [src/learned/](src/learned/) | Metrica loader, proxy labels, LightGBM training, and evaluation |
| [evaluation/](evaluation/) | Challenge validator and possession metrics |
| [models/](models/) | Trained LightGBM model and experiment reports |
| [TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md) | Technical submission document; fill in its placeholders |
