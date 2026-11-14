# More Than a Hack 2027: football possession

This repository produces one possession label per frame for the BAR–ATH tactical match:

- `H`: Barcelona in possession
- `A`: Athletic Club in possession
- `D`: ball is not in play
- `X`: abstention (not used by the selected estimator)

The selected approach is the **hand-written physics evidence plus Viterbi decoder, with switch
penalty 10**. The learned LightGBM model and the competition's nearest-player baseline were tested
as comparisons; neither is part of the selected submission. The original challenge instructions
are in [README_CHALLENGE.md](README_CHALLENGE.md), provenance in [THIRD_PARTY.md](THIRD_PARTY.md),
and implementation changes in [CHANGES.md](CHANGES.md).

## Selected method and why

Possession is treated as a persistent state. For each frame, hand-written evidence estimates which
team is near a slow, tracked ball; evidence is reduced when the ball is moving quickly, is
interpolated, or is far from players. When the ball is missing or evidence is weak, the decoder
tends to retain its current state. Viterbi chooses the H/A sequence that balances agreement with
that evidence against a fixed cost for changing teams. The dead-ball pipeline supplies `D` frames.

The selected switch penalty is **10**. It was chosen from cross-game Metrica comparisons: hand-only
Viterbi with penalty 10 averaged about 0.763 spellF1 over the four reported conditions, versus
about 0.752 with penalty 20. The advantage is modest and the Metrica possession labels are
event-derived proxies, not perfect ground truth. This is a conservative, reproducible choice, not
a guarantee of the best BAR–ATH score.

### Can the penalty be selected automatically for each match?

Not reliably from tracking alone: without labels for the match, a system cannot know which penalty
would maximize spellF1. It is possible to define a pre-set adaptive rule using observable signal
quality (for example ball visibility or evidence confidence), or to train a separate penalty
selection model on many labelled matches. Either approach needs validation on multiple held-out
matches; otherwise it risks fitting a heuristic to only two sample games. For this submission we
use a fixed penalty of 10 rather than an unvalidated per-match selector.

## Methods tested

### Competition nearest-player baselines

The baseline assigns the ball to the nearest tracked player, then optionally applies a centered
majority vote. Its four variants were tested on the Metrica games using the same feature pipeline
and scorer. The labels are the Metrica event-derived proxy labels.

| Method | Game 1 spellF1 / accuracy / predicted spells | Game 2 spellF1 / accuracy / predicted spells |
|---|---:|---:|
| Raw nearest player | 0.185 / 0.755 / 1,521 | 0.172 / 0.738 / 1,705 |
| Majority smoothing, ±25 frames | 0.578 / 0.811 / 525 | 0.586 / 0.800 / 532 |
| Majority smoothing, ±50 frames | 0.666 / 0.833 / 416 | 0.716 / 0.822 / 418 |
| Majority smoothing, ±125 frames | 0.622 / 0.833 / 392 | 0.671 / 0.819 / 386 |

### Hand-written Viterbi

These are held-out Metrica results using the same hand-written evidence and changing only the
Viterbi switch penalty. `Clean` is the provided tracking; `masked` adds synthetic ball dropouts as
a stress test. Each cell is spellF1 / frame accuracy / predicted possession count. The proxy
ground-truth counts are 302 for Game 1 and 307 for Game 2.

| Penalty | G1 clean | G1 masked | G2 clean | G2 masked | Mean spellF1 |
|---:|---:|---:|---:|---:|---:|
| 10 (selected) | 0.814 / 0.886 / 314 | 0.724 / 0.809 / 323 | 0.807 / 0.848 / 334 | 0.705 / 0.814 / 336 | **0.763** |
| 15 | 0.791 / 0.884 / 305 | 0.700 / 0.808 / 320 | 0.818 / 0.860 / 321 | 0.716 / 0.813 / 323 | 0.756 |
| 20 | 0.781 / 0.882 / 303 | 0.707 / 0.813 / 315 | 0.813 / 0.859 / 318 | 0.707 / 0.810 / 323 | 0.752 |

On the original (clean) Metrica tracking, penalty 10 scores 0.814 and 0.807, above the best
nearest-player baseline (0.666 and 0.716). On the extra-dropout stress tests its mean is 0.715.
These results favor a persistent-state decoder over frame-by-frame nearest-player voting, but are
not a prediction of the BAR–ATH challenge score.

### Learned evidence and combined estimator

A LightGBM model was also trained on event-derived labels from Metrica Sample Games 1 and 2. Its
raw frame predictions had high frame accuracy but flickered into too many short possessions. Adding
the learned evidence to hand-written evidence improved the two-game cross-validation average only
slightly (about 0.764 combined versus 0.752 hand-written Viterbi in the repository's original
penalty-selection report). The event-derived labels differ from the competition's
corrected-tracking ownership labels, so the final submission omits the learned component. Details
are in [models/cross_game_results.md](models/cross_game_results.md) and
[models/training_report.json](models/training_report.json).

## Public-window check

The selected hand-written Viterbi method with penalty 10 was reported as scoring approximately
**0.44 spellF1** on the public five-minute BAR–ATH window. This window contains only 17 reference
possessions, so the score has high variance and is not a reliable estimate of full-match
performance. It is included for transparency, not as the basis for choosing the penalty.

Challenge rules say the public window is for checking pipeline correctness, not for tuning
parameters or selecting a method. Other reported public-window penalty results should not be used
to justify changing the selected configuration. The official full-match ground truth is private,
so no full-match spellF1 is available in this repository.

## Installation and tests

The pinned environment was tested with Python 3.12:

```bash
python3 -m pip install -r requirements.txt
python3 -m src.data_preprocessing.selftest
python3 -m src.possession_estimators.selftest
python3 -m src.learned.selftest
```

The self-tests do not require the gated BAR–ATH data or the Metrica sample data.

## Generate the selected submission

Obtain the gated competition bundle and unpack it under `data/` as described in
[DATA.md](DATA.md). Then run from the repository root:

```bash
python3 make_submission.py --team <TeamName> --estimator viterbi --vit-penalty 10
```

To keep the data elsewhere, add `--data-dir /path/to/data`. The command runs preprocessing,
dead-ball detection, the selected estimator and submission validation. It writes
`submission/<TeamName>.csv`; inspect its summary and validate the exact file before upload. Do not
commit or redistribute the gated match data.

The model's recorded outputs can also be regenerated/evaluated on Metrica Sample Games 1 and 2
(downloaded separately):

```bash
python3 -m src.learned.train --metrica-dir sample-data/data
```

That command includes training because it also regenerates the learned model and the cross-game
report. To reproduce the selected hand-written Viterbi evaluation without training, run this from
the repository root:

```bash
python3 - <<'PY'
from src.learned import metrica
from src.learned.labels import possession_labels
from src.learned.heldout import score
from src.possession_estimators import viterbi
from src.possession_estimators.config import Params
from src.possession_estimators.evidence import compute_evidence

params = Params(vit_penalty=10)
for game in (1, 2):
    raw = metrica.load_raw("sample-data/data", game)
    labels = possession_labels(raw["events"], raw["frame"])
    for condition, seed in (("clean", None), ("masked", 777 + game)):
        table = metrica.build_table(raw, ball_dropout_seed=seed)
        codes = viterbi.estimate(
            table, params, dead=None, evidence=compute_evidence(table, params)
        )
        result = score(codes, labels, table["half"].to_numpy())
        print(
            f"G{game} {condition}: spellF1={result['spell_f1']:.3f}, "
            f"accuracy={result['frame_acc']:.3f}, "
            f"predicted spells={result['n_pred_spells']}"
        )
PY
```

The Metrica labels are approximate event-derived proxies. Neither this test nor its scores are
equivalent to evaluation against the challenge's corrected-tracking labels.

## Repository map

| Path | Purpose |
|---|---|
| [make_submission.py](make_submission.py) | End-to-end preprocessing, prediction and validation |
| [src/data_preprocessing/](src/data_preprocessing/) | Tracking cleanup and per-frame features (P1) |
| [src/Dead_Ball/](src/Dead_Ball/) | Selected dead-ball detector and mask (P3) |
| [src/possession_estimators/](src/possession_estimators/) | Baselines, evidence and sequence decoders (P2) |
| [src/learned/](src/learned/) | Metrica loader, proxy labels, LightGBM and held-out comparisons |
| [evaluation/](evaluation/) | Challenge validator and possession metrics |
| [models/](models/) | Learned model and experiment reports |
| [TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md) | Technical submission document; fill in its placeholders |
