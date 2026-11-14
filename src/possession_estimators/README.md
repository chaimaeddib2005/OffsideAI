# possession_estimators

Turns `outputs/features.parquet` (built by `data_preprocessing`) into per-frame possession labels
(`H` / `A` / `D`) as submission CSVs. The estimators share the same output format and evaluation
path. The selected submission method is hand-written evidence decoded by Viterbi, with the `dead_ball`
dead-ball mask and a switch penalty of 15.

## Install and run

Put this folder at `<repo>/src/possession_estimators/`. Run from the repo root:

    python -m src.possession_estimators.selftest            # no data needed
    python -m src.possession_estimators.run_estimators      # all estimators -> outputs/possession_estimates/
    python evaluation/run_evaluation.py --results outputs/possession_estimates
    python evaluation/validate_submission.py outputs/possession_estimates/viterbi.csv

Useful options:

    --estimators viterbi state_machine majority vidal_codina learned combined
    --vit_penalty 30 --sm_switch 12 --min_run_seconds 1.5     (every parameter in config.py is a flag)
    --sweep                       how stable is the output when the main knob moves (no labels used)
    --dead outputs/dead_mask.csv       dead_ball mask (frame,is_dead; frame,label with D/N/X also works)
    --dead_gap_seconds 6          treat ball gaps this long as dead (OFF by default, see below)
    --outside X                   write X instead of D for frames outside the two halves (never scored)

## The idea

The baseline decides every frame on its own, so it flickers (131 possessions against 17 real ones).
Here possession is a **state that persists** until the other team clearly takes the ball, and the
tracking is only noisy **evidence** about it. Two layers:

1. **Evidence** (`evidence.py`): each frame becomes one number `e` in [-1, 1]. Sign = which team
   looks closer, size = how much to trust it.
2. **Decision** (`majority.py`, `state_machine.py`, `viterbi.py`): turn the evidence into a
   sequence of H/A with inertia.

### Evidence layer

`e = tanh(margin / 2 m) x proximity x not_flying`

| factor | meaning | why |
|---|---|---|
| `tanh(margin/2)` | margin = nearest-away minus nearest-home distance | a 3 m lead is near certain, 0.2 m is a coin flip |
| `proximity = exp(-(d_nearest/3 m)^2)` | is anyone actually near the ball | a player 8 m away is not in control |
| `not_flying` | 1 below 6 m/s, falling to 0 at 18 m/s | a ball in flight (pass, clearance) belongs to nobody yet, and the nearest player is often an opponent in the lane |
| x 0.5 on interpolated frames | | filled positions are guesses |
| `e = 0` with no ball | | no information: the state simply persists |

### The estimators

| name | what it does | strengths | weaknesses |
|---|---|---|---|
| `majority` | nearest team, then +-50 frame majority vote (the published baseline, from the cleaned table) | the floor; always printed next to the others | no confidence, no flight logic, no inertia |
| `state_machine` | leaky accumulator with a Schmitt trigger: `acc = 0.99 acc + e`, clipped to +-1.5T; flips H to A when `acc <= -T` | explainable, you can see why every switch happened; hysteresis makes leaving harder than staying | detection latency (~0.9 s with defaults); a causal rule cannot use what comes after |
| `viterbi` | minimises (disagreement with the evidence) + (penalty x number of switches) over the whole segment | global optimum; places the switch where the takeover happened (no latency); the penalty directly controls the possession count; one knob | offline (fine here, the whole match is available); the penalty has to be chosen in the right scale |
| `learned` | Viterbi on the per-frame probability of a LightGBM trained ONLY on other games (`src/learned`, `models/`); adds who controlled the ball before/after the frame | handles passes in flight better; sees context | trained on cleaner tracking and on event-derived, not tracking-derived, possession |
| `combined` | Viterbi on 0.5 x (hand-written evidence + learned evidence) | the hedge: physics needs no data, the model adds memory; best mean on held-out games | the gain over either part alone is small (~0.01 spellF1, inside the noise band) |
| `vidal_codina` | team-level adaptation of the possession-control method in Vidal-Codina et al. (2022) | explicit spatial control threshold; requires ball direction/speed change to confirm a team switch | the cached features do not retain individual player tracks, so this is not a full reproduction of the paper's player-level method |

The evidence-based estimators and `vidal_codina` pass through `merge_short_runs`: a "possession"
shorter than 1 s is absorbed by its
neighbours (a blip, not a possession).

`vidal_codina` adapts the possession-control method from Vidal-Codina et al., “Automatic event detection in football
using tracking data,” *Sports Engineering* 25, 18 (2022), https://doi.org/10.1007/s12283-022-00381-6.
It marks a team as controlling the ball when its nearest tracked player is within the
configured possession-zone radius and the opposing team is not also nearby. If both teams fall
within the duel radius, it treats the frame as a duel and does not change the owner. A switch is
accepted only when the ball's direction or speed changes across the new team's control episode.
Because `data_preprocessing` stores only each team's nearest-player distances (not all 46 player positions or stable
player identities), the duel and control-episode tests are necessarily team-level approximations.
Its radii and motion thresholds are project starting values, not the paper's calibrated values.
Compare it with the other methods using `--estimators vidal_codina` and `--sweep`; do not treat it
as a faithful reproduction or tune it on the public ground truth.

### Segments, halves and dead balls

Decoding is done independently inside each **segment**: one half, with no dead ball. Nothing is
carried across halftime, and a dead ball in the `dead_ball` mask **ends** a possession, so the first
possession after a restart is a fresh decision rather than inherited from before the whistle.
Frames before the first decision in a segment are back-filled with it.

By default no frame is labelled `D` except those in the `dead_ball` mask. `--dead_gap_seconds` can mark long
ball gaps as dead, but it is off by default: a tracker dropout inside a real possession would split
it in two, which spellF1 punishes. Dead balls are meant to come from the Smart Tagging and tracking rules in `dead_ball`.

## Parameters

All in `config.py`, each chosen for a physical reason:

| parameter | default | reasoning |
|---|---|---|
| `margin_scale` | 2 m | what counts as "clearly closer" |
| `control_radius` | 3 m | beyond ~6 m nobody controls the ball |
| `speed_lo` / `speed_hi` | 6 / 18 m/s | a carry or dribble vs a pass in flight |
| `sm_switch`, `sm_hold` | 9, 1.5 | a takeover needs ~22 frames (~0.9 s) of confident evidence, the scale of `min_run_seconds` |
| `sm_decay` | 0.99 | old evidence fades (half-life ~2.8 s) |
| `vit_penalty` | 15 | one switch costs 15 frames of full evidence, a there-and-back blip costs 30 |
| `min_run_seconds` | 1 s | real possessions do not last under a second |

## What was and was NOT verified

- **Verified (`selftest.py`):** unit tests of hysteresis, Viterbi, run merging and the dead-ball
  reset; and a *simulated* match with known possessions (carries, passes in flight, pressing
  opponents, tracker dropouts), scored with the real `possession_metrics`. In that simulation the
  state machine and Viterbi score clearly above the majority vote (majority 230, state machine 72,
  Viterbi 87 against 71 true). This synthetic test checks implementation behavior, not match accuracy.
- **Not verified:** anything about the real match. The simulation is something I invented, so it
  only shows the code does what it is designed to do, not that the defaults are right for BAR-ATH.
  The defaults were set on principle; during testing I noticed the state machine was twitchy at
  `sm_switch = 6` and raised it to 9 so a takeover needs about a second, which matches
  `min_run_seconds`. That was a judgement from a synthetic world, so treat it as a starting point.

## How to set the parameters honestly on the real match

1. Run `run_estimators` and read the printed line: possessions, mean duration, home share.
   Real football is roughly 15 to 20 s per possession (the public ground truth's 14.2 s is a sanity
   range, not a target). Hundreds of possessions mean too low a penalty; a handful means too high.
2. Check `evidence: ... mean |e|`. If it is near zero, the evidence is too weak for the penalty
   (the penalty is in units of "one frame of full evidence"). On real tracking expect a few tenths.
3. Run `--sweep` and choose a value from a **flat** region, where possession count and the
   "differs" column both change slowly. A sharp optimum is an artefact.
4. Score on the public 5 minutes only to confirm nothing is broken. It has 17 possessions, so it
   cannot rank settings (`run_evaluation.py --bootstrap 1000` shows the band). Never tune on it.
5. Compare with the majority line printed next to yours. `baseline_smooth50` is the 0.2433 floor.

## Rules

- No training and no labels from THIS match, no hand annotation (RULES.md). `majority`, `state_machine`,
  `viterbi`, `vidal_codina` are rule-based. `learned` / `combined` use a model trained on OTHER games only
  (allowed by RULES.md; documented in TECHNICAL_DOCUMENT.md).
- Do not read `evaluation/gt` from this package.

## Files

| file | job |
|---|---|
| `config.py` | paths and every parameter (`Params`) |
| `io_utils.py` | read features / dead mask, write submissions, count possessions |
| `evidence.py` | the shared evidence layer |
| `postprocess.py` | segmentation, back-fill, short-run merging, shared decode loop |
| `majority.py`, `state_machine.py`, `viterbi.py`, `vidal_codina.py`, `learned.py`, `combined.py` | the possession estimators |
| `run_estimators.py` | the CLI (`--sweep`, `--dead`, parameter flags) |
| `selftest.py` | unit tests and the simulated match |

## Generate a submission

From the repository root, the end-to-end command runs `data_preprocessing`, `dead_ball`, and the
selected Viterbi estimator with penalty 15, then validates the output:

    python make_submission.py --team OffsideAI

The resulting `submission/OffsideAI.csv` is ignored by git because it is derived from the gated
match data. See the root [README.md](../../README.md) for training, evaluation, and submission
instructions.
