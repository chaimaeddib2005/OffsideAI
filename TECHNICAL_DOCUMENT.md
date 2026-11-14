# Technical document: ball possession, BAR-ATH
**Team:** <<TEAM NAME>> &nbsp;|&nbsp; **Members:** <<NAMES>> &nbsp;|&nbsp; **Repository:** <<PUBLIC REPO URL>>

> Fill every `<<...>>` before uploading. Numbers marked `<<from run>>` come from the output of
> `python make_submission.py` on the real match; they cannot be known without the gated data.

## 1. Approach

We treat possession as a **state that persists** until the other team clearly takes the ball, and the tracking as
noisy **evidence** about that state. The ranking metric (spellF1) scores whole possessions, so frame-by-frame
decisions are punished: the organisers' nearest-player baseline reports 131 possessions against 17 real ones in the
public window, and merely smoothing it raises spellF1 from 0.04 to 0.42 while macroF1 barely moves. Pipeline:

1. **P1, cleaning** (`src/data_preprocessing`): ATD to metres; remove isolated ball jumps above 40 m/s; fill ball
   gaps up to 12 frames inside a half; one row per frame with distances of the nearest home/away player, ball speed,
   counts of players within 5 m, gap lengths. Longer ball gaps stay `NaN`, never guessed.
2. **P3, dead balls** (`src/Dead_Ball`): a `D` mask. Smart Tagging set pieces and goals are used as *hints* to
   search for the real restart in ball/player activity, then the dead interval is searched backwards from it.
   Three tracking-only rules add stoppages Smart Tagging missed (long ball gaps with still players, ball outside the
   pitch, stationary ball). Short `D` runs (<2 s) are removed; short gaps merged.
3. **P2, possession** (`src/possession_estimators`): hand-written per-frame evidence `e` in [-1, 1] (sign = team,
   size = trust), then a two-state Viterbi decoder minimising disagreement with the evidence plus a fixed cost of
   **10 per switch** (`vit_penalty=10`). It is a global optimum over each segment, so it places a switch where the
   evidence supports a takeover and suppresses short blips using evidence from both sides.
4. **Evidence** `e_physics`: tanh(margin / 2 m) x proximity (nearest player within ~3 m) x not-in-flight (ball speed
   6 to 18 m/s fades to 0). No ball means `e = 0` and the state simply persists. The LightGBM evidence model was
   evaluated as an alternative and in combination with the physics evidence, but is not used in the selected output.

Why this method: it is explainable and directly uses the provided tracking. On Metrica Sample Games 1 and 2, the
hand-written Viterbi method with penalty 10 outperformed the tested nearest-player baselines on clean tracking and
slightly exceeded penalty 20 on average across clean and synthetic-dropout tests. The learned component's small
proxy-label advantage was not considered reliable enough to include, because its event-derived supervision differs
from the challenge's corrected-tracking labels.

## 2. Assets used (and not used)

| asset | use |
|---|---|
| ATD tracking (players, ball) | all features (P1), player speeds for P3 |
| ATD metadata XML | player-slot count for P3 |
| Smart Tagging XML | P3 only: restart and goal hints (never copied as labels; team tag ignored) |
| `games.py` constants, FAQ.md | pitch size, frame counts, half blocks; the FAQ's statement that the first 2,934 graded frames of the second half are all `D` |
| Public 5-minute ground truth | a small diagnostic check; not a valid parameter-selection set under the challenge rules |
| Available but not used for the submission | `state_machine` and `paper_rules` estimators (kept for comparison); `--dead_gap_seconds` (left off: a tracker dropout inside a possession would split it) |
| Not used | tactical and broadcast video, homography, broadcast ATD |

## 3. Metrica experiments (not used by the selected estimator)

The selected hand-written estimator has no training step. For comparison, a LightGBM model was trained only on
Metrica Sports public sample data
(`github.com/metrica-sports/sample-data`), Sample Games 1 and 2: 25 Hz tracking and event data, provided by Metrica
Sports (not by us). Per-frame labels are derived **by an algorithm** from the event data: PASS, SHOT, RECOVERY and SET
PIECE start possession for the event's team and it is carried forward; BALL LOST, BALL OUT, CHALLENGE and FAULT
RECEIVED make it undetermined and those frames are excluded (`src/learned/labels.py`). 78,386 and 80,655 labelled
frames. Nothing was labelled by hand. **Nothing from BAR-ATH** (tracking, tagging, homography, public or private
ground truth, or any human-produced data about it) was used to train or validate that model.

Both Metrica games were passed through the same P1 code as BAR-ATH. The model training also used two extra copies
per game with synthetic ball dropouts (an assumption about the dropout pattern, harsher than BAR-ATH). LightGBM used
200 rounds, 15 leaves, fixed seeds and one thread.

**Validation** is cross-game (train on one Metrica game, test on the other) with the organisers' scorer on the
proxy labels. In the original four-condition report, mean spellF1 was 0.595 for majority vote, 0.752 for
hand-written Viterbi at penalty 20, 0.762 for Viterbi on learned evidence, and 0.764 for combined evidence
(`models/cross_game_results.md`). A separate hand-only comparison gave penalty 10 a mean of 0.763 versus 0.752 for
penalty 20 (the four per-condition values are in `README.md`). These proxy-label comparisons are not estimates of
the challenge score. The learned model and combined estimator were dropped from the selected output because their
small cross-game gain is uncertain under the label-semantics mismatch.

## 3b. Result on the real match

The selected hand-written Viterbi estimator with penalty 10 was reported as scoring approximately 0.44 spellF1 on
the public five-minute window. This window contains only 17 reference possessions and is not a reliable estimate of
full-match performance. The private full-match ground truth is unavailable, so no official full-match score is
reported here. Fill in the remaining submission-run statistics from `make_submission.py` before uploading.

## 4. Possession rules implemented

* **Persistence.** Possession stays with the last team that had it until the other team gains controlled possession
  (CHALLENGE.md). Control is evidenced by proximity to a slow ball, accumulated over time; a single touch does not flip
  the state because a switch has a fixed cost.
* **Loose balls and passes in flight.** A ball faster than ~6 m/s carries less physics evidence and above ~18 m/s none;
  the state persists through it.
* **Challenges and duels.** Both teams close gives a small margin and so little evidence; the state persists.
* **Missing ball (41% of graded frames).** No evidence, state persists. No `X` is written inside the halves, because
  under spellF1 an abstention is filled with the previous label and costs most exactly at a hand-over.
* **Restarts and stoppages.** A dead ball is `D` and ends a possession; the first possession after it is decided
  afresh. Nothing is carried across halftime. Runs shorter than 1 s are absorbed by their neighbours.

## 5. Limitations

* The experimental learned model learned **event-derived** possession; the challenge truth is **tracking-derived**
  ownership. They disagree most during passes in flight and loose balls; this label mismatch is why the learned
  component was not selected.
* The Metrica comparison covers only two games, with event-derived proxy labels. It does not establish a guaranteed
  best penalty or predict the challenge score.
* BAR-ATH tracking has 46 slots for 22 players, with tracklets and identity switches.
* P3 dead-ball recall on the public window was 0.509 (precision 0.874): some stoppages are missed, and a missed `D`
  merges two possessions. Smart Tagging is automatic and contains false positives/negatives; its `Team` is `N/A` for every
  throw-in, so it gives no team information there.
* Hard-coded match constants are limited to those the organisers publish (half blocks, the 2,934-frame dead block).
* With more time: evaluate on more independently labelled matches; validate any adaptive penalty rule across many
  held-out games; per-possession confidence and selective `X` away from hand-overs; use the homography for restart
  geometry; evaluate P3 against more matches.

## 6. Reproduction

```
pip install -r requirements.txt
python make_submission.py --team <<TEAM>> --estimator viterbi --vit-penalty 10 --data-dir <path to data bundle>
python -m src.learned.train --metrica-dir <sample-data>/data                 # optional: regenerates models/ exactly
```
Python 3.12, pins in `requirements.txt`. The selected Viterbi submission does not load the committed learned model,
so reproducing the CSV does not require the Metrica download. SHA-256 of the submitted file: <<from run>>.
