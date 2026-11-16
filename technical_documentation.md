# OffsideAI — Technical Documentation

**Project:** More Than a Hack 2027 — Phase 1: Ball Possession  
**Repository:** https://github.com/chaimaeddib2005/OffsideAI  
**Match:** Barcelona vs Athletic Club (BAR–ATH)  
**Primary task:** Frame-level football possession estimation

---

## 1. Project Overview

### 1.1 Problem Definition

OffsideAI is a football analytics solution developed for the More Than a Hack 2027 challenge, organized by Metrica Sports, FC Barcelona, and Mobile World Capital Barcelona.

The objective is to estimate which team controls the ball at every frame of a complete football match.

The challenge provides tracking data for players and the ball, tactical annotations, and other match-related information. However, the tracking data is noisy, contains missing observations, and can suffer from identity switches.

The system must generate one label per frame, following four possible classes:

| Label | Meaning |
|---|---|
| H | Barcelona in possession |
| A | Athletic Club in possession |
| D | Ball is not in play |
| X | Ungraded frame / no prediction |

The final output is a CSV file containing 199,319 frame-level predictions.

### 1.2 Main Technical Challenges

The problem is not simply identifying the player closest to the ball.

Several difficulties must be addressed:

- **Noisy tracking:** Player and ball coordinates may contain errors and inconsistencies.
- **Missing ball observations:** The ball is invisible or unavailable in approximately 41% of the scored frames.
- **Identity switches:** Tracking IDs may change or become associated with different players.
- **Possession persistence:** Possession generally continues until another team gains control; it should not change because of every small movement.
- **Loose balls and passes:** Proximity alone cannot reliably distinguish possession from a pass or an uncontrolled ball.
- **Dead-ball situations:** Throw-ins, corners, free kicks, and other stoppages must be distinguished from active play.

These characteristics make temporal consistency particularly important.

### 1.3 Evaluation Objective

The main ranking metric is **spellF1**, which evaluates the recovery of complete possession sequences rather than isolated frame predictions.

The challenge also reports:

- **macroF1:** Measures frame-level classification performance.
- **spellF1:** Measures possession-spell recovery and is the primary ranking metric.

Consequently, a system that produces frequent possession switches can achieve reasonable frame-level accuracy while performing poorly at recovering actual possession sequences.

---

## 2. System Architecture

The selected approach combines three main processing stages:

1. Tracking preprocessing and feature engineering.
2. Dead-ball detection.
3. Possession estimation using hand-crafted physical evidence and Viterbi decoding.

The architecture separates dead-ball detection from possession estimation. This prevents the possession estimator from carrying an old possession state through a detected stoppage.

### 2.1 Processing Pipeline

**Stage 1 — Data preprocessing**

Raw tracking data is cleaned and transformed into a structured feature table.

**Stage 2 — Dead-ball detection**

A frame-level mask identifies stoppages and separates active-play segments.

**Stage 3 — Possession estimation**

Physical evidence is calculated for both teams and processed by a Viterbi decoder to generate temporally consistent possession predictions.

**Stage 4 — Submission generation**

Predictions are formatted, validated, and exported into the required CSV submission.

---

## 3. Data Preprocessing and Feature Engineering

**Implementation:** `src/data_preprocessing/`

### 3.1 Tracking Data

The system uses Automatic Tracking Data (ATD), which provides player and ball positions over time.

The tracking information is converted into a consistent coordinate system expressed in metres.

This allows distances and velocities to be interpreted in physically meaningful units.

### 3.2 Ball Trajectory Cleaning

Ball tracking is particularly unreliable because the ball is small, moves quickly, and can disappear from the tracking system.

The preprocessing component implements several operations:

**A. Removal of implausible jumps**

Isolated ball movements exceeding 40 m/s are treated as tracking anomalies and removed.

**B. Short-gap interpolation**

Missing ball positions are interpolated when the gap is at most 12 consecutive frames and remains within the same half.

**C. Preservation of long gaps**

Longer missing intervals are not artificially reconstructed. Their values remain NaN.

This is important because reconstructing long trajectories without sufficient information could introduce false evidence about ball movement and possession.

### 3.3 Feature Engineering

After cleaning, a feature table is generated with one row per frame.

The main features include:

| Feature | Description |
|---|---|
| Nearest home-player distance | Distance from the ball to the closest Barcelona player |
| Nearest away-player distance | Distance from the ball to the closest Athletic Club player |
| Ball speed | Estimated instantaneous ball velocity |
| Nearby home-player count | Number of home players within 5 metres of the ball |
| Nearby away-player count | Number of away players within 5 metres of the ball |
| Ball-gap length | Information about missing ball observations |

These features provide the foundation for both dead-ball detection and possession estimation.

The generated feature table is stored in `outputs/features.parquet` when Parquet output is available.

---

## 4. Dead-Ball Detection

**Implementation:** `src/Dead_Ball/`

### 4.1 Motivation

Possession estimation should operate only during active play.

If the ball goes out of play or a referee stops the match, the previous possession state should not automatically continue into the next restart.

The dead-ball detector therefore produces a frame-level classification mask.

Its intermediate labels are:

| Label | Meaning |
|---|---|
| D | Detected dead-ball situation |
| N | Candidate live-play frame |
| X | Ungraded frame |

The final possession pipeline converts live-play frames into H or A.

### 4.2 Smart Tagging Integration

The system uses Smart Tagging annotations as contextual hints.

Relevant events include:

- Set pieces.
- Goals.
- Other annotated restart situations.

These annotations are not directly copied into the final possession labels.

Instead, they help identify potential restart points. The algorithm then examines ball and player activity to estimate the actual stoppage interval.

The Smart Tagging team information is not used as a source of possession labels.

### 4.3 Tracking-Based Detection Rules

The detector supplements Smart Tagging with tracking-based rules to identify situations that automatic annotations may have missed.

The implemented signals include:

**A. Long ball gaps with low player activity**

A prolonged absence of ball observations combined with limited player movement can indicate a stoppage.

**B. Ball outside the pitch**

Ball positions outside the playing area provide evidence of an out-of-play situation.

**C. Stationary-ball intervals**

A ball remaining stationary for a sufficiently long period can indicate a dead-ball situation.

**D. Conservative slow/isolated-ball detection**

An additional rule identifies eligible slow or isolated-ball intervals without extending or merging existing dead-ball intervals.

The detector also uses Smart Tagging information and tracking activity to refine restart boundaries.

### 4.4 Dead-Ball Mask Output

The detector produces:

- `outputs/p3_dead_mask.csv`
- `outputs/dead_mask.csv`

The final mask is consumed by the possession estimator.

Dead-ball frames receive the label D, and each subsequent live-play segment is decoded independently.

This prevents possession information from propagating across stoppages.

---

## 5. Possession Estimation

**Implementation:** `src/possession_estimators/`

### 5.1 Selected Approach: Physics-Inspired Evidence Model

The selected estimator uses hand-crafted physical evidence rather than a trained machine-learning classifier.

The underlying assumption is that possession is a persistent state and that the tracking observations provide imperfect evidence about which team currently controls the ball.

Instead of assigning possession independently at every frame, the system calculates a continuous evidence score.

Let:

\[
e_t \in [-1,1]
\]

where:

- Positive values indicate evidence supporting Barcelona.
- Negative values indicate evidence supporting Athletic Club.
- Values close to zero indicate weak or ambiguous evidence.

The magnitude represents the strength of the evidence.

### 5.2 Proximity-Based Evidence

The estimator compares the distances between the ball and the nearest players from both teams.

Define:

\[
d_H(t) = \text{distance to nearest home player}
\]

\[
d_A(t) = \text{distance to nearest away player}
\]

A distance margin is calculated:

\[
m_t = d_A(t)-d_H(t)
\]

A positive margin supports Barcelona, while a negative margin supports Athletic Club.

The margin is normalized using a hyperbolic tangent function:

\[
E_{\text{margin}}(t)
=
\tanh\left(\frac{m_t}{2}\right)
\]

This limits the evidence to a bounded interval and prevents large distance differences from producing excessively strong evidence.

### 5.3 Proximity Confidence

Being the closest player does not necessarily mean that a player controls the ball.

For this reason, the evidence is weakened when the nearest player is too far away.

The implementation uses a proximity factor that provides stronger evidence when the nearest player is within approximately 3 metres.

### 5.4 Ball-Speed Confidence

Ball speed is another important indicator.

A fast-moving ball may correspond to:

- A pass.
- A shot.
- A clearance.
- A loose ball.
- A ball travelling between players.

In such situations, the closest player should not automatically be considered the possessor.

The model therefore reduces the contribution of proximity evidence as ball speed increases.

The implementation progressively fades the evidence between approximately 6 m/s and 18 m/s, reaching zero evidence at sufficiently high speeds.

### 5.5 Combined Evidence

The physics-based evidence combines:

1. The signed distance margin.
2. The proximity confidence.
3. The ball-speed confidence.

Conceptually:

\[
e_t =
E_{\text{margin}}(t)
\cdot
C_{\text{proximity}}(t)
\cdot
C_{\text{speed}}(t)
\]

The implementation uses the resulting evidence to support one of the two possession states.

When ball observations are unavailable, the evidence is set to zero rather than inventing a position or velocity.

The temporal decoder then maintains the most plausible possession state.

---

## 6. Temporal Decoding Using the Viterbi Algorithm

### 6.1 Why Temporal Decoding?

A simple nearest-player method makes an independent decision at every frame.

This can create unrealistic behaviour:

- Frequent possession switches.
- Short-lived false possession sequences.
- Inconsistent labels during player duels.
- Unnecessary switches caused by tracking noise.

Possession is inherently temporal. Therefore, the system must consider evidence from multiple consecutive frames.

### 6.2 Two-State Model

The Viterbi decoder considers two possible states:

\[
S_t \in \{H,A\}
\]

For each frame, the decoder evaluates how well each state agrees with the calculated physical evidence.

It also introduces a switching penalty.

The optimization can be expressed as:

\[
\min_{S_1,\ldots,S_T}
\left[
\sum_{t=1}^{T} C_{\text{evidence}}(S_t,e_t)
+
\lambda
\sum_{t=2}^{T}
\mathbb{1}(S_t\neq S_{t-1})
\right]
\]

where:

- \(S_t\) is the possession state.
- \(e_t\) is the physical evidence.
- \(C_{\text{evidence}}\) measures disagreement between the state and the evidence.
- \(\lambda\) is the possession-switching penalty.

The actual implementation defines the evidence cost and transition cost in the possession estimator.

### 6.3 Switching Penalty

The selected configuration uses:

\[
\boxed{\lambda=15}
\]

This value controls the resistance to changing possession.

A lower penalty allows more frequent switches, while a higher penalty encourages longer possession sequences.

The choice of 15 is a fixed compromise within the tested 10–20 range. It was not selected by optimizing against the public BAR–ATH ground truth.

### 6.4 Why Viterbi Instead of a Simple Rule?

The Viterbi algorithm performs global sequence optimization.

Unlike a strictly causal state machine, it can consider evidence from both before and after a potential possession transition.

This helps locate handovers more consistently and suppresses short-lived fluctuations.

### 6.5 Dead-Ball Segmentation

The Viterbi decoder is not run continuously across the entire match.

Instead:

1. The dead-ball mask identifies stoppages.
2. Dead-ball frames receive D.
3. Active-play intervals are separated into independent segments.
4. Viterbi decoding is performed independently for each segment.

This ensures that the next possession is determined from fresh evidence after a restart.

---

## 7. Alternative Possession Estimators

Several approaches were implemented to compare different strategies.

They share the same preprocessing and feature table.

| Estimator | Description |
|---|---|
| `viterbi` | Selected hand-crafted physics evidence with global Viterbi decoding |
| `state_machine` | Causal leaky evidence accumulator with hysteresis |
| `majority` | Nearest-team baseline with centred majority filtering |
| `vidal_codina` | Team-level adaptation of the Vidal-Codina possession-control method |
| `learned` | Viterbi decoding using LightGBM-generated evidence |
| `combined` | Viterbi decoding using averaged hand-crafted and learned evidence |

These alternatives are retained for reproducibility and comparison.

The selected submission uses `viterbi` with a switching penalty of 15.

---

## 8. Experimental Machine Learning Component

**Implementation:** `src/learned/`

### 8.1 Motivation

In addition to the hand-crafted physics model, an experimental machine-learning approach was developed to investigate whether possession evidence could be learned from tracking features.

The selected machine-learning algorithm is LightGBM.

Unlike the selected Viterbi estimator, this component requires training data.

### 8.2 Training Dataset

The experiments use the publicly available Metrica Sports sample dataset:

https://github.com/metrica-sports/sample-data

The dataset contains tracking and event information for two sample matches.

The training labels are generated automatically from event annotations rather than manually labelled frame-by-frame possession ground truth.

The label-generation procedure uses events such as:

- PASS
- SHOT
- RECOVERY
- SET PIECE
- BALL LOST
- BALL OUT
- CHALLENGE
- FAULT RECEIVED

Certain events establish possession for the corresponding team, while ambiguous events cause frames to be excluded from training.

The resulting datasets contain approximately:

- Game 1: 78,386 labelled frames.
- Game 2: 80,655 labelled frames.

### 8.3 Training Procedure

Both sample matches are processed using the same preprocessing pipeline used for the challenge data.

Synthetic ball dropouts are also introduced to investigate model robustness to missing observations.

The LightGBM configuration includes:

| Parameter | Value |
|---|---|
| Boosting rounds | 200 |
| Maximum leaves | 15 |
| Random seeds | Fixed |
| Number of threads | 1 |

### 8.4 Cross-Game Evaluation

The experimental model is evaluated using cross-game validation:

1. Train using Metrica Game 1 and test on Game 2.
2. Train using Game 2 and test on Game 1.

This is more informative than evaluating on randomly separated frames from the same match because it tests generalization to a different game.

### 8.5 Why the Learned Model Was Not Selected

Although the learned and combined estimators showed small improvements in some proxy-label experiments, their supervision does not perfectly match the challenge's possession definition.

Metrica event-derived labels are not equivalent to the challenge's tracking-derived possession ground truth.

In particular, disagreements can occur during:

- Passes in flight.
- Loose-ball situations.
- Challenges and duels.
- Ambiguous possession transitions.

Given the limited number of training matches, these improvements were not considered sufficiently reliable to replace the hand-crafted model.

Therefore, LightGBM remains an experimental component and is not required to reproduce the selected submission.

---

## 9. Experimental Results

### 9.1 Hand-Crafted Viterbi Experiments

The hand-crafted model was evaluated on Metrica proxy labels using different switching penalties.

The results below represent the mean spellF1 over four game/condition combinations.

| Switching penalty | Mean spellF1 |
|---:|---:|
| 5 | 0.734 |
| 10 | 0.763 |
| 15 | 0.756 |
| 20 | 0.752 |

The masked conditions introduce synthetic ball dropouts.

These experiments show that the system is sensitive to the switching penalty, but the differences are relatively small.

Penalty 15 was retained as a fixed compromise rather than selected as the maximum-scoring configuration.

### 9.2 Challenge Public-Window Result

The selected hand-crafted Viterbi estimator with penalty 15 achieved:

\[
\text{spellF1}=0.5212
\]

on the public five-minute BAR–ATH evaluation window, using the current dead-ball mask.

This window contains only 17 reference possessions.

Consequently, the result has considerable statistical uncertainty and should not be interpreted as a reliable estimate of full-match performance.

The private full-match ground truth is unavailable, so no official full-match score is reported.

### 9.3 Interpretation

The experiments demonstrate why optimizing frame-level predictions alone is insufficient.

A good possession estimator must recover coherent possession sequences while remaining robust to missing and noisy tracking observations.

The chosen architecture explicitly addresses this requirement through evidence modelling and temporal decoding.

---

## 10. Software Architecture and Repository Structure

The repository is organized into independent modules to facilitate experimentation, testing, and reproducibility.

| Directory / File | Responsibility |
|---|---|
| `src/data_preprocessing/` | Tracking cleaning and feature engineering |
| `src/Dead_Ball/` | Dead-ball detection and mask generation |
| `src/possession_estimators/` | Evidence computation and possession estimators |
| `src/learned/` | Metrica loading, proxy-label generation, LightGBM training and evaluation |
| `evaluation/` | Challenge metrics and submission validation |
| `models/` | Trained LightGBM artifacts and experimental reports |
| `starter/` | Challenge reference implementation and baseline utilities |
| `data/` | Challenge match data, supplied separately |
| `submission/` | Final submission artifacts |
| `games.py` | Match-specific constants and data configuration |
| `make_submission.py` | End-to-end submission generation |
| `TECHNICAL_DOCUMENT.md` | Technical description of the submitted methodology |

---

## 11. Installation and Execution

### 11.1 Requirements

The project uses Python 3.12 and the dependencies pinned in `requirements.txt`.

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

### 11.2 Data Preparation

The challenge match bundle must first be obtained through the official competition registration process.

After accepting the required terms and conditions, extract the data into the repository's `data/` directory.

Alternatively, specify its location using:

```bash
export MTAH_DATA_DIR=/path/to/data
```

The gated match data must not be redistributed or committed to the repository.

### 11.3 Running the Complete Pipeline

The recommended execution command is:

```bash
python3 make_submission.py \
    --team OffsideAI \
    --estimator viterbi \
    --vit-penalty 15 \
    --data-dir ./data
```

This command performs:

1. Tracking preprocessing.
2. Feature generation.
3. Dead-ball detection.
4. Possession estimation.
5. Submission validation.
6. CSV and technical-document generation.

The expected submission artifacts are:

```text
submission/
    OffsideAI.csv
    OffsideAI.md
```

### 11.4 Running the Components Independently

The components can also be executed separately.

**Step 1 — Preprocessing**

```bash
python3 -m src.data_preprocessing.build_features \
    --data-dir ./data
```

**Step 2 — Dead-ball detection**

```bash
python3 -m src.Dead_Ball.build_p3 \
    --features outputs/features.parquet \
    --raw-atd data/tactical/BAR-ATH_tactical_atd.txt \
    --out outputs/p3_dead_mask.csv \
    --dead-out outputs/dead_mask.csv
```

**Step 3 — Possession estimation**

```bash
python3 -m src.possession_estimators.run_estimators \
    --estimators viterbi \
    --vit_penalty 15 \
    --dead outputs/dead_mask.csv \
    --out outputs/possession_estimates
```

If preprocessing generates `features.csv.gz` instead of Parquet, use that generated file as the input to the dead-ball component.

### 11.5 Running Alternative Estimators

To compare all implemented estimators:

```bash
python3 -m src.possession_estimators.run_estimators \
    --estimators viterbi state_machine majority \
                vidal_codina learned combined \
    --dead outputs/dead_mask.csv \
    --out outputs/possession_estimates
```

To perform a Viterbi parameter-stability sweep:

```bash
python3 -m src.possession_estimators.run_estimators \
    --estimators viterbi \
    --dead outputs/dead_mask.csv \
    --sweep
```

### 11.6 Testing

The repository provides self-tests for the principal modules:

```bash
python3 -m src.data_preprocessing.selftest

python3 -m src.possession_estimators.selftest

python3 -m src.learned.selftest
```

### 11.7 Evaluation and Validation

To evaluate predictions against available public ground truth:

```bash
python3 evaluation/run_evaluation.py \
    --gt public \
    --results outputs/possession_estimates \
    --out outputs/viterbi_score
```

To validate the final submission:

```bash
python3 evaluation/validate_submission.py \
    submission/OffsideAI.csv
```

The evaluation module reports frame-level metrics, possession-spell metrics, transition metrics, and diagnostic information.

---

## 12. Limitations and Future Improvements

### 12.1 Current Limitations

**Tracking quality**

The system depends on automatically generated tracking data. Identity switches, missing observations, and incorrect player positions can affect the estimated possession state.

**Limited experimental data**

The LightGBM experiments use only two Metrica sample matches. This is insufficient to establish robust generalization across different leagues, teams, and playing styles.

**Dead-ball detection**

Some stoppages remain difficult to identify. A missed dead-ball interval can merge two distinct possession sequences.

The public-window dead-ball detection results reported in the technical document are:

- Precision: 0.874.
- Recall: 0.509.

**Fixed switching penalty**

The selected Viterbi penalty is fixed at 15. Different matches and playing situations may require different temporal regularization strengths.

**Limited evaluation ground truth**

The public evaluation window is short and contains only 17 reference possessions. The private full-match ground truth is unavailable.

### 12.2 Future Improvements

Several directions could improve the system:

1. **Larger evaluation datasets:** Evaluate the approach on more independently labelled matches.
2. **Adaptive switching penalty:** Investigate context-dependent penalties based on ball speed, player proximity, and possession confidence.
3. **Improved dead-ball detection:** Develop more robust detection rules and validate them across multiple matches.
4. **Confidence estimation:** Estimate uncertainty for each predicted possession and investigate selective abstention.
5. **Spatial information:** Explore the use of homography and pitch geometry to improve restart detection and possession interpretation.
6. **Improved learned models:** Investigate better supervision strategies that more closely match the challenge's definition of possession.
7. **Robustness testing:** Evaluate the effect of tracking gaps, identity switches, and noisy ball trajectories under different conditions.

---

## 13. Conclusion

OffsideAI implements a modular football possession estimation system designed around the temporal nature of possession.

Rather than relying exclusively on nearest-player decisions, the selected approach combines cleaned tracking information, physical evidence, dead-ball detection, and global temporal decoding.

The principal contribution is the integration of an interpretable hand-crafted evidence model with Viterbi sequence optimization, allowing the system to produce more coherent possession sequences despite noisy tracking and missing ball observations.

Alternative rule-based and machine-learning estimators are retained to support experimentation and comparative evaluation.

The final submission uses the hand-crafted Viterbi approach with a switching penalty of 15, without requiring a training stage.

**Primary implementation:** Python  
**Selected estimator:** Physics-inspired evidence + Viterbi decoding  
**Temporal regularization:** Switching penalty = 15  
**Machine-learning experiment:** LightGBM  
**Evaluation priority:** Possession-spell F1

---

## 14. References and Resources

1. [OffsideAI — Project Repository](https://github.com/chaimaeddib2005/OffsideAI)
2. [Challenge Description — CHALLENGE.md](https://github.com/chaimaeddib2005/OffsideAI/blob/main/CHALLENGE.md)
3. [Challenge Data Documentation — DATA.md](https://github.com/chaimaeddib2005/OffsideAI/blob/main/DATA.md)
4. [Evaluation Methodology — EVALUATION.md](https://github.com/chaimaeddib2005/OffsideAI/blob/main/EVALUATION.md)
5. [Existing Technical Document](https://github.com/chaimaeddib2005/OffsideAI/blob/main/TECHNICAL_DOCUMENT.md)
6. [Metrica Sports — Public Sample Dataset](https://github.com/metrica-sports/sample-data)
7. [LightGBM Documentation](https://lightgbm.readthedocs.io/)
8. [Viterbi Algorithm — Original Paper](https://doi.org/10.1109/TIT.1967.1054010)

---

*End of technical documentation.*