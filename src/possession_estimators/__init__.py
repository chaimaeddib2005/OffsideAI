"""Possession estimators: per-frame features -> H / A labels.

Three estimators share one evidence layer and one post-processing step:

    majority       centred majority vote on the nearest team (the reference baseline)
    state_machine  leaky evidence accumulator with hysteresis (Schmitt trigger)
    viterbi        two-state HMM decoding with a fixed cost per switch

Run from the repository root:   python -m src.possession_estimators.run_estimators
See README.md in this folder.
"""
