"""Paths and default parameters.

Every default is chosen for a PHYSICAL reason (see README.md), not fitted to a score. They have
been sanity-checked on a synthetic match (selftest.py) and must be checked on the real one by
looking at possession counts and durations -- never by tuning against the public 5 minutes.
"""

import sys
from dataclasses import dataclass, asdict
from pathlib import Path


def _find_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "games.py").is_file():
            return parent
    raise RuntimeError("games.py not found above {}. Keep this folder inside the repository "
                       "(e.g. <repo>/src/possession_estimators).".format(Path(__file__).resolve()))


ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games import GAME, GAMES  # noqa: E402

FPS = GAMES[GAME]["fps"]
N_FRAMES = GAMES[GAME]["n_video_frames"]

OUTPUT_DIR = ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "features.parquet"
ESTIMATES_DIR = OUTPUT_DIR / "possession_estimates"

HOME, AWAY, DEAD = 0, 1, 2
LETTERS = ("H", "A", "D")


@dataclass
class Params:
    # --- evidence layer ---------------------------------------------------------------------
    margin_scale: float = 2.0      # m. margin of this size gives tanh(1) = 76% of full side evidence
    control_radius: float = 3.0    # m. beyond ~2x this, the nearest player is not "in control"
    speed_lo: float = 6.0          # m/s. below: ball is carried/dribbled, evidence fully trusted
    speed_hi: float = 18.0         # m/s. above: ball is in flight (a pass or clearance), no evidence
    interp_weight: float = 0.5     # interpolated ball positions are half as trustworthy

    # --- state machine ----------------------------------------------------------------------
    sm_switch: float = 9.0         # weighted frames of opposing evidence. With sm_hold a takeover
                                   # needs ~22 frames (~0.9 s) of confident evidence -- about the
                                   # shortest real possession, the same scale as min_run_seconds.
    sm_hold: float = 1.5           # reservoir cap = hold * switch. >1 makes leaving harder than staying
    sm_decay: float = 0.99         # per-frame leak; stale evidence fades (half-life ~70 frames)
    sm_init: float = 4.0           # weighted frames needed to decide the first possession of a segment

    # --- viterbi ----------------------------------------------------------------------------
    vit_penalty: float = 20.0      # cost of one switch, in units of "one frame of full evidence"
    vit_lambda: float = 1.0        # evidence scale (only the ratio penalty/lambda matters)
    learned_penalty: float = 10.0   # same, for the learned evidence alone
    combined_penalty: float = 10.0  # same, for the hand-written + learned average (the submission default)
    # Both chosen on OTHER games by a pre-declared rule (flat region of cross-game spellF1, see
    # models/training_report.json), never on the BAR-ATH public window.

    # --- paper-inspired team control rules --------------------------------------------------
    paper_pz_radius: float = 1.5    # m; team-nearest adaptation of the paper's possession zone
    paper_duel_radius: float = 2.5  # m; both teams this close suppresses a team control vote
    paper_motion_window_frames: int = 3
    paper_direction_cosine: float = 0.5
    paper_speed_delta: float = 3.0  # m/s

    # --- post-processing --------------------------------------------------------------------
    min_run_seconds: float = 1.0   # a "possession" shorter than this is absorbed by its neighbours

    # --- majority reference -----------------------------------------------------------------
    majority_window: int = 50      # half-width in frames; 50 is the published baseline

    # --- dead balls (P3) --------------------------------------------------------------------
    dead_gap_seconds: float = 0.0  # >0: treat ball gaps this long as dead. OFF by default: a
                                   # tracker dropout inside a possession would split it in two

    def as_dict(self):
        return asdict(self)
