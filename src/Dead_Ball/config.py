"""Dead-ball configuration: paths, match constants, and detection heuristics.

The frame structure is imported from games.py where possible and then checked against the
published values for BAR-ATH. Thresholds below are physical or operational choices, not values
fitted to the public ground truth. They are documented in RULES.md and the run report.
"""

import sys
from pathlib import Path


def _find_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "games.py").is_file():
            return parent
    raise RuntimeError("games.py not found above {}".format(Path(__file__).resolve()))


ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games import GAME, GAMES, half_video_blocks  # noqa: E402


FPS = GAMES[GAME]["fps"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]

FIRST_HALF_START = 11185
FIRST_HALF_END = 90651
SECOND_HALF_START = 110808
SECOND_HALF_END = 190197
FIXED_DEAD_START = 110808
FIXED_DEAD_END = 113741
FIXED_DEAD_FRAMES = FIXED_DEAD_END - FIXED_DEAD_START + 1

GRADED_RANGES = (
    (FIRST_HALF_START, FIRST_HALF_END),
    (SECOND_HALF_START, SECOND_HALF_END),
)
HALF_RANGES = {1: (FIRST_HALF_START, FIRST_HALF_END),
               2: (SECOND_HALF_START, SECOND_HALF_END)}

EXPECTED_BLOCKS = {
    1: (FIRST_HALF_START, FIRST_HALF_END),
    2: (SECOND_HALF_START, SECOND_HALF_END),
}

OUTPUT_DIR = ROOT / "outputs"
FEATURES_PATH = OUTPUT_DIR / "features.parquet"      # data_preprocessing output; falls back to features.csv.gz
FEATURES_META_PATH = OUTPUT_DIR / "features_meta.json"
ATD_RAW_PATH = ROOT / "data" / "tactical" / "BAR-ATH_tactical_atd.txt"
ATD_META_PATH = ROOT / "data" / "tactical" / "BAR-ATH_tactical_atd.xml"
OUTPUT_PATH = OUTPUT_DIR / "p3_dead_mask.csv"          # frame,label  (D / N / X)
DEAD_MASK_PATH = OUTPUT_DIR / "dead_mask.csv"          # frame,is_dead -> read by P2 (--dead)
MODULE_DIR = Path(__file__).resolve().parent
DIAGNOSTICS_DIR = OUTPUT_DIR / "p3_diagnostics"        # outputs/ is gitignored

LABELS = ("D", "N", "X")

BALL_TRUST_MAX_SPEED = 40.0
PITCH_TOLERANCE_M = 1.0
BALL_ALIVE_SPEED = 5.0
BALL_STRONG_SPEED = 12.0
PLAYER_ACTIVE_SPEED = 1.6
PLAYER_DEAD_SPEED = 1.0
PLAYER_OOP_MAX_SPEED = 1.5
PLAYER_MIN_OBS = 8

RESTART_COMBINED_MIN_FRAMES = 12
RESTART_STRONG_BALL_MIN_FRAMES = 2
RESTART_MIN_STOPPAGE_S = 2.0
RESTART_BACKTRACK_MAX_S = 60.0

MAX_BACKWARD_SEARCH_S = 60.0
MIN_D_RUN_S = 2.0
MERGE_GAP_S = 2.0

GAP_MIN_S = 3.0
GAP_MAX_MEAN_PLAYER_SPEED = 1.0

OOP_MIN_S = 1.5
OOP_PAD_S = 1.0
OOP_MAX_MEAN_PLAYER_SPEED = 1.5
OOP_MIN_FRACTION = 0.5

STATIONARY_WINDOW_S = 10.0
STATIONARY_MAX_BALL_SPEED = 2.5
STATIONARY_MAX_PLAYER_SPEED = 1.5
STATIONARY_MIN_VALID_FRACTION = 0.9
STATIONARY_MIN_DURATION_S = 8.0
STATIONARY_MERGE_GAP_S = 1.0

GOAL_PRE_PAD_S = 0.0
GOAL_END_PAD_S = 0.0

C2_SEED_MAX_OPPONENT_DISTANCE_M = 25.0
C2_SEED_MAX_BALL_SPEED = 6.0
C2_SEED_MIN_DURATION_S = 5.0
C2_EXPAND_DURATION_S = 10.0
C2_EXPAND_MAX_BALL_SPEED = 12.0
C2_EXPAND_MAX_PLAYER_SPEED = 1.5

RESTART_PROFILES = {
    "THROW_IN": {"pre_pad_s": 6.0, "post_search_s": 20.0, "pre_search_s": 1.0},
    "CORNER": {"pre_pad_s": 8.0, "post_search_s": 15.0, "pre_search_s": 1.0},
    "GOAL_KICK": {"pre_pad_s": 6.0, "post_search_s": 20.0, "pre_search_s": 1.0},
    "FREE_KICK": {"pre_pad_s": 10.0, "post_search_s": 20.0, "pre_search_s": 1.0},
    "KICKOFF_START": {"pre_pad_s": 0.0, "post_search_s": 5.0, "pre_search_s": 0.0},
    "KICKOFF_AFTER_GOAL": {"pre_pad_s": 0.0, "post_search_s": 25.0, "pre_search_s": 0.0},
}

RESTART_TYPE_MAP = {
    "throw in": "THROW_IN",
    "throw-in": "THROW_IN",
    "corner": "CORNER",
    "corner kick": "CORNER",
    "goal kick": "GOAL_KICK",
    "free kick": "FREE_KICK",
    "kick off (start)": "KICKOFF_START",
    "kick off (after goal)": "KICKOFF_AFTER_GOAL",
    "kickoff (start)": "KICKOFF_START",
    "kickoff (after goal)": "KICKOFF_AFTER_GOAL",
}

RESTART_TYPES = tuple(RESTART_PROFILES)

MIN_D_RUN_FRAMES = int(round(MIN_D_RUN_S * FPS))
MERGE_GAP_FRAMES = int(round(MERGE_GAP_S * FPS))
GAP_MIN_FRAMES = int(round(GAP_MIN_S * FPS))
OOP_MIN_FRAMES = int(round(OOP_MIN_S * FPS))
OOP_PAD_FRAMES = int(round(OOP_PAD_S * FPS))
STATIONARY_WINDOW_FRAMES = int(round(STATIONARY_WINDOW_S * FPS))
STATIONARY_MIN_DURATION_FRAMES = int(round(STATIONARY_MIN_DURATION_S * FPS))
STATIONARY_MERGE_GAP_FRAMES = int(round(STATIONARY_MERGE_GAP_S * FPS))
C2_SEED_MIN_FRAMES = int(round(C2_SEED_MIN_DURATION_S * FPS))
C2_EXPAND_MAX_FRAMES = int(round(C2_EXPAND_DURATION_S * FPS))
MAX_BACKWARD_SEARCH_FRAMES = int(round(MAX_BACKWARD_SEARCH_S * FPS))
RESTART_MIN_STOPPAGE_FRAMES = int(round(RESTART_MIN_STOPPAGE_S * FPS))
RESTART_BACKTRACK_MAX_FRAMES = int(round(RESTART_BACKTRACK_MAX_S * FPS))


def check_match_constants():
    blocks = {half: (first, last) for half, first, last in half_video_blocks(GAME)}
    if blocks != EXPECTED_BLOCKS:
        raise AssertionError(
            "games.py half blocks {} disagree with dead-ball constants {}".format(
                blocks, EXPECTED_BLOCKS))
    if FIXED_DEAD_FRAMES != 2934:
        raise AssertionError("fixed second-half dead block is not 2,934 frames")
    if N_FRAMES != 199319:
        raise AssertionError("unexpected tactical frame count {}".format(N_FRAMES))
    return blocks


def graded_mask():
    import numpy as np
    mask = np.zeros(N_FRAMES, dtype=bool)
    for start, end in GRADED_RANGES:
        mask[start:end + 1] = True
    return mask


def half_of_frame(frame):
    for half, (start, end) in HALF_RANGES.items():
        if start <= frame <= end:
            return half
    return 0


def half_start(frame):
    half = half_of_frame(frame)
    return HALF_RANGES[half][0] if half else None
