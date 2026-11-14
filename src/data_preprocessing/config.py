"""Every path and every tunable number in one place.

Thresholds are chosen for PHYSICAL reasons (half a second, shot speed...), never by watching a
score move. That is what keeps the later spellF1 honest -- see RULES.md.
"""

import sys
from pathlib import Path


def _find_root():
    """Repository root = the nearest parent folder that contains games.py."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "games.py").is_file():
            return parent
    raise RuntimeError("games.py not found above {}. Keep this folder inside the repository "
                       "(e.g. <repo>/src/data_preprocessing).".format(Path(__file__).resolve()))


ROOT = _find_root()
if str(ROOT) not in sys.path:          # so `import games` and `import starter.baseline` work
    sys.path.insert(0, str(ROOT))

from games import GAME, GAMES  # noqa: E402

# ---- paths ----------------------------------------------------------------------------------
OUTPUT_DIR = ROOT / "outputs"                       # gitignored: derived from gated data
CACHE_DIR = OUTPUT_DIR / "cache"
FEATURES_PATH = OUTPUT_DIR / "features.parquet"     # falls back to features.csv.gz without pyarrow
META_PATH = OUTPUT_DIR / "features_meta.json"
BASELINE_SMOOTH50 = ROOT / "evaluation" / "results" / "baseline_smooth50.csv"

# ---- match constants (from games.py, so there is one source of truth) -----------------------
FPS = GAMES[GAME]["fps"]                            # 25
N_FRAMES = GAMES[GAME]["n_video_frames"]            # 199319
PITCH = GAMES[GAME]["pitch"]                        # (105.3, 68.0) metres

HOME, AWAY, NOBODY = 0, 1, -1                       # same codes as starter/baseline.py

# ---- cleaning thresholds --------------------------------------------------------------------
MAX_BALL_SPEED = 40.0     # m/s. Hard shots are ~35 m/s; anything faster is a tracker jump.
MAX_GAP_FRAMES = 12       # fill ball gaps up to 12 frames = 0.48 s. Longer gaps stay NaN.
DESPIKE_PASSES = 2

# ---- feature parameters ---------------------------------------------------------------------
NEAR_RADIUS = 5.0         # metres, for "players close to the ball" counts
SPEED_SMOOTH = 5          # frames, centred rolling mean for ball speed

# ---- QA expectations (all quoted in DATA.md) ------------------------------------------------
EXPECT_BALL_OBSERVED_ALL = 113_488        # of 199,319 frames (56.9%)
EXPECT_BALL_OBSERVED_GRADED = 94_214      # of 158,857 graded frames (59.3%)
EXPECT_GRADED_FRAMES = 158_857
EXPECT_MEAN_TRACKED = 20.5                # mean tracked objects over graded frames
