# features.parquet: column dictionary

One row per **tactical video frame**: row `i` = frame `i` = row `i` of a submission.
199,319 rows. Distances in metres (pitch 105.3 x 68.0). `NaN` = unknown, never a guess.
Team codes: `0` = home (Barcelona), `1` = away (Athletic), `-1` = nobody.

| column | type | meaning |
|---|---|---|
| `frame` | int32 | video frame, 0..199318 |
| `half` | int8 | 1, 2 during play; 0 = pre-match, halftime, post-match (not graded) |
| `graded` | bool | `half > 0` (158,857 frames) |
| `ball_x`, `ball_y` | float32 | ball position (m) after cleaning; NaN if no ball |
| `ball_observed` | bool | ball present in the RAW tracking |
| `ball_valid` | bool | cleaned ball available (observed, minus spikes, plus interpolated) |
| `is_interp` | bool | position was interpolated (gap <= 12 frames, inside a half) |
| `is_spike` | bool | raw observation removed as an impossible jump (> 40 m/s) |
| `frames_since_ball` | int32 | 1,2,3... inside a gap in the cleaned ball; 0 when available |
| `ball_gap_len` | int32 | total length of the gap this frame is in; 0 when available |
| `ball_speed` | float32 | m/s, 5-frame centred mean; NaN without ball |
| `d_home`, `d_away` | float32 | distance from ball to nearest home / away player |
| `margin` | float32 | `d_away - d_home`; > 0 means home is closer. Input for hysteresis |
| `near_team` | int8 | team of nearest player, cleaned ball |
| `near_team_raw` | int8 | same on the RAW ball (reproduces the published baseline) |
| `n_home_near`, `n_away_near` | int8 | players within 5 m of the ball |
| `n_home`, `n_away` | int8 | players tracked this frame (low = bad frame) |

Notes
- Track slots are tracklets (46 slots for 22 players), so there are no per-player features.
- Halves attack in opposite directions after halftime; no direction feature is provided.
- Long ball gaps (> 12 frames) stay NaN on purpose.
