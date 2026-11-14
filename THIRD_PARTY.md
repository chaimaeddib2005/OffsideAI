# Provenance and licences

This repository combines work from several sources. Everything below is MIT-licensed or public; none of it
contains the gated BAR-ATH match data.

| what | from | licence | how it is used here |
|---|---|---|---|
| Challenge scaffold: `evaluation/`, `starter/`, `games.py`, organiser documents | Metrica Sports, FC Barcelona, Mobile World Capital Barcelona (More than a Hack 2027) | MIT (code and docs only, see `LICENSE`; the match data is NOT covered) | unchanged; organiser README kept as `README_CHALLENGE.md` |
| `src/data_preprocessing/` (P1), `src/Dead_Ball/` (P3), `src/possession_estimators/` (P2) | our team's implementation in `chaimaeddib2005/OffsideAI` | the repository's `LICENSE` | the pipeline; the only changes are listed in `CHANGES.md` |
| Learned-evidence idea, event-to-possession proxy labels (`src/learned/labels.py`), Metrica loader (adapted into `src/learned/metrica.py`), the LightGBM + Viterbi comparison | `saideepojha/football-possession-estimator` | MIT, Copyright (c) 2026 Saideep Ojha | adapted and extended: features are computed from the P1 table, trained with a cross-game protocol |
| Training data | Metrica Sports public sample data, `github.com/metrica-sports/sample-data`, Sample Games 1 and 2 | public sample data, see that repository | downloaded by the user at training time; not included here |
| `paper_rules` estimator | Vidal-Codina et al. (2022), "Automatic event detection in football using tracking data", Sports Engineering 25, 18, https://doi.org/10.1007/s12283-022-00381-6 | n/a (method only) | team-level adaptation, see `src/possession_estimators/README.md` |

MIT notice for the adapted third-party code (`saideepojha/football-possession-estimator`):

> MIT License. Copyright (c) 2026 Saideep Ojha.
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
> documentation files (the "Software"), to deal in the Software without restriction, including without limitation
> the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
> to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above
> copyright notice and this permission notice shall be included in all copies or substantial portions of the
> Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
