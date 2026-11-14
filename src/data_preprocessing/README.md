# data_preprocessing (P1)

Turns the raw tracking (ATD) into ONE clean per-frame table that P2 and P3 read.
It decides nothing about possession: no H/A, no D, no X.

## Install

Put this folder at `<repo>/src/data_preprocessing/`. Add `pyarrow` to `requirements.txt`
(otherwise the table is written as `features.csv.gz`) and add `outputs/` to `.gitignore`
(the table is derived from gated data and must not be committed).

## Run (from the repo root)

    python -m src.data_preprocessing.selftest         # no data needed; proves the code works
    python -m src.data_preprocessing.build_features   # builds outputs/features.parquet, then runs QA

Options: `--max-gap N`, `--max-speed V`, `--no-cache` (re-parse raw ATD), `--skip-qa`, `--data-dir`.
First run parses the ATD (~1 min) and caches it in `outputs/cache/`.

## Files

| file | job |
|---|---|
| `config.py` | all paths and thresholds |
| `load_atd.py` | raw ATD -> arrays; the only place with the `frame_id - 1` shift |
| `clean.py` | metres, despike the ball, fill short ball gaps (per half) |
| `features.py` | per-frame features (distances, margin, counts, gap lengths, speed) |
| `build_features.py` | the one command |
| `qa.py` | sanity checks against numbers published in DATA.md and the published baseline |
| `selftest.py` | synthetic-data test of the pipeline |
| `FEATURES.md` | column dictionary: the contract with P2 and P3 |

## Rules

- Never use the public ground truth here. Never fix tracking by hand.
- Thresholds are chosen for physical reasons, not by watching a score.
- Do not rename columns once P2/P3 use them.
