"""Label-free checks for choosing the estimator parameters. No ground truth from the match is used.

    python -m src.possession_estimators.tuning_checks --dead outputs/dead_mask.csv
    python -m src.possession_estimators.tuning_checks --dead outputs/dead_mask.csv --list-codes
    python -m src.possession_estimators.tuning_checks --selftest        # no match data needed

CHECK A -- Smart Tagging team agreement
    Smart Tagging is Metrica's automatic annotation. Most of its phases carry a Team tag, and
    every set piece is a restart. For each setting we count how often OUR possession label
    names the same team as the tag:
      * set pieces : the 5 s AFTER the tagged restart (the team that takes it should have the ball)
      * goals      : the 10 s BEFORE the goal           (assumes Team = the scorer)
      * phases     : inside the tagged interval         (every other code with a Team tag)
    Next to each number sits a CONTROL: the same events with the Team tags shuffled among them.
    If agreement is not clearly above its control, the check says nothing. Read "lift" =
    agreement minus control, not the raw percentage.

    Caveats you must keep in mind:
      * It is automatic, not ground truth, and probably derived from the same tracking as our
        evidence, so its errors are correlated with ours. It can warn; it cannot prove.
      * Phases are not possessions, and what "Team" means differs by code. Use --list-codes, read
        the per-code table, and restrict with --codes if some codes behave like "the defending
        team". A code whose agreement is far BELOW its control is probably tagged for the other side.

CHECK B -- hidden-ball test
    Hide the ball in random blocks (evidence set to zero, the same thing a tracker dropout does),
    decode again, and compare the hidden frames with the decode that had the ball. It measures how
    much of a decision survives a missing ball and how many possessions appear or vanish. It does
    NOT depend strongly on the penalty (with no evidence nothing pushes a switch either way), so
    it is a robustness report, not a way to pick the penalty. Do not over-read it.
"""

import argparse
import dataclasses
import re
import sys

import numpy as np

from . import config, io_utils
from .config import Params
from .evidence import compute_evidence
from .run_estimators import (ESTIMATORS, SWEEPS, add_param_args, dead_mask, params_from_args,
                             run_one)

HOME, AWAY, DEAD = config.HOME, config.AWAY, config.DEAD
FPS = config.FPS

# Grids a little denser than --sweep around the region that matters. Override with --grid.
DEFAULT_GRID = {"vit_penalty": [8.0, 12.0, 16.0, 20.0, 30.0, 40.0],
                "sm_switch": [6.0, 9.0, 12.0],
                "majority_window": [50, 125]}


# ----------------------------------------------------------------------------- helpers
def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


_BLANK = {"", "na", "none", "null", "unknown", "nan"}
_STOP = {"fc", "cf", "club", "de", "the", "sc", "ac", "afc", "cd", "ud"}


def _score(a, b):
    """3 = same name, 2 = one contains the other, 1 = a shared meaningful word, 0 = nothing."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0
    if a == b:
        return 3
    if min(len(a), len(b)) >= 4 and (a in b or b in a):
        return 2
    return 1 if (set(a.split()) & set(b.split())) - _STOP else 0


def map_teams(tag_values, home, away, override=None):
    """{tag value -> 0 home / 1 away}. Raises with a readable message if a value cannot be mapped,
    because guessing the side would flip every agreement number."""
    mapping, unknown = {}, []
    for v in tag_values:
        if _norm(v).replace(" ", "") in _BLANK:
            continue                      # "N/A", "none"...: no team, the event is skipped
        if override and v in override:
            mapping[v] = override[v]
            continue
        h, a = _score(v, home), _score(v, away)
        if h > a:
            mapping[v] = HOME
        elif a > h:
            mapping[v] = AWAY
        else:
            unknown.append(v)
    if unknown:
        raise ValueError(
            "cannot map Smart Tagging Team value(s) {} to home/away.\nATD metadata names: home={!r}, "
            "away={!r}.\nPass --team-map \"<tag value>=H,<tag value>=A\" (home = {}).".format(
                unknown, home, away, home))
    return mapping


def parse_team_map(text):
    out = {}
    for part in (text or "").split(","):
        if "=" in part:
            k, v = part.rsplit("=", 1)
            out[k.strip()] = HOME if v.strip().upper() == "H" else AWAY
    return out


def parse_grid(items):
    grid = {}
    for item in items or []:
        key, values = item.split("=")
        grid[key] = [float(v) for v in values.split(",")]
    return grid


def settings(p, names, grid):
    out = []
    for name in names:
        knob, _ = SWEEPS[name]
        values = grid.get(knob, DEFAULT_GRID.get(knob, SWEEPS[name][1]))
        for v in values:
            cast = type(getattr(p, knob))
            out.append((name, knob, cast(v), dataclasses.replace(p, **{knob: cast(v)})))
    return out


def label_of(name, knob, value, p):
    return "{} {}={:g}{}".format(name, {"vit_penalty": "pen", "sm_switch": "T", "majority_window": "w"}[knob],
                                 value, "*" if value == getattr(Params(), knob) else "")


# ---------------------------------------------------------------- CHECK A: Smart Tagging
def collect_events(events, mapping, codes_filter=None, include_defending=False):
    """Events that carry a mapped Team tag, tagged with a kind: set_piece / goal / phase.

    Codes with DEFEN in the name ("DEFENDING IN MIDDLE THIRD", "DEFENSIVE TRANSITION"...) are
    skipped unless include_defending: their Team is the team WITHOUT the ball, and they repeat the
    same intervals as the attacking codes with the other team, so they would only double-count."""
    rows = []
    for e in events:
        team = e["tags"].get("Team")
        if team not in mapping:
            continue
        code = (e["code"] or "").strip().upper()
        if codes_filter and code not in codes_filter:
            continue
        if not include_defending and "DEFEN" in code:
            continue
        kind = "set_piece" if code == "SET PIECES" else "goal" if code.startswith("GOAL") else "phase"
        rows.append({"code": code, "type": (e["tags"].get("Type") or "").strip(), "kind": kind,
                     "team": mapping[team], "start": e["start_frame"], "end": e["end_frame"]})
    return rows


def _window_arrays(rows, kind, post_frames, pre_frames, offset_frames=0):
    sel = [r for r in rows if r["kind"] == kind]
    if not sel:
        return None
    start = np.array([r["start"] for r in sel]); end = np.array([r["end"] for r in sel])
    if kind == "set_piece":
        a, b = end + offset_frames, end + offset_frames + post_frames
    elif kind == "goal":
        a, b = start - pre_frames, start
    else:
        a, b = start, end
    return sel, a, b, np.array([r["team"] for r in sel])


def _counts(codes, a, b):
    n = len(codes)
    ch = np.concatenate(([0], np.cumsum(codes == HOME)))
    ca = np.concatenate(([0], np.cumsum(codes == AWAY)))
    a = np.clip(a, 0, n); b = np.maximum(np.clip(b, 0, n), a)
    return ch[b] - ch[a], ca[b] - ca[a]


def _agree(n_home, n_away, teams, min_alive):
    own = np.where(teams == HOME, n_home, n_away)
    alive = n_home + n_away
    frame = own.sum() / max(alive.sum(), 1)
    use = alive >= min_alive
    phase = float((own[use] > (alive - own)[use]).mean()) if use.any() else float("nan")
    return frame, phase, int(use.sum())


def _control(n_home, n_away, teams, rng, reps=200):
    alive = n_home + n_away
    total = max(alive.sum(), 1)
    vals = []
    for _ in range(reps):
        t = rng.permutation(teams)
        vals.append(np.where(t == HOME, n_home, n_away).sum() / total)
    return float(np.mean(vals))


def tagging_agreement(codes, rows, post_s, pre_s, min_alive_s=1.0, seed=0, offset_s=0.0):
    """{kind: dict(n, frame, phase, control)} for one decoded label array."""
    rng = np.random.default_rng(seed)
    out = {}
    for kind in ("set_piece", "goal", "phase"):
        w = _window_arrays(rows, kind, int(post_s * FPS), int(pre_s * FPS), int(offset_s * FPS))
        if w is None:
            continue
        sel, a, b, teams = w
        nh, na = _counts(codes, a, b)
        frame, phase, used = _agree(nh, na, teams, int(min_alive_s * FPS))
        out[kind] = {"n": len(sel), "used": used, "frame": frame, "phase": phase,
                     "control": _control(nh, na, teams, rng)}
    return out


def per_code_table(codes, rows, min_alive_s=1.0, seed=0):
    rng = np.random.default_rng(seed)
    table = []
    for code in sorted({r["code"] for r in rows if r["kind"] == "phase"}):
        sel = [r for r in rows if r["kind"] == "phase" and r["code"] == code]
        a = np.array([r["start"] for r in sel]); b = np.array([r["end"] for r in sel])
        teams = np.array([r["team"] for r in sel])
        nh, na = _counts(codes, a, b)
        frame, phase, used = _agree(nh, na, teams, int(min_alive_s * FPS))
        table.append((code, len(sel), int((nh + na).sum()), frame, _control(nh, na, teams, rng)))
    return table


def per_set_piece_type(codes, rows, post_s, offset_s, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for typ in sorted({r["type"] for r in rows if r["kind"] == "set_piece"}):
        sel = [r for r in rows if r["kind"] == "set_piece" and r["type"] == typ]
        a = np.array([r["end"] for r in sel]) + int(offset_s * FPS); b = a + int(post_s * FPS)
        teams = np.array([r["team"] for r in sel])
        nh, na = _counts(codes, a, b)
        frame, _, _ = _agree(nh, na, teams, 1)
        out.append((typ or "(none)", len(sel), frame, _control(nh, na, teams, rng)))
    return out


def run_tagging_check(table, evidence, dead, p, args, events, mapping):
    rows = collect_events(events, mapping, set(c.strip().upper() for c in args.codes) if args.codes else None,
                          include_defending=args.include_defending)
    skipped_team = sum(1 for e in events if e["tags"].get("Team") not in mapping)
    skipped_def = 0 if args.include_defending else len(
        {id(e) for e in events if "DEFEN" in (e["code"] or "").upper() and e["tags"].get("Team") in mapping})
    kinds = {k: sum(1 for r in rows if r["kind"] == k) for k in ("set_piece", "goal", "phase")}
    print("\n" + "=" * 100)
    print("CHECK A -- Smart Tagging team agreement   (events with a mapped Team tag: {})".format(kinds))
    print("=" * 100)
    print("agreement = share of our alive frames that name the tagged team; (control) = same with the tags "
          "shuffled; lift = agreement - control.")
    print("skipped: {} events with no usable Team tag (e.g. N/A); {} 'defending' events (Team = side without "
          "the ball; use --include-defending to keep them).\n".format(skipped_team, skipped_def))
    head = "{:<22} {:>24} {:>24} {:>26}".format("setting", "set pieces, 5 s after", "goals, 10 s before",
                                                "phases, inside")
    print(head)
    print("-" * len(head))
    default_codes = {}
    for name, knob, value, q in settings(p, args.estimators, parse_grid(args.grid)):
        codes = run_one(name, table, q, dead, evidence)
        res = tagging_agreement(codes, rows, args.post_seconds, args.pre_seconds, offset_s=args.post_offset_seconds)
        cells = []
        for kind in ("set_piece", "goal", "phase"):
            r = res.get(kind)
            cells.append("n/a" if not r else "{:>5.1f}% ({:>4.1f}%) lift {:+5.1f}{}".format(
                100 * r["frame"], 100 * r["control"], 100 * (r["frame"] - r["control"]),
                " n<20!" if r["n"] < 20 else ""))
        print("{:<22} {:>24} {:>24} {:>26}".format(label_of(name, knob, value, p), *cells))
        if value == getattr(Params(), knob):
            default_codes[name] = codes
    print("\n(* = package default)")

    if default_codes:
        name = "viterbi" if "viterbi" in default_codes else sorted(default_codes)[0]
        print("\nPer code, {} default (phases only). A code far BELOW its control is probably tagged for "
              "the other side:".format(name))
        print("{:<34} {:>7} {:>12} {:>10} {:>10} {:>8}".format("code", "events", "alive frames", "agree", "control", "lift"))
        for code, n, alive, frame, ctrl in per_code_table(default_codes[name], rows):
            print("{:<34} {:>7} {:>12} {:>9.1f}% {:>9.1f}% {:>+7.1f}".format(
                code[:34], n, alive, 100 * frame, 100 * ctrl, 100 * (frame - ctrl)))
        sp = per_set_piece_type(default_codes[name], rows, args.post_seconds, args.post_offset_seconds)
        if sp:
            print("\nSet pieces by type, {} default ({} s window starting {} s after the tag ends):".format(
                name, args.post_seconds, args.post_offset_seconds))
            print("{:<34} {:>7} {:>10} {:>10} {:>8}".format("type", "events", "agree", "control", "lift"))
            for typ, n, frame, ctrl in sp:
                print("{:<34} {:>7} {:>9.1f}% {:>9.1f}% {:>+7.1f}".format(typ[:34], n, 100 * frame, 100 * ctrl,
                                                                        100 * (frame - ctrl)))
    print("\nHow to read it: choose between settings only if the lift differs clearly and consistently across "
          "the three columns. Differences of a point or two are noise (a few hundred events).")


# ------------------------------------------------------------------ CHECK B: hidden ball
def hidden_ball(table, evidence, dead, name, q, block_s, share, seed=0):
    n = len(table)
    L = max(int(block_s * FPS), 1)
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    cell = max(int(L / share), L + 1)
    for c in range(0, n - cell, cell):
        off = int(rng.integers(0, cell - L + 1))
        mask[c + off:c + off + L] = True
    ev = evidence.copy()
    ev[mask] = 0.0
    t2 = table.copy()
    t2["near_team"] = np.where(mask, -1, table["near_team"].to_numpy()).astype(np.int8)
    base = run_one(name, table, q, dead, evidence)
    hid = run_one(name, t2, q, dead, ev)
    graded = table["graded"].to_numpy()
    eval_frames = mask & graded & (evidence != 0) & ((base == HOME) | (base == AWAY))
    agree = float((hid[eval_frames] == base[eval_frames]).mean()) if eval_frames.any() else float("nan")
    return agree, int(eval_frames.sum()), io_utils.possession_runs(base, graded)[0], \
        io_utils.possession_runs(hid, graded)[0]


def run_hidden_check(table, evidence, dead, p, args):
    print("\n" + "=" * 100)
    print("CHECK B -- hidden-ball test   (ball hidden in blocks covering {:.0f}% of the match)".format(100 * args.hide_share))
    print("=" * 100)
    print("agree = hidden frames whose label equals the decode WITH the ball. runs = possessions before -> after.\n")
    head = "{:<22} {:>22} {:>22}".format("setting", "3 s blocks", "8 s blocks")
    print(head)
    print("-" * len(head))
    for name, knob, value, q in settings(p, args.estimators, parse_grid(args.grid)):
        cells = []
        for block_s in (3.0, 8.0):
            agree, n_eval, runs0, runs1 = hidden_ball(table, evidence, dead, name, q, block_s, args.hide_share)
            cells.append("{:>5.1f}%  {:>4d} -> {:>4d}".format(100 * agree, runs0, runs1))
        print("{:<22} {:>22} {:>22}".format(label_of(name, knob, value, p), *cells))
    print("\nA possession count that explodes or collapses when the ball is hidden means the setting leans on "
          "the ball too hard or too little. This check is a robustness report; it does not pick the penalty.")


# --------------------------------------------------------------------------------- CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", default=None)
    parser.add_argument("--dead", default=None, help="dead_ball mask (frame,label D/N/X)")
    parser.add_argument("--data-dir", default=None, help="override MTAH_DATA_DIR / ./data")
    parser.add_argument("--estimators", nargs="+", default=["viterbi", "state_machine"], choices=list(ESTIMATORS))
    parser.add_argument("--grid", nargs="*", help="e.g. vit_penalty=8,12,20,30  sm_switch=6,9,12")
    parser.add_argument("--codes", nargs="*", help="restrict CHECK A to these Smart Tagging codes")
    parser.add_argument("--team-map", default=None, help='e.g. "FC Barcelona=H,Athletic Club=A"')
    parser.add_argument("--post-seconds", type=float, default=5.0)
    parser.add_argument("--pre-seconds", type=float, default=10.0)
    parser.add_argument("--post-offset-seconds", type=float, default=0.0,
                        help="start the set-piece window this long after the tag ends")
    parser.add_argument("--include-defending", action="store_true",
                        help="keep codes with DEFEN in the name (default: skipped, see collect_events)")
    parser.add_argument("--hide-share", type=float, default=0.25)
    parser.add_argument("--list-codes", action="store_true", help="print Smart Tagging codes/teams and exit")
    parser.add_argument("--skip-tagging", action="store_true")
    parser.add_argument("--skip-hidden", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    add_param_args(parser)
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    p = params_from_args(args)
    table = io_utils.load_features(args.features)
    graded = table["graded"].to_numpy()
    dead = dead_mask(table, p, args.dead)
    evidence = compute_evidence(table, p)
    print("{} frames, {} graded, dead-ball frames inside graded play: {}".format(
        len(table), int(graded.sum()), 0 if dead is None else int((dead & graded).sum())))

    mapping = events = None
    if not args.skip_tagging or args.list_codes:
        from games import GAME, require
        from starter.baseline import team_names
        from starter.load_smart_tagging import load as load_tagging
        events = load_tagging(GAME, args.data_dir)
        home, away = team_names(require(["atd_meta"], GAME, args.data_dir)[0])
        tag_teams = sorted({e["tags"].get("Team") for e in events} - {None})
        mapping = map_teams(tag_teams, home, away, parse_team_map(args.team_map))
        print("team mapping (ATD home={!r}, away={!r}): {}".format(
            home, away, {k: "H" if v == HOME else "A" for k, v in mapping.items()}))
        if args.list_codes:
            counts = {}
            for e in events:
                key = ((e["code"] or "").strip().upper(), "Team" in e["tags"])
                counts[key] = counts.get(key, 0) + 1
            team_counts = {}
            for e in events:
                v = e["tags"].get("Team")
                team_counts[v] = team_counts.get(v, 0) + 1
            print("\nTeam tag values (count): {}".format(team_counts))
            print("\n{:<34} {:>7} {:>10}".format("code", "events", "has Team"))
            for (code, has), n in sorted(counts.items(), key=lambda kv: -kv[1]):
                print("{:<34} {:>7} {:>10}".format(code, n, "yes" if has else "no"))
            return 0

    if not args.skip_tagging:
        run_tagging_check(table, evidence, dead, p, args, events, mapping)
    if not args.skip_hidden:
        run_hidden_check(table, evidence, dead, p, args)
    return 0


# --------------------------------------------------------------------------- self-test
def selftest():
    """Simulated match with known possessions and synthetic 'Smart Tagging' built from the truth
    (85% of phases tagged with the true team). Checks the machinery, not the real match."""
    import pandas as pd
    from .selftest import simulate

    table, truth = simulate(n=30000, seed=3)
    p = Params()
    ev = compute_evidence(table, p)
    rng = np.random.default_rng(0)

    events = []
    t = 200
    while t < len(table) - 900:
        length = int(rng.integers(100, 300))
        seg = truth[t:t + length]
        team = int(np.bincount(seg, minlength=2).argmax())
        if rng.random() > 0.85:
            team = 1 - team
        events.append({"code": "BUILD UP", "tags": {"Team": "Home FC" if team == HOME else "Away FC"},
                       "start_frame": t, "end_frame": t + length})
        t += length + int(rng.integers(20, 100))
    # set pieces: the team that has the ball right after the tag
    for s in range(1000, len(table) - 1000, 2500):
        team = int(truth[s + 60])
        events.append({"code": "SET PIECES", "tags": {"Team": "Home FC" if team == HOME else "Away FC",
                                                      "Type": "Throw-in"},
                       "start_frame": s, "end_frame": s + 40})

    mapping = map_teams(["Home FC", "Away FC"], "Home FC", "Away FC")
    rows = collect_events(events, mapping)
    good = viterbi_codes = run_one("viterbi", table, p, None, ev)
    res = tagging_agreement(good, rows, 5.0, 10.0)
    print("simulated tagging agreement (viterbi): phases {:.1f}% (control {:.1f}%), set pieces {:.1f}% "
          "(control {:.1f}%)".format(100 * res["phase"]["frame"], 100 * res["phase"]["control"],
                                     100 * res["set_piece"]["frame"], 100 * res["set_piece"]["control"]))
    ideal = tagging_agreement(truth.astype(np.int8), rows, 5.0, 10.0)      # upper bound: the TRUE labels
    print("  upper bound with the true labels: phases {:.1f}%, set pieces {:.1f}%".format(
        100 * ideal["phase"]["frame"], 100 * ideal["set_piece"]["frame"]))
    assert res["phase"]["frame"] > res["phase"]["control"] + 0.10, "agreement must clearly beat the control"
    assert res["phase"]["frame"] <= ideal["phase"]["frame"] + 0.03
    assert res["set_piece"]["frame"] > res["set_piece"]["control"] + 0.05

    # a decode of the OTHER team must score below its control (the sign check)
    flipped = np.where(good == HOME, AWAY, np.where(good == AWAY, HOME, good)).astype(np.int8)
    assert tagging_agreement(flipped, rows, 5.0, 10.0)["phase"]["frame"] < 0.35

    agree, n_eval, runs0, runs1 = hidden_ball(table, ev, None, "viterbi", p, 3.0, 0.25)
    print("hidden-ball (viterbi, 3 s blocks): agreement {:.1f}% on {} frames, possessions {} -> {}".format(
        100 * agree, n_eval, runs0, runs1))
    assert agree > 0.6

    try:
        map_teams(["Mystery FC"], "Home FC", "Away FC")
    except ValueError as err:
        assert "--team-map" in str(err)
    else:
        raise AssertionError("an unmappable team must raise")
    print("\nall tuning_checks self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())