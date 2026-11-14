"""Segmenting the match and cleaning decoded labels."""

import numpy as np

from . import config

HOME, AWAY, DEAD = config.HOME, config.AWAY, config.DEAD


def segments(half, dead=None):
    """[(start, stop)) spans that are decoded independently: inside one half and with no dead ball.
    Nothing is carried across halftime, and a dead ball ends a possession, so the next span starts
    from scratch (the first possession after a restart is a fresh decision)."""
    key = np.where(half > 0, half, 0).astype(np.int64)
    if dead is not None:
        key = np.where(dead, 0, key)
    spans = []
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(key) != 0) + 1, [len(key)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if key[a] > 0:
            spans.append((int(a), int(b)))
    return spans


def fill_undecided(codes, carry):
    """-1 = the decoder had not decided yet. Back-fill with the first decision of the segment;
    a segment with no decision at all inherits the previous segment's last state."""
    codes = codes.copy()
    decided = np.flatnonzero(codes >= 0)
    if len(decided) == 0:
        codes[:] = carry
        return codes
    codes[:decided[0]] = codes[decided[0]]
    # forward-fill any interior holes (a decoder should not leave any, this is a safety net)
    idx = np.where(codes >= 0, np.arange(len(codes)), 0)
    np.maximum.accumulate(idx, out=idx)
    return codes[idx]


def merge_short_runs(codes, min_len):
    """Absorb runs shorter than `min_len` frames into their neighbours, shortest first.
    With two labels a short interior run is always flanked by the same label on both sides, so
    this simply deletes a blip. A short run at the edge joins its only neighbour."""
    if min_len <= 1 or len(codes) == 0:
        return codes
    change = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], change))
    lengths = list(np.diff(np.concatenate((starts, [len(codes)]))).astype(int))
    labels = [int(codes[s]) for s in starts]
    while len(lengths) > 1:
        i = int(np.argmin(lengths))
        if lengths[i] >= min_len:
            break
        if i == 0:
            lengths[1] += lengths[0]; del lengths[0]; del labels[0]
        elif i == len(lengths) - 1:
            lengths[i - 1] += lengths[i]; del lengths[i]; del labels[i]
        else:
            lengths[i - 1] += lengths[i] + lengths[i + 1]
            del lengths[i:i + 2]; del labels[i:i + 2]
    return np.repeat(np.array(labels, dtype=codes.dtype), lengths)


def decode_all(table, evidence, decode_segment, p, dead=None):
    """Run `decode_segment(evidence_slice) -> codes (-1 = undecided)` on every segment, fill,
    clean, and assemble the full-length code array (0 H, 1 A, 2 D)."""
    n = len(table)
    codes = np.full(n, DEAD, dtype=np.int8)
    half = table["half"].to_numpy()
    carry = HOME
    min_len = int(round(p.min_run_seconds * config.FPS))
    for a, b in segments(half, dead):
        seg = decode_segment(evidence[a:b])
        seg = fill_undecided(np.asarray(seg, dtype=np.int8), carry)
        seg = merge_short_runs(seg, min_len)
        codes[a:b] = seg
        carry = int(seg[-1])
    return codes
