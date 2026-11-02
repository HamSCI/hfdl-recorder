"""HFDL frame grid: nominal squitter slot starts on the UTC clock.

A squitter carries a 12-bit frame_index and a 4-bit frame_offset k.  The
frame index counts 32 s frames from 00:00 UTC, 2700 a day, so frame n starts
32 * n seconds after the day's midnight.  Slot k of a frame starts k * 32/13 s
later.  2700 * 32 = 86400, so the grid runs on without a seam at midnight.

Times are Unix seconds (UTC floats).  Every function here is pure: none reads
the host clock.  The frame base is integer arithmetic, and the slot offset is
added last, so the only rounding is in that final add.
"""

from __future__ import annotations

import math

FRAME_S = 32.0
SLOT_S = 32.0 / 13          # 13 slots per frame
FRAMES_PER_DAY = 2700

_FRAME_I = 32               # FRAME_S as an int, so frame bases stay exact
_DAY_I = 86400              # FRAMES_PER_DAY * _FRAME_I
_SLOTS_PER_FRAME = 13


def _check_index(frame_index: int) -> None:
    if not 0 <= frame_index < FRAMES_PER_DAY:
        raise ValueError(f"frame_index {frame_index} outside 0..{FRAMES_PER_DAY - 1}")


def _check_offset(frame_offset: int) -> None:
    if not 0 <= frame_offset < _SLOTS_PER_FRAME:
        raise ValueError(f"frame_offset {frame_offset} outside 0..{_SLOTS_PER_FRAME - 1}")


def slot_start_utc(frame_index: int, frame_offset: int, near_utc: float) -> float:
    """Nominal start of a squitter's slot, on the UTC day nearest near_utc.

    The day is chosen by the slot start itself: the candidates are the
    frame's start on the day before, the day of, and the day after near_utc,
    and the nearest one wins.
    """
    _check_index(frame_index)
    _check_offset(frame_offset)
    day = math.floor(near_utc / _DAY_I)
    best = None
    for d in (day - 1, day, day + 1):
        t = d * _DAY_I + frame_index * _FRAME_I + frame_offset * SLOT_S
        if best is None or abs(t - near_utc) < abs(best - near_utc):
            best = t
    return best


def next_slot_starts(frame_offset: int, after_utc: float, until_utc: float) -> list[float]:
    """Every nominal start of slot frame_offset in the span [after_utc, until_utc).

    after_utc is inclusive and until_utc is exclusive.  Starts are spaced
    exactly 32 s apart, across UTC midnight as well.
    """
    _check_offset(frame_offset)
    c = frame_offset * SLOT_S
    n = math.floor((after_utc - c) / _FRAME_I) - 1   # absolute frame number, one spare
    out = []
    while True:
        t = n * _FRAME_I + c
        if t >= until_utc:
            break
        if t >= after_utc:
            out.append(t)
        n += 1
    return out
