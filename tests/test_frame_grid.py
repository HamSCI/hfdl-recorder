import calendar

import pytest

from hfdl_recorder.timing import frame_grid as fg

DAY_S = 86400
SLOT = 32 / 13


def _epoch(y, mo, d, h=0, mi=0, s=0):
    return float(calendar.timegm((y, mo, d, h, mi, s)))


def test_constants():
    assert fg.FRAME_S == 32.0
    assert fg.SLOT_S == 32.0 / 13
    assert 2700 * fg.FRAME_S == DAY_S


def test_riverhead_on_nd():
    # Frame 2280 is 20:16:00 UTC; slot 5 adds 5 * 32/13 s.
    got = fg.slot_start_utc(2280, 5, _epoch(2026, 11, 1, 20, 16, 12))
    assert got == pytest.approx(_epoch(2026, 11, 1, 20, 16, 0) + 5 * SLOT, abs=1e-9)


def test_midnight_picks_the_right_day():
    # Frame 2699 is 23:59:28 UTC on the day before; 00:00:20 on 11-02 is nearest to it.
    got = fg.slot_start_utc(2699, 12, _epoch(2026, 11, 2, 0, 0, 20))
    assert got == pytest.approx(_epoch(2026, 11, 1, 23, 59, 28) + 12 * SLOT, abs=1e-9)


def test_near_picks_the_nearest_day_both_ways():
    # Frame 0 starts at 00:00:00 every day. Near 23:59:50 on 11-01, the 11-02
    # start is 10 s away and the 11-01 start is 86390 s away, so 11-02 wins.
    assert fg.slot_start_utc(0, 0, _epoch(2026, 11, 1, 23, 59, 50)) == _epoch(2026, 11, 2)
    assert fg.slot_start_utc(0, 0, _epoch(2026, 11, 2, 0, 0, 5)) == _epoch(2026, 11, 2)


def test_frame_index_range_is_validated():
    near = _epoch(2026, 11, 1, 12)
    with pytest.raises(ValueError):
        fg.slot_start_utc(-1, 0, near)
    with pytest.raises(ValueError):
        fg.slot_start_utc(2700, 0, near)
    assert fg.slot_start_utc(2699, 0, near) is not None


def test_frame_offset_range_is_validated():
    near = _epoch(2026, 11, 1, 12)
    with pytest.raises(ValueError):
        fg.slot_start_utc(0, -1, near)
    with pytest.raises(ValueError):
        fg.slot_start_utc(0, 13, near)
    with pytest.raises(ValueError):
        fg.next_slot_starts(13, near, near + 100)


def test_next_slot_starts_spacing():
    starts = fg.next_slot_starts(3, _epoch(2026, 11, 1, 12), _epoch(2026, 11, 1, 13))
    assert len(starts) in (112, 113)     # one hour holds 112.5 frames
    diffs = [b - a for a, b in zip(starts, starts[1:])]
    assert all(d == 32.0 for d in diffs)


def test_next_slot_starts_across_midnight_is_continuous():
    starts = fg.next_slot_starts(12, _epoch(2026, 11, 1, 23, 59), _epoch(2026, 11, 2, 0, 1))
    diffs = [b - a for a, b in zip(starts, starts[1:])]
    assert diffs and all(d == 32.0 for d in diffs)
    # The start that straddles midnight is the 2699 -> 0 transition.
    assert any(abs(t - _epoch(2026, 11, 2) - 12 * SLOT) < 1e-6 for t in starts)
    assert any(abs(t - (_epoch(2026, 11, 1, 23, 59, 28) + 12 * SLOT)) < 1e-6 for t in starts)


def test_next_slot_starts_boundaries_half_open():
    # after is inclusive, until is exclusive.
    a = _epoch(2026, 11, 2, 0, 0, 0)
    starts = fg.next_slot_starts(0, a, a + 64)
    assert starts == [pytest.approx(a), pytest.approx(a + 32)]
    assert fg.next_slot_starts(0, a + 1, a + 32) == []
    assert fg.next_slot_starts(0, a, a) == []


def test_next_slot_starts_matches_slot_start_utc():
    after = _epoch(2026, 11, 2, 10)
    starts = fg.next_slot_starts(7, after, after + 200)
    for t in starts:
        frame = int(((t - 7 * SLOT) % DAY_S) // 32)
        assert fg.slot_start_utc(frame, 7, t) == pytest.approx(t, abs=1e-6)
