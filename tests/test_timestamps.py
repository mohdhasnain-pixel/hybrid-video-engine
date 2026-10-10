"""Timeline resampling must be exact: no drift, correct frame counts."""

from fractions import Fraction

import numpy as np
import pytest

from video_engine.interpolation import TimelineResampler


def frames(n):
    # each frame encodes its index so we can see what the resampler emitted
    return [np.full((8, 8, 3), i, np.uint8) for i in range(n)]


def mid(a, b, t):
    return np.full_like(a, 200)  # marker for "synthesised"


@pytest.mark.parametrize("src,dst,n,expected", [(30, 60, 30, 60), (24, 60, 24, 60), (25, 60, 25, 60),
                                                (50, 60, 50, 60), (30000 / 1001, 60000 / 1001, 30, 60)])
def test_output_count_matches_duration(src, dst, n, expected):
    r = TimelineResampler(Fraction(src).limit_denominator(1001), Fraction(dst).limit_denominator(1001), mid, 1.0)
    out = list(r(frames(n)))
    assert len(out) == expected


def test_30_to_60_alternates_real_and_generated():
    r = TimelineResampler(Fraction(30), Fraction(60), mid, 1.0)
    out = [int(f[0, 0, 0]) for f in r(frames(4))]
    assert out == [0, 200, 1, 200, 2, 200, 3, 3]  # last frame held for its full interval
    assert r.stats()["generated"] == 3


def test_24_to_60_timesteps():
    seen = []
    r = TimelineResampler(Fraction(24), Fraction(60), lambda a, b, t: seen.append(round(t, 3)) or a, 1.0)
    list(r(frames(3)))
    assert seen[:4] == [0.4, 0.8, 0.2, 0.6]


def test_scene_cut_is_not_morphed():
    a = np.zeros((72, 128, 3), np.uint8)
    b = np.full((72, 128, 3), 255, np.uint8)
    called = []
    r = TimelineResampler(Fraction(30), Fraction(60), lambda *x: called.append(1) or a, 0.12)
    out = list(r([a, b]))
    assert not called and r.stats()["scene_cut_holds"] == 1
    assert len(out) == 4


def test_pairs_use_view_for_cuts_and_out_for_frames():
    src = frames(3)
    pairs = [(f, f * 2) for f in src]
    r = TimelineResampler(Fraction(30), Fraction(60), lambda a, b, t: np.full_like(a[1], 99), 1.0,
                          view=lambda p: p[0], out=lambda p: p[1])
    out = [int(f[0, 0, 0]) for f in r(pairs)]
    assert out == [0, 99, 2, 99, 4, 4]
