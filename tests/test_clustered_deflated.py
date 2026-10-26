"""`stats.clustered_interval_or_deflated`: the clustered interval where it exists, otherwise
Wilson on a DEFLATED effective sample, never on the raw total. The trigger-check verdict rests on
this (tests/test_trigger.py pins the end-to-end numbers); these pin the helper itself."""

import pytest

from iago.stats import clustered_interval, clustered_interval_or_deflated, wilson_interval


def test_where_the_clustered_interval_exists_it_is_returned_unchanged():
    clusters = [(1, 3)] * 7 + [(0, 3)]
    assert clustered_interval_or_deflated(clusters) == clustered_interval(clusters)
    assert clustered_interval_or_deflated(clusters, fallback_n_eff=3.0) == clustered_interval(clusters)


def test_one_cluster_deflates_to_one_draw_not_twenty_four():
    res = clustered_interval_or_deflated([(16, 24)])
    assert res.reason and "1 cluster" in res.reason
    assert res.n_eff == 1 and res.deff == 24
    assert res.interval == pytest.approx(wilson_interval(16 / 24, 1))
    # Never the raw total: the deflated interval is wider on both sides.
    raw = wilson_interval(16, 24)
    assert res.interval[0] < raw[0] and res.interval[1] > raw[1]


def test_a_degenerate_rate_uses_the_callers_n_eff_when_given_else_the_cluster_count():
    zero = [(0, 3)] * 8
    borrowed = clustered_interval_or_deflated(zero, fallback_n_eff=24.0)
    assert borrowed.reason and "0%" in borrowed.reason
    assert borrowed.n_eff == 24.0 and borrowed.interval == pytest.approx(wilson_interval(0, 24))
    own = clustered_interval_or_deflated(zero)
    assert own.n_eff == 8 and own.interval == pytest.approx(wilson_interval(0, 8))
    full = clustered_interval_or_deflated([(12, 12), (12, 12)])
    assert full.reason and "100%" in full.reason
    assert full.n_eff == 2 and full.interval == pytest.approx(wilson_interval(2, 2))


def test_no_trials_is_the_uninformative_interval():
    res = clustered_interval_or_deflated([])
    assert res.interval == (0.0, 1.0) and res.n_eff is None and res.reason == "no trials"
