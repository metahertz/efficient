import pytest

from efficient.daemon import cache_guardian as cg


@pytest.fixture(autouse=True)
def fresh_state():
    cg.reset()
    yield
    cg.reset()


def _usage(read=0, creation=0, inp=0):
    return {"cache_read_input_tokens": read,
            "cache_creation_input_tokens": creation,
            "input_tokens": inp}


BASE = {"model": "m", "tools": [{"name": "t1"}], "system": [{"text": "s"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}


def test_first_request_no_findings():
    r = cg.analyze("s1", b"body", BASE, _usage(read=100, inp=50))
    assert r["invalidator"] is None
    assert r["cache_read_ratio"] == round(100 / 150, 4)
    assert r["content_blocks"] == 1


def test_tools_change_detected():
    cg.analyze("s1", b"a", BASE, _usage(read=100, inp=10))
    changed = {**BASE, "tools": [{"name": "t1"}, {"name": "t2"}]}
    r = cg.analyze("s1", b"ab", changed, _usage(read=0, inp=5000))
    assert r["invalidator"] == "tools_changed"
    assert "tool set changed" in r["alert"]


def test_system_change_detected():
    cg.analyze("s1", b"a", BASE, _usage(read=100, inp=10))
    changed = {**BASE, "system": [{"text": "different"}]}
    r = cg.analyze("s1", b"ab", changed, _usage(read=0, inp=5000))
    assert r["invalidator"] == "system_changed"


def test_cache_cold_detected():
    cg.analyze("s1", b"a", BASE, _usage(read=5000, inp=10))
    r = cg.analyze("s1", b"ab", BASE, _usage(read=0, creation=3000, inp=2000))
    assert r["invalidator"] == "cache_cold"
    assert "5000" in r["alert"] or "went cold" in r["alert"]


def test_cold_ignored_below_min_tokens():
    cg.analyze("s1", b"a", BASE, _usage(read=500, inp=10))
    r = cg.analyze("s1", b"ab", BASE, _usage(read=0, inp=100))
    assert r["invalidator"] is None


def test_lookback_risk_detected():
    cg.analyze("s1", b"a", BASE, _usage(read=100, inp=10))
    big = {**BASE, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "x"}] * 25},
    ]}
    r = cg.analyze("s1", b"ab", big, _usage(read=100, inp=10))
    assert r["invalidator"] == "lookback_risk"


def test_prefix_overlap_computed():
    cg.analyze("s1", b"aaaa-bbbb", BASE, _usage(read=1, inp=1))
    r = cg.analyze("s1", b"aaaa-cccc-dddd", BASE, _usage(read=1, inp=1))
    assert r["prefix_overlap"] == round(5 / 9, 4)


def test_sessions_isolated():
    cg.analyze("s1", b"a", BASE, _usage(read=100, inp=10))
    changed = {**BASE, "tools": []}
    r = cg.analyze("s2", b"b", changed, _usage(read=0, inp=5000))
    assert r["invalidator"] is None  # first request of s2, nothing to compare


def test_session_lru_cap():
    for i in range(cg._MAX_SESSIONS + 10):
        cg.analyze(f"s{i}", b"a", BASE, _usage(inp=1))
    assert len(cg._sessions) == cg._MAX_SESSIONS
