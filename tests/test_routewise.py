import pytest

from routewise.cache import PrefixCache
from routewise.policies import POLICIES, least_loaded, round_robin
from routewise.sim import Backend, Config, simulate
from routewise.workload import Request, make_workload


def test_cache_prefix_match_counts_leading_blocks_only():
    c = PrefixCache(10)
    c.insert(["a", "b", "c"])
    assert c.match(["a", "b", "c", "d"]) == 3
    assert c.match(["a", "x", "c"]) == 1  # stops at first miss
    assert c.match(["z"]) == 0


def test_cache_evicts_tail_before_head():
    c = PrefixCache(3)
    c.insert(["a", "b", "c", "d"])  # 4 blocks into capacity 3: tail "d" goes
    assert c.match(["a", "b", "c"]) == 3
    assert c.match(["a", "b", "c", "d"]) == 3


def test_cache_lru_keeps_recently_used_prefix():
    c = PrefixCache(4)
    c.insert(["a", "b"])
    c.insert(["x", "y"])
    c.match(["a", "b"])  # match must not refresh recency
    c.insert(["a", "b"])  # explicit re-insert refreshes
    c.insert(["p", "q"])  # evicts x,y (oldest)
    assert c.match(["a", "b"]) == 2 and c.match(["x", "y"]) == 0


def test_cache_capacity_validation():
    with pytest.raises(ValueError):
        PrefixCache(0)


def test_workload_deterministic_and_prefix_sharing():
    a, b = make_workload(1, 4.0, 20.0), make_workload(1, 4.0, 20.0)
    assert a == b and make_workload(2, 4.0, 20.0) != a
    first = {}
    for r in a:
        sys_id = r.blocks[0]
        first.setdefault(sys_id, r.blocks[:64])
        assert r.blocks[:64] == first[sys_id]  # same system prompt -> identical prefix
    assert len({r.blocks[-1] for r in a}) == len(a)  # suffixes are unique per request


def test_round_robin_cycles_and_least_loaded_picks_min():
    cfg = Config(n_backends=3)
    bs = [Backend(i, cfg) for i in range(3)]
    state = {}
    assert [round_robin(bs, None, cfg, state) for _ in range(5)] == [0, 1, 2, 0, 1]
    bs[0].running = 2
    bs[1].running = 1
    assert least_loaded(bs, None, cfg, {}) == 2


def test_hand_computed_single_request():
    # 1 backend, 1 slot, 10 blocks x 16 tokens x 0.001 s = 0.16 s prefill; decode 0.02; 5 tokens
    cfg = Config(n_backends=1, slots=1, cache_blocks=100, prefill_s_per_token=0.001, decode_s=0.02)
    r = Request(0, 1.0, tuple(("b", i) for i in range(10)), 5)
    out = simulate([r], POLICIES["round_robin"], cfg)["records"][0]
    assert out["ttft"] == pytest.approx(0.16 + 0.02)
    assert out["e2e"] == pytest.approx(0.16 + 5 * 0.02)


def test_hand_computed_queueing_and_cache_hit():
    # Same prompt twice, 1 slot. 2nd waits for 1st and then hits the cache (no prefill).
    cfg = Config(n_backends=1, slots=1, cache_blocks=100, prefill_s_per_token=0.001, decode_s=0.1)
    blocks = tuple(("b", i) for i in range(10))
    r1, r2 = Request(0, 0.0, blocks, 2), Request(1, 0.0, blocks, 2)
    out = simulate([r1, r2], POLICIES["round_robin"], cfg)["records"]
    first_done = 0.16 + 2 * 0.1
    assert out[0]["ttft"] == pytest.approx(0.16 + 0.1)
    assert out[1]["matched"] == 10
    assert out[1]["ttft"] == pytest.approx(first_done + 0.1)  # waited, no prefill


@pytest.mark.parametrize("name", list(POLICIES))
def test_every_request_completes_exactly_once(name):
    cfg = Config()
    reqs = make_workload(3, 6.0, 60.0)
    out = simulate(reqs, POLICIES[name], cfg)["records"]
    assert sorted(r["rid"] for r in out) == [r.rid for r in reqs]
    assert all(r["ttft"] > 0 and r["e2e"] >= r["ttft"] for r in out)


def test_simulation_is_deterministic():
    cfg = Config()
    reqs = make_workload(4, 5.0, 60.0)
    a = simulate(reqs, POLICIES["prefix_load_aware"], cfg)["records"]
    b = simulate(reqs, POLICIES["prefix_load_aware"], cfg)["records"]
    assert a == b


def test_prefix_routing_raises_hit_rate_over_round_robin():
    cfg = Config()
    reqs = make_workload(0, 4.0, 200.0)
    rr = simulate(reqs, POLICIES["round_robin"], cfg)["hit_rate"]
    pg = simulate(reqs, POLICIES["prefix_greedy"], cfg)["hit_rate"]
    assert pg > rr
