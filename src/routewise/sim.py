"""Discrete-event simulation of N backends behind a router.

Each backend has `slots` concurrent service slots, a FIFO queue and a prefix cache.
Service time of a request that starts with `matched` cached blocks:
    prefill = (len(blocks) - matched) * block_tokens * prefill_s_per_token
    ttft_service = prefill + decode_s          (first token after one decode step)
    total = prefill + out_tokens * decode_s
TTFT (what we report) = queue wait + ttft_service. Batching effects are NOT modelled.
The prompt's blocks enter the cache when service starts (simplification).
The router sees exact cache contents and queue lengths (an optimistic assumption).
"""

import heapq
from collections import deque
from dataclasses import dataclass

from routewise.cache import PrefixCache


@dataclass(frozen=True)
class Config:
    n_backends: int = 4
    slots: int = 2
    cache_blocks: int = 200
    block_tokens: int = 16
    prefill_s_per_token: float = 0.0005
    decode_s: float = 0.02
    est_service_s: float = 1.0  # router's guess of mean service time, used by load-aware policy


class Backend:
    def __init__(self, idx: int, cfg: Config):
        self.idx, self.cfg = idx, cfg
        self.cache = PrefixCache(cfg.cache_blocks)
        self.queue: deque = deque()
        self.running = 0

    @property
    def load(self) -> int:
        return self.running + len(self.queue)


def simulate(requests: list, policy, cfg: Config) -> dict:
    """Run to completion. `policy(backends, request, cfg, state) -> backend index`."""
    backends = [Backend(i, cfg) for i in range(cfg.n_backends)]
    state: dict = {}
    events: list = []  # (time, seq, kind, payload)
    seq = 0
    for r in requests:
        heapq.heappush(events, (r.arrival, seq, "arrive", r))
        seq += 1
    records, hits, total_blocks = [], 0, 0

    def try_start(b: Backend, now: float) -> None:
        nonlocal seq, hits, total_blocks
        while b.running < cfg.slots and b.queue:
            r = b.queue.popleft()
            matched = b.cache.match(r.blocks)
            b.cache.insert(list(r.blocks))
            prefill = (len(r.blocks) - matched) * cfg.block_tokens * cfg.prefill_s_per_token
            ttft = (now - r.arrival) + prefill + cfg.decode_s
            done = now + prefill + r.out_tokens * cfg.decode_s
            hits += matched
            total_blocks += len(r.blocks)
            records.append({"rid": r.rid, "backend": b.idx, "ttft": ttft, "e2e": done - r.arrival,
                            "matched": matched, "blocks": len(r.blocks), "done": done})
            b.running += 1
            heapq.heappush(events, (done, seq, "done", b))
            seq += 1

    now = 0.0
    while events:
        now, _, kind, payload = heapq.heappop(events)
        if kind == "arrive":
            idx = policy(backends, payload, cfg, state)
            backends[idx].queue.append(payload)
            try_start(backends[idx], now)
        else:
            payload.running -= 1
            try_start(payload, now)
    return {"records": records, "hit_rate": hits / total_blocks if total_blocks else 0.0,
            "makespan": now}
