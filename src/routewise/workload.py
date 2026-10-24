"""Seeded synthetic workload: requests share one of K long system prompts plus a unique suffix."""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    rid: int
    arrival: float
    blocks: tuple  # prompt as block ids (shared prefix, then unique suffix)
    out_tokens: int


def make_workload(
    seed: int,
    rate: float,
    duration: float,
    n_prompts: int = 8,
    prefix_blocks: int = 64,
    zipf_s: float = 1.0,
    suffix_range: tuple = (4, 16),
    out_range: tuple = (20, 60),
) -> list:
    """Poisson arrivals. Prompt popularity ~ Zipf(zipf_s); zipf_s=0 is uniform."""
    if rate <= 0 or duration <= 0 or n_prompts < 1:
        raise ValueError("rate, duration must be > 0 and n_prompts >= 1")
    weights = [1.0 / (k + 1) ** zipf_s for k in range(n_prompts)]
    rng = random.Random(f"{seed}:workload")
    reqs, t, rid = [], 0.0, 0
    while True:
        t += rng.expovariate(rate)
        if t >= duration:
            break
        p = rng.choices(range(n_prompts), weights)[0]
        suffix = rng.randint(*suffix_range)
        blocks = tuple(("sys", p, i) for i in range(prefix_blocks)) + tuple(
            ("req", rid, i) for i in range(suffix)
        )
        reqs.append(Request(rid, t, blocks, rng.randint(*out_range)))
        rid += 1
    return reqs
