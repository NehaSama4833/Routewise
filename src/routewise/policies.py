"""Routing policies. Each returns the index of the chosen backend.

round_robin       cycle through backends; ignores everything else.
least_loaded      fewest running+queued requests (ties: lowest index).
prefix_greedy     longest cached prefix match; ties -> least loaded. Ignores load otherwise,
                  so a hot prefix can pile onto one backend.
prefix_load_aware minimise ESTIMATED TTFT = (load / slots) * est_service_s + uncached prefill time.
                  Trades cache reuse against queueing; est_service_s is a guess, not a measurement.
"""


def round_robin(backends, req, cfg, state):
    i = state.get("rr", 0)
    state["rr"] = (i + 1) % len(backends)
    return i


def least_loaded(backends, req, cfg, state):
    return min(backends, key=lambda b: (b.load, b.idx)).idx


def prefix_greedy(backends, req, cfg, state):
    return min(backends, key=lambda b: (-b.cache.match(req.blocks), b.load, b.idx)).idx


def prefix_load_aware(backends, req, cfg, state):
    def est_ttft(b):
        uncached = len(req.blocks) - b.cache.match(req.blocks)
        prefill = uncached * cfg.block_tokens * cfg.prefill_s_per_token
        return (b.load / cfg.slots) * cfg.est_service_s + prefill

    return min(backends, key=lambda b: (est_ttft(b), b.idx)).idx


POLICIES = {
    "round_robin": round_robin,
    "least_loaded": least_loaded,
    "prefix_greedy": prefix_greedy,
    "prefix_load_aware": prefix_load_aware,
}
