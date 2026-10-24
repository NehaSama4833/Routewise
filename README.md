# Routewise

A small discrete-event simulator that compares **LLM request-routing policies** across several inference backends that keep a prefix (KV) cache. It answers a narrow question: *when does sending a request to the backend that already holds its prompt prefix help, and when does it backfire?*

**Status: one-day MVP, simulation only.** There is no real model or network in the loop. All numbers below are from a synthetic workload and a simple timing model.

## Problem and existing work
Many requests share a long prefix (system prompt, few-shot examples, chat history). A backend that already cached it skips most of the prefill, so routing by prefix can cut time-to-first-token (TTFT). Production systems already do this: SGLang's cache-aware load balancer, llm-d with the Gateway API Inference Extension, AMD's Infera. **This is not a new idea.** Routewise is a small, reproducible way to compare policies and see the failure mode of pure prefix affinity (a hot prefix overloading one backend). It does not replace those systems.

Sources: [SGLang v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4), [llm-d routing](https://docs.sglang.io/docs/advanced_features/llm-d), [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension), [KV-aware routing (AMD Infera)](https://rocm.docs.amd.com/projects/infera/en/latest/features/kv_aware_routing.html).

## Policies
| Policy | Rule |
|---|---|
| `round_robin` | cycle through backends |
| `least_loaded` | fewest running + queued requests |
| `prefix_greedy` | backend with the longest cached prefix match, ties by load; ignores load otherwise |
| `prefix_load_aware` | minimise estimated TTFT = queueing estimate + uncached prefill time |

## Model (what is and isn't simulated)
4 backends x 2 slots, FIFO queues, LRU prefix cache of 200 blocks of 16 tokens (evicting a prompt's tail before its head). Prefill 0.5 ms/token for uncached tokens, decode 20 ms/token. Workload: Poisson arrivals; each request is one of 8 shared 64-block system prompts (Zipf popularity) plus 4-16 unique blocks, and 20-60 output tokens. The timing constants are made up, not measured from any real server.

Not modelled: continuous batching, memory limits beyond the cache size, network, cache-transfer cost, delayed cache updates. The router sees exact cache contents and queue lengths, which is optimistic compared with real routers.

## Run
```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q                                              # 14 tests
.venv/bin/python -m routewise.experiment --seeds 5 --duration 300 --zipf 1.0   # about 2 s
```
Raw data and tables are in `results/`.

## Results
5 seeds, 300 simulated seconds each, Zipf 1.0 skew. TTFT p99 averaged over seeds, with the min..max across seeds. Hit rate = cached prompt blocks / all prompt blocks.

| rate (req/s) | policy | TTFT p50 | TTFT p99 (min..max) | hit rate | busiest backend |
|---|---|---|---|---|---|
| 3 | round_robin | 0.35 s | 0.82 s (0.77..0.92) | 43% | 25% |
| 3 | prefix_greedy | 0.12 s | 1.87 s (1.55..2.26) | 82% | 44% |
| 3 | prefix_load_aware | 0.12 s | 0.66 s (0.66..0.66) | 71% | 30% |
| 5 | round_robin | 0.57 s | 1.65 s (1.55..1.73) | 42% | 25% |
| 5 | least_loaded | 0.57 s | 1.64 s (1.53..1.72) | 43% | 29% |
| 5 | prefix_greedy | 1.40 s | 22.35 s (3.19..70.04) | 83% | 43% |
| 5 | prefix_load_aware | 0.14 s | 1.09 s (1.04..1.14) | 68% | 26% |
| 7 | round_robin | 4.37 s | 13.32 s (7.95..21.01) | 42% | 25% |
| 7 | least_loaded | 3.58 s | 10.20 s (5.74..14.96) | 43% | 26% |
| 7 | prefix_greedy | 1.15 s | 58.31 s (42.31..89.74) | 83% | 39% |
| 7 | prefix_load_aware | 0.71 s | 3.48 s (2.27..6.18) | 62% | 26% |

(Full table with p95 and `least_loaded` at rate 3: `results/experiment_zipf1.0.txt`.)

**What it shows, in this model:**
- Prefix-aware routing raises the cache hit rate from about 42% to 62-83%.
- **Greedy prefix affinity is dangerous under load.** It has the best p50 but by far the worst tail: at 5 req/s its p99 is 22 s (up to 70 s on one seed) because popular prefixes pile onto a few backends.
- The load-aware version keeps most of the cache benefit without that collapse: p99 1.09 s vs 1.65 s for round-robin at 5 req/s, and 3.48 s vs 13.32 s at 7 req/s.

**Sensitivity checks (rate 5 req/s only, 5 seeds):**

| Variation | round_robin p99 | prefix_greedy p99 | prefix_load_aware p99 |
|---|---|---|---|
| Uniform prompts (Zipf 0) | 1.94 s | 3.89 s | 1.24 s |
| Heavy skew (Zipf 2) | 1.16 s | 135.84 s | 0.90 s |
| Cache doubled to 400 blocks | 1.15 s | 34.34 s | 0.85 s |
| Router's service-time guess 0.5 s / 2.0 s (default 1.0 s) | 1.65 s | 22.35 s | 1.11 s / 1.34 s |

- The load-aware policy beat round-robin in every variation tested, but the margin depends on conditions. With a larger cache or heavy skew, round-robin already hits 65-66% of blocks, so the gain shrinks to about 20-25% on p99 (versus about 35% with uniform prompts, where round-robin hits only 29%).
- The load-aware policy's result moved by about 20% when its guessed service time changed (1.11 s to 1.34 s). That guess is a tunable parameter here, not something it learns.

## Limitations
- **Synthetic and simplified.** Real behaviour depends on batching, memory pressure, real prefill/decode costs and cache-update delay. Do not quote these numbers as real-system performance.
- **Optimistic router.** It sees exact cache state. Real routers estimate it, often with a delay.
- **Greedy prefix affinity looks worse here partly because it ignores load by construction.** Real implementations (e.g. SGLang's) include load-balancing thresholds, so this is a lower bound on how well real systems do, not a claim about them.
- **Only 5 seeds** and one cluster size (4 x 2). Tail percentiles are noisy; see the ranges.
- No tests compare the simulator with a real server.

## Future work
Approximate (router-side) cache tracking; real llama.cpp backends for validation; batching model; autoscaling interaction.
