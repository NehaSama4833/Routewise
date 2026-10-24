# Routewise MVP (Day 2), defined before coding

Problem: show when prefix-aware routing helps vs hurts, reproducibly, without GPUs.
MVP done = (1) per-backend LRU prefix cache, (2) seeded shared-prefix workload, (3) DES with queueing,
(4) four policies, (5) experiment over seeds and rates with tables, (6) tests incl. hand-computed cases,
(7) README stating limitations and that the idea is not new.
Hypothesis: pure prefix affinity gives the best hit rate but can have the worst tail latency under load;
a load-aware variant should avoid that. (Result: supported in this model; see README.)
Out of scope: real model/network, batching model, approximate cache tracking.
