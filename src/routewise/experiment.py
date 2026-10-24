"""Compare policies over seeds and arrival rates.

    python -m routewise.experiment --seeds 5 --duration 300 --zipf 1.0
"""

import argparse
import json
import math
import platform
from pathlib import Path

from routewise.policies import POLICIES
from routewise.sim import Config, simulate
from routewise.workload import make_workload


def percentile(values, p):
    s = sorted(values)
    return s[max(1, math.ceil(p / 100 * len(s))) - 1]


def run(seeds, duration, rates, zipf, cfg):
    rows = []
    for rate in rates:
        for name, pol in POLICIES.items():
            per_seed = []
            for seed in range(seeds):
                reqs = make_workload(seed, rate, duration, zipf_s=zipf)
                out = simulate(reqs, pol, cfg)
                t = [r["ttft"] for r in out["records"]]
                backs = [r["backend"] for r in out["records"]]
                share = max(backs.count(i) for i in range(cfg.n_backends)) / len(backs)
                per_seed.append({"n": len(t), "p50": percentile(t, 50), "p95": percentile(t, 95),
                                 "p99": percentile(t, 99), "hit": out["hit_rate"],
                                 "max_share": share})
            rows.append({"rate": rate, "policy": name, "seeds": per_seed})
    return rows


def print_table(rows):
    print("| rate (req/s) | policy | TTFT p50 | TTFT p95 | TTFT p99 (min..max over seeds) | "
          "cache hit rate | busiest backend share |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        s = r["seeds"]
        a = lambda k: sum(x[k] for x in s) / len(s)  # noqa: E731
        p99s = [x["p99"] for x in s]
        print(f"| {r['rate']:g} | {r['policy']} | {a('p50'):.2f}s | {a('p95'):.2f}s | "
              f"{a('p99'):.2f}s ({min(p99s):.2f}..{max(p99s):.2f}) | {a('hit'):.0%} | "
              f"{a('max_share'):.0%} |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--rates", type=float, nargs="+", default=[3.0, 5.0, 7.0])
    p.add_argument("--zipf", type=float, default=1.0)
    p.add_argument("--est-service", type=float, default=1.0)
    p.add_argument("--cache-blocks", type=int, default=200)
    p.add_argument("--out", default="results/experiment.json")
    a = p.parse_args()
    cfg = Config(est_service_s=a.est_service, cache_blocks=a.cache_blocks)
    rows = run(a.seeds, a.duration, a.rates, a.zipf, cfg)
    meta = {"platform": platform.platform(), "python": platform.python_version(),
            "seeds": a.seeds, "duration_s": a.duration, "zipf": a.zipf, "config": cfg.__dict__,
            "workload": "synthetic: 8 shared 64-block prompts + 4-16 unique blocks, Poisson"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    print_table(rows)


if __name__ == "__main__":
    main()
