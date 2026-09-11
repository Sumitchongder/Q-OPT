#!/usr/bin/env python3
"""
Run benchmark_live() multiple times (as independent trials) and produce a
mean+-std summary table -- more defensible for a paper than a single run,
since sub-millisecond crypto timing (especially ML-DSA's rejection-sampling
based signing) has real run-to-run variance.

Run:
    python scripts/summarize_pqc_benchmark.py --n-runs 3 --n-iters 2000
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from qopt.pqc.profiles import benchmark_live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=3, help="Independent benchmark_live() calls")
    ap.add_argument("--n-iters", type=int, default=2000, help="Iterations per call")
    ap.add_argument("--out", default="tables/pqc_benchmark_measured_summary.csv")
    args = ap.parse_args()

    all_runs = {}
    for i in range(args.n_runs):
        print(f"Run {i+1}/{args.n_runs} (n_iters={args.n_iters})...")
        profiles = benchmark_live(n_iters=args.n_iters)
        for alg, p in profiles.items():
            all_runs.setdefault(alg, {"keygen": [], "op1": [], "op2": []})
            all_runs[alg]["keygen"].append(p.keygen_latency_ms)
            all_runs[alg]["op1"].append(p.encapsulation_latency_ms)
            all_runs[alg]["op2"].append(p.decapsulation_latency_ms)

    rows = []
    for alg, d in all_runs.items():
        is_kem = "KEM" in alg
        rows.append(dict(
            algorithm=alg,
            operation_pair="encap/decap" if is_kem else "sign/verify",
            keygen_mean_ms=round(float(np.mean(d["keygen"])), 4),
            keygen_std_ms=round(float(np.std(d["keygen"], ddof=1)) if len(d["keygen"]) > 1 else 0.0, 4),
            op1_mean_ms=round(float(np.mean(d["op1"])), 4),
            op1_std_ms=round(float(np.std(d["op1"], ddof=1)) if len(d["op1"]) > 1 else 0.0, 4),
            op2_mean_ms=round(float(np.mean(d["op2"])), 4),
            op2_std_ms=round(float(np.std(d["op2"], ddof=1)) if len(d["op2"]) > 1 else 0.0, 4),
            n_independent_runs=len(d["keygen"]), n_iters_per_run=args.n_iters,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")
    print(df.to_string(index=False))
    print("\nNote for Table 2 caption: op1=sign, op2=verify for ML-DSA; "
          "op1=encapsulation, op2=decapsulation for ML-KEM.")


if __name__ == "__main__":
    main()
