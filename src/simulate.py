import numpy as np
import pandas as pd

from config import (RESULTS_DIR, SLICES, POOL_PRB, PRB_PER_REQ, BASE_LATENCY_MS,
                    ROBUST_SEEDS, ROBUST_POOLS)

METHODS = ["Round-Robin", "Proportional Fair", "LSTM-proactive"]
PRB_REQ = np.array([PRB_PER_REQ[s] for s in SLICES])
BASE_LAT = np.array([BASE_LATENCY_MS[s] for s in SLICES])


def load_test_series():
    df = pd.read_csv(RESULTS_DIR / "predictions_test.csv", parse_dates=["timestamp"])
    actual = df[[f"{s}_actual" for s in SLICES]].values
    forecast = np.clip(df[[f"{s}_pred" for s in SLICES]].values, 0.1, None)
    return df["timestamp"], actual, forecast


def proportional(weights, pool):
    w = np.clip(weights, 1e-6, None)
    return pool * w / w.sum(axis=1, keepdims=True)


def allocate(method, actual, forecast, pool):
    T = len(actual)
    if method == "Round-Robin":
        return np.full((T, 3), pool / 3)
    if method == "Proportional Fair":
        last = np.vstack([actual[:1], actual[:-1]])
        return proportional(last, pool)
    if method == "LSTM-proactive":
        return proportional(forecast, pool)
    raise ValueError(method)


def run(method, actual, forecast, pool=POOL_PRB, seed=0):
    rng = np.random.default_rng(seed)
    alloc = allocate(method, actual, forecast, pool)

    requests = rng.poisson(actual / PRB_REQ)
    capacity = np.floor(alloc / PRB_REQ)
    served = np.minimum(requests, capacity)
    blocked = requests - served
    used_prb = served * PRB_REQ
    rho = np.where(alloc > 0, used_prb / alloc, 0)
    latency = BASE_LAT / (1 - np.minimum(rho, 0.95))

    sat = np.where(requests > 0, served / np.maximum(requests, 1), 1.0)
    jain = sat.sum(1) ** 2 / (3 * (sat ** 2).sum(1))

    summary = {
        "method": method, "pool": pool, "seed": seed,
        "utilization_%": 100 * used_prb.sum() / (pool * len(actual)),
        "blocking_%": 100 * blocked.sum() / max(requests.sum(), 1),
        "latency_ms": float((latency * served).sum() / max(served.sum(), 1)),
        "jain": float(jain.mean()),
    }
    for i, s in enumerate(SLICES):
        summary[f"blocking_{s}_%"] = 100 * blocked[:, i].sum() / max(requests[:, i].sum(), 1)
        summary[f"latency_{s}_ms"] = float((latency[:, i] * served[:, i]).sum()
                                           / max(served[:, i].sum(), 1))
    trace = {"alloc": alloc, "requests": requests, "served": served,
             "blocked": blocked, "latency": latency, "jain": jain}
    return summary, trace


def compare(pool=POOL_PRB, seed=0):
    _, actual, forecast = load_test_series()
    return pd.DataFrame([run(m, actual, forecast, pool, seed)[0] for m in METHODS])


def robustness(seeds=ROBUST_SEEDS, pools=ROBUST_POOLS):
    _, actual, forecast = load_test_series()
    rows = [run(m, actual, forecast, p, s)[0] for p in pools for s in seeds for m in METHODS]
    return pd.DataFrame(rows)


def main():
    cmp = compare()
    cmp.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    print(f"Сравнение при {POOL_PRB} PRB, seed 0:")
    print(cmp[["method", "utilization_%", "blocking_%", "latency_ms", "jain"]].round(3).to_string(index=False))

    rob = robustness()
    rob.to_csv(RESULTS_DIR / "robustness.csv", index=False)
    agg = (rob.groupby(["pool", "method"])[["blocking_%", "latency_ms", "utilization_%", "jain"]]
              .agg(["mean", "std"]).round(3))
    agg.to_csv(RESULTS_DIR / "robustness_summary.csv")
    print(f"\nУстойчивост ({len(ROBUST_SEEDS)} seed-а x пулове {ROBUST_POOLS}):")
    print(agg.to_string())

    for p in ROBUST_POOLS:
        r = rob[rob.pool == p].pivot(index="seed", columns="method", values="blocking_%")
        wins = (r["LSTM-proactive"] < r["Proportional Fair"]).sum()
        print(f"  {p} PRB: LSTM < PF по блокиране в {wins}/{len(r)} seed-а")


if __name__ == "__main__":
    main()
