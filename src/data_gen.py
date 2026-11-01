import numpy as np
import pandas as pd

from config import DAYS, STEP_MIN, STEPS_PER_DAY, TRAFFIC_CSV, SEED


def _bump(hour, center, width):
    d = np.minimum(np.abs(hour - center), 24 - np.abs(hour - center))
    return np.exp(-0.5 * (d / width) ** 2)


def generate(days=DAYS, seed=SEED, start_day=0, growth=1.0):
    rng = np.random.default_rng(seed)
    n = days * STEPS_PER_DAY
    t = np.arange(n) + start_day * STEPS_PER_DAY
    hour = (t % STEPS_PER_DAY) * STEP_MIN / 60.0
    dow = (t // STEPS_PER_DAY) % 7
    weekend = dow >= 5

    embb = 12 + 18 * _bump(hour, 13, 3) + 30 * _bump(hour, 21, 2.2)
    embb *= np.where(weekend, 1.15, 1.0)
    embb += rng.normal(0, 2.5, n)
    for _ in range(max(1, days // 3)):
        s = rng.integers(0, n - 30)
        L = rng.integers(6, 25)
        embb[s:s + L] += rng.uniform(15, 30) * np.hanning(L)

    work = (~weekend) & (hour >= 7) & (hour < 19)
    urllc = np.where(work, 18.0, 4.0) + rng.normal(0, 1.5, n)
    for _ in range(days * 2):
        s = rng.integers(0, n - 4)
        urllc[s:s + rng.integers(1, 4)] += rng.uniform(5, 12)

    mmtc = 14 + 3 * _bump(hour, 12, 5) + rng.normal(0, 1.2, n)
    mmtc[(t % 12) == 0] += 12

    loads = np.vstack([embb, urllc, mmtc]).T * np.asarray(growth).reshape(-1, 1)
    loads = np.clip(loads, 0.5, None)

    users = 80 + 9 * loads[:, 0] + 4 * loads[:, 1] + rng.normal(0, 25, n)
    users = np.clip(users, 10, None).round()

    ts = pd.Timestamp("2026-01-05") + pd.to_timedelta(t * STEP_MIN, unit="min")
    return pd.DataFrame({"timestamp": ts, "eMBB": loads[:, 0], "URLLC": loads[:, 1],
                         "mMTC": loads[:, 2], "users": users})


if __name__ == "__main__":
    df = generate()
    df.to_csv(TRAFFIC_CSV, index=False)
    total = df[["eMBB", "URLLC", "mMTC"]].sum(axis=1)
    print(f"Записани {len(df)} реда в {TRAFFIC_CSV}")
    print(df.describe().round(2))
    print(f"Общо търсене: средно {total.mean():.1f} PRB, макс {total.max():.1f} PRB, "
          f"над 100 PRB в {(total > 100).mean():.1%} от времето")
