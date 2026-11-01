import json
import shutil
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from config import (TRAFFIC_CSV, MODEL_PATH, SCALER_PATH, ORIGINAL_MODEL_PATH, RESULTS_DIR,
                    FEATURES, SLICES, WINDOW, TRAIN_SPLIT, VAL_SPLIT, BATCH, LR,
                    MAX_EPOCHS, PATIENCE, SEED)
from model import TrafficLSTM, make_windows, device, predict, inverse_loads


def fit(model, Xtr, ytr, Xval, yval, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE, log=print):
    dev = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xtr_t, ytr_t = torch.from_numpy(Xtr).to(dev), torch.from_numpy(ytr).to(dev)
    Xval_t, yval_t = torch.from_numpy(Xval).to(dev), torch.from_numpy(yval).to(dev)

    hist = {"train": [], "val": []}
    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xval_t), yval_t).item()
        hist["train"].append(tot / len(Xtr_t))
        hist["val"].append(val)
        log(f"Епоха {ep:3d}  train={hist['train'][-1]:.5f}  val={val:.5f}")
        if val < best - 1e-6:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"Early stopping: {patience} епохи без подобрение.")
                break
    model.load_state_dict(best_state)
    return hist


def metrics(y_true, y_pred):
    out = {}
    for i, s in enumerate(SLICES):
        e = y_pred[:, i] - y_true[:, i]
        out[s] = {"MAE": float(np.mean(np.abs(e))),
                  "RMSE": float(np.sqrt(np.mean(e ** 2))),
                  "MAPE_%": float(np.mean(np.abs(e) / np.maximum(y_true[:, i], 1)) * 100)}
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    df = pd.read_csv(TRAFFIC_CSV, parse_dates=["timestamp"])
    data = df[FEATURES].values

    split = int(len(data) * TRAIN_SPLIT)
    train_raw, test_raw = data[:split], data[split - WINDOW:]

    scaler = MinMaxScaler().fit(train_raw)
    Xtr_all, ytr_all = make_windows(scaler.transform(train_raw))
    Xte, yte = make_windows(scaler.transform(test_raw))

    v = int(len(Xtr_all) * (1 - VAL_SPLIT))
    Xtr, ytr, Xval, yval = Xtr_all[:v], ytr_all[:v], Xtr_all[v:], ytr_all[v:]
    print(f"Train: {len(Xtr)}  Val: {len(Xval)}  Test: {len(Xte)} прозореца, устройство: {device()}")

    model = TrafficLSTM().to(device())
    t0 = time.time()
    hist = fit(model, Xtr, ytr, Xval, yval)
    train_sec = time.time() - t0

    torch.save(model.state_dict(), MODEL_PATH)
    shutil.copy(MODEL_PATH, ORIGINAL_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    y_true = inverse_loads(scaler, yte)
    y_pred = inverse_loads(scaler, predict(model, Xte))
    y_naive = test_raw[WINDOW - 1:-1, :len(SLICES)]

    res = {"epochs": len(hist["train"]), "train_seconds": round(train_sec, 1),
           "lstm": metrics(y_true, y_pred), "naive_persistence": metrics(y_true, y_naive)}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame({"timestamp": df["timestamp"].values[split:]})
    for i, s in enumerate(SLICES):
        pred_df[f"{s}_actual"] = y_true[:, i]
        pred_df[f"{s}_pred"] = y_pred[:, i]
    pred_df.to_csv(RESULTS_DIR / "predictions_test.csv", index=False)
    pd.DataFrame(hist).to_csv(RESULTS_DIR / "loss_history.csv", index_label="epoch")

    plt.figure(figsize=(7, 4))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="validation")
    plt.xlabel("епоха"); plt.ylabel("MSE (нормализирано)"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Крива на загубата")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "loss.png", dpi=120)

    print("\nМетрики върху теста (PRB):")
    for s in SLICES:
        l, n = res["lstm"][s], res["naive_persistence"][s]
        print(f"  {s:6s} LSTM MAE={l['MAE']:.2f} RMSE={l['RMSE']:.2f} | "
              f"наивен MAE={n['MAE']:.2f} RMSE={n['RMSE']:.2f}")


if __name__ == "__main__":
    main()
