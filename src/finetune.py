import json
import shutil
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch

from config import (MODEL_PATH, SCALER_PATH, ORIGINAL_MODEL_PATH, HISTORY_DIR, FEATURES,
                    SLICES, DAYS, STEPS_PER_DAY, VAL_SPLIT)
from data_gen import generate
from model import TrafficLSTM, make_windows, device, predict, inverse_loads
from train import fit

HISTORY_JSON = HISTORY_DIR / "history.json"


def load_model(path=MODEL_PATH):
    m = TrafficLSTM().to(device())
    m.load_state_dict(torch.load(path, map_location=device()))
    return m


def history():
    return json.loads(HISTORY_JSON.read_text(encoding="utf-8")) if HISTORY_JSON.exists() else []


def _log(entry):
    h = history() + [entry]
    HISTORY_JSON.write_text(json.dumps(h, indent=2, ensure_ascii=False), encoding="utf-8")


def new_days(days=7, growth_to=1.3, seed=123):
    g = np.linspace(1.0, growth_to, days * STEPS_PER_DAY)
    return generate(days=days, seed=seed, start_day=DAYS, growth=g)


def mae(model, scaler, df):
    X, y = make_windows(scaler.transform(df[FEATURES].values))
    yt, yp = inverse_loads(scaler, y), inverse_loads(scaler, predict(model, X))
    return {s: float(np.mean(np.abs(yp[:, i] - yt[:, i]))) for i, s in enumerate(SLICES)}


def finetune(df_new, eval_days=2, epochs=10, lr=1e-4, log=print):
    scaler = joblib.load(SCALER_PATH)
    model = load_model()
    cut = len(df_new) - eval_days * STEPS_PER_DAY
    tune_df, eval_df = df_new.iloc[:cut], df_new.iloc[cut:]

    before = mae(model, scaler, eval_df)
    X, y = make_windows(scaler.transform(tune_df[FEATURES].values))
    v = int(len(X) * (1 - VAL_SPLIT))
    fit(model, X[:v], y[:v], X[v:], y[v:], lr=lr, max_epochs=epochs, patience=3, log=log)
    after = mae(model, scaler, eval_df)

    version = len(history()) + 1
    path = HISTORY_DIR / f"lstm_v{version}.pt"
    torch.save(model.state_dict(), path)
    shutil.copy(path, MODEL_PATH)
    _log({"version": version, "action": "finetune", "file": path.name,
          "time": datetime.now().isoformat(timespec="seconds"),
          "mae_before": before, "mae_after": after})
    return before, after


def rollback():
    shutil.copy(ORIGINAL_MODEL_PATH, MODEL_PATH)
    _log({"version": 0, "action": "rollback", "file": ORIGINAL_MODEL_PATH.name,
          "time": datetime.now().isoformat(timespec="seconds")})


if __name__ == "__main__":
    b, a = finetune(new_days())
    print("MAE преди:", {k: round(v, 2) for k, v in b.items()})
    print("MAE след: ", {k: round(v, 2) for k, v in a.items()})
    rollback()
    print("Върнат е оригиналният модел (демо на rollback).")
