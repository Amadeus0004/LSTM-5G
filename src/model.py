import numpy as np
import torch
import torch.nn as nn

from config import HIDDEN, LAYERS, DROPOUT, WINDOW, FEATURES, SLICES


class TrafficLSTM(nn.Module):
    def __init__(self, n_in=len(FEATURES), n_out=len(SLICES)):
        super().__init__()
        self.lstm = nn.LSTM(n_in, HIDDEN, num_layers=LAYERS,
                            dropout=DROPOUT, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def make_windows(arr, window=WINDOW):
    X = np.stack([arr[i:i + window] for i in range(len(arr) - window)])
    y = arr[window:, :len(SLICES)]
    return X.astype(np.float32), y.astype(np.float32)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, X, batch=1024):
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            outs.append(model(torch.from_numpy(X[i:i + batch]).to(dev)).cpu().numpy())
    return np.concatenate(outs)


def inverse_loads(scaler, y_scaled):
    k = len(SLICES)
    return y_scaled * (scaler.data_max_[:k] - scaler.data_min_[:k]) + scaler.data_min_[:k]
