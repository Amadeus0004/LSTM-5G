from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
HISTORY_DIR = MODEL_DIR / "history"
RESULTS_DIR = ROOT / "results"
for d in (DATA_DIR, MODEL_DIR, HISTORY_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TRAFFIC_CSV = DATA_DIR / "traffic.csv"
MODEL_PATH = MODEL_DIR / "lstm.pt"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ORIGINAL_MODEL_PATH = HISTORY_DIR / "lstm_v0_original.pt"

DAYS = 30
STEP_MIN = 5
STEPS_PER_DAY = 24 * 60 // STEP_MIN
N_STEPS = DAYS * STEPS_PER_DAY
SLICES = ["eMBB", "URLLC", "mMTC"]
FEATURES = SLICES + ["users"]

WINDOW = 24
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.2

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
BATCH = 64
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 5
SEED = 42

POOL_PRB = 100
PRB_PER_REQ = {"eMBB": 2.0, "URLLC": 0.5, "mMTC": 0.25}
BASE_LATENCY_MS = {"eMBB": 10.0, "URLLC": 1.0, "mMTC": 50.0}
ROBUST_SEEDS = list(range(10))
ROBUST_POOLS = [50, 100, 200]
