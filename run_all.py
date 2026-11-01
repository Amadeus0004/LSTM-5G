import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
for step in ["data_gen.py", "train.py", "simulate.py", "finetune.py"]:
    print(f"\n===== {step} =====", flush=True)
    subprocess.run([sys.executable, str(SRC / step)], cwd=SRC, check=True)
