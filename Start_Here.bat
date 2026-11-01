@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Създавам виртуална среда...
    python -m venv venv
    venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    venv\Scripts\python.exe -m pip install -r requirements.txt
)
if not exist models\lstm.pt venv\Scripts\python.exe run_all.py
venv\Scripts\python.exe -m streamlit run src\app.py
