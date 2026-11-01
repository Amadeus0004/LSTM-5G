import json
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import RESULTS_DIR, TRAFFIC_CSV, SLICES, SCALER_PATH, ROBUST_POOLS
import simulate as sim
import finetune as ft

st.set_page_config(page_title="5G LSTM slicing", layout="wide")
page = st.sidebar.radio("Страница", ["1. Преглед", "2. Прогнози", "3. Сравнение на методите",
                                     "4. Симулация на живо", "5. Дотрениране"])


@st.cache_data
def traffic():
    return pd.read_csv(TRAFFIC_CSV, parse_dates=["timestamp"]).set_index("timestamp")


@st.cache_data
def preds():
    return pd.read_csv(RESULTS_DIR / "predictions_test.csv", parse_dates=["timestamp"]).set_index("timestamp")


if page.startswith("1"):
    st.title("LSTM модел за разпределение на ресурси в 5G мрежи")
    st.write("Общ пул от **100 PRB** се дели между eMBB, URLLC и mMTC. LSTM прогнозира "
             "натоварването за следващите 5 минути и ресурсите се разпределят предварително.")
    df = traffic()
    c = st.columns(4)
    c[0].metric("Стъпки (5 мин)", len(df))
    c[1].metric("Дни", len(df) // 288)
    c[2].metric("Средно общо търсене", f"{df[SLICES].sum(axis=1).mean():.1f} PRB")
    c[3].metric("Време над 100 PRB", f"{(df[SLICES].sum(axis=1) > 100).mean():.1%}")
    days = st.slider("Покажи дни", 1, 30, (1, 7))
    part = df.iloc[(days[0] - 1) * 288: days[1] * 288]
    st.subheader("Търсене по slice (PRB)")
    st.line_chart(part[SLICES])
    st.subheader("Активни потребители")
    st.line_chart(part["users"])
    st.subheader("Среден дневен профил")
    st.line_chart(df[SLICES].groupby(df.index.hour).mean())

elif page.startswith("2"):
    st.title("Прогнози върху тест сета (последните 6 дни)")
    m = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    rows = [{"slice": s, "LSTM MAE": m["lstm"][s]["MAE"], "LSTM RMSE": m["lstm"][s]["RMSE"],
             "Наивен MAE": m["naive_persistence"][s]["MAE"],
             "Наивен RMSE": m["naive_persistence"][s]["RMSE"]} for s in SLICES]
    st.write(f"Тренирано за {m['epochs']} епохи ({m['train_seconds']} s). "
             "Наивният модел = „следващата стъпка ще е като последната“ (това ползва PF).")
    st.dataframe(pd.DataFrame(rows).set_index("slice").round(2))
    p = preds()
    s = st.selectbox("Slice", SLICES)
    n = st.slider("Брой стъпки", 50, len(p), 288)
    st.line_chart(p[[f"{s}_actual", f"{s}_pred"]].iloc[:n])
    st.subheader("Крива на загубата")
    st.image(str(RESULTS_DIR / "loss.png"))

elif page.startswith("3"):
    st.title("Сравнение: Round-Robin vs Proportional Fair vs LSTM")
    pool = st.select_slider("Пул PRB", [50, 75, 100, 150, 200], 100)
    cmp = sim.compare(pool=pool).set_index("method")
    cols = ["blocking_%", "latency_ms", "utilization_%", "jain"]
    st.dataframe(cmp[cols].round(3))
    c = st.columns(2)
    c[0].subheader("Вероятност за блокиране, %"); c[0].bar_chart(cmp["blocking_%"])
    c[1].subheader("Средна латентност, ms"); c[1].bar_chart(cmp["latency_ms"])
    st.subheader("Блокиране по slice, %")
    st.bar_chart(cmp[[f"blocking_{s}_%" for s in SLICES]])
    st.subheader("Устойчивост: 10 seed-а x пулове " + str(ROBUST_POOLS))
    f = RESULTS_DIR / "robustness.csv"
    if f.exists():
        rob = pd.read_csv(f)
        agg = rob.groupby(["pool", "method"])[cols].agg(["mean", "std"]).round(3)
        st.dataframe(agg)
        st.bar_chart(rob.groupby(["pool", "method"])["blocking_%"].mean().unstack())

elif page.startswith("4"):
    st.title("Симулация на живо")
    ts, actual, forecast = sim.load_test_series()
    pool = st.slider("Пул PRB", 50, 200, 100, 10)
    start = st.slider("Начална стъпка", 0, len(actual) - 100, 200)
    steps = st.slider("Брой стъпки", 10, 288, 72)
    speed = st.slider("Пауза между стъпките (s)", 0.0, 0.5, 0.05)
    if st.button("▶ Старт"):
        a, f = actual[start:start + steps], forecast[start:start + steps]
        traces = {m: sim.run(m, a, f, pool, seed=0)[1] for m in sim.METHODS}
        head, cols_ph, chart_ph = st.empty(), st.columns(3), st.empty()
        ph = [c.empty() for c in cols_ph]
        cum = {m: [] for m in sim.METHODS}
        for t in range(steps):
            head.markdown(f"**{ts.iloc[start + t]}** — реално търсене: " +
                          ", ".join(f"{s}={a[t, i]:.1f}" for i, s in enumerate(SLICES)))
            for k, m in enumerate(sim.METHODS):
                tr = traces[m]
                req, blk = tr["requests"][:t + 1].sum(), tr["blocked"][:t + 1].sum()
                cum[m].append(100 * blk / max(req, 1))
                ph[k].markdown(f"#### {m}")
                ph[k].bar_chart(pd.DataFrame({"разпределени": tr["alloc"][t], "търсене": a[t]},
                                             index=SLICES), stack=False)
            chart_ph.line_chart(pd.DataFrame(cum).rename_axis("стъпка"), height=220)
            time.sleep(speed)
        st.success("Готово. Линията отдолу е кумулативната вероятност за блокиране (%).")

else:
    st.title("Дотрениране при ръст на трафика")
    days = st.slider("Нови дни", 3, 14, 7)
    growth = st.slider("Ръст до края на периода", 1.0, 1.6, 1.3, 0.05)
    epochs = st.slider("Макс. епохи", 1, 30, 10)
    c = st.columns(2)
    if c[0].button("Генерирай нови дни и дотренирай"):
        df_new = ft.new_days(days, growth)
        st.line_chart(df_new.set_index("timestamp")[SLICES])
        logbox = st.empty(); lines = []
        def log(msg):
            lines.append(msg); logbox.code("\n".join(lines[-12:]))
        with st.spinner("Дотренира се..."):
            before, after = ft.finetune(df_new, epochs=epochs, log=log)
        st.dataframe(pd.DataFrame({"MAE преди": before, "MAE след": after}).round(2))
        st.cache_data.clear()
    if c[1].button("↩ Върни оригиналния модел"):
        ft.rollback()
        st.success("Оригиналният модел е активен.")
    st.subheader("История на версиите")
    h = ft.history()
    st.dataframe(pd.json_normalize(h) if h else pd.DataFrame())
