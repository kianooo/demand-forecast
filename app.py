import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

st.set_page_config(page_title="Demand Forecast", layout="wide")

# ---------- load model + data (cached so it doesn't reload on every click) ----------

@st.cache_resource
def load_models():
    return {q: lgb.Booster(model_file=f"model_q{q}.txt") for q in (10, 50, 90)}

@st.cache_data
def load_data():
    df = pd.read_csv("app_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

models = load_models()
data = load_data()

FEATURES = ["dayofweek", "is_weekend", "month", "is_event", "snap_CA",
            "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28",
            "sell_price", "price_change", "price_rel"]

# ---------- sidebar: planner controls ----------
st.sidebar.title("Planner Controls")
item = st.sidebar.selectbox("Product", sorted(data["item_id"].unique()))
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 28, 28)
promo = st.sidebar.checkbox("Promo next week (10% off)")


# ---------- recursive forecast (same logic as the notebook) ----------
def forecast(item_id, horizon, promo_discount=0.0):
    df = data[data["item_id"] == item_id].sort_values("date").copy()
    hist = df["units"].tolist()
    last_date = df["date"].max()
    last_price = df["sell_price"].iloc[-1]
    mean_price = df["sell_price"].mean()

    preds = []
    for i in range(1, horizon + 1):
        day = last_date + pd.Timedelta(i, unit="D")
        price = last_price * (1 - promo_discount) if 7 <= i <= 14 else last_price
        row = pd.DataFrame([{
            "dayofweek": day.dayofweek,
            "is_weekend": int(day.dayofweek >= 5),
            "month": day.month,
            "is_event": 0,
            "snap_CA": 0,
            "lag_7": hist[-7], "lag_14": hist[-14], "lag_28": hist[-28],
            "roll_mean_7": np.mean(hist[-7:]),
            "roll_mean_28": np.mean(hist[-28:]),
            "sell_price": price,
            "price_change": (price - last_price) / last_price,
            "price_rel": price / mean_price,
        }])[FEATURES]
        q10 = max(0, models[10].predict(row)[0])
        q50 = max(0, models[50].predict(row)[0])
        q90 = max(0, models[90].predict(row)[0])
        preds.append({"date": day, "q10": q10, "q50": q50, "q90": q90})
        hist.append(q50)                       # roll forward on the median
    return df, pd.DataFrame(preds)

hist_df, fc = forecast(item, horizon, 0.10 if promo else 0.0)

# ---------- main page ----------
st.title("Demand Forecasting Tool")
st.caption("LightGBM recursive forecast · trained on M5 Walmart data · CA_1 store, top-10 SKUs")

# forecast chart: last 60 days of history + forecast
recent = hist_df.tail(60)
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(recent["date"], recent["units"], label="history")
ax.plot(fc["date"], fc["q50"], label="forecast (median)", linewidth=2)
ax.fill_between(fc["date"], fc["q10"], fc["q90"], alpha=0.2,
                label="10th-90th percentile")
ax.legend(); ax.set_title(item)
st.pyplot(fig)
st.caption("Interval note: in backtesting, actual sales fell inside this band "
           "74% of days (nominal 80%) — multi-step uncertainty compounds and "
           "intervals are slightly overconfident.")

total_demand = fc["q50"].sum()
order_at_service = fc["q90"].sum()          # 90th percentile = ~90% service level
safety_stock = order_at_service - total_demand
col1, col2, col3 = st.columns(3)
col1.metric("Expected demand (median)", f"{total_demand:,.0f} units")
col2.metric("Safety buffer", f"{safety_stock:,.0f} units")
col3.metric("Order for ~90% service", f"{order_at_service:,.0f} units")