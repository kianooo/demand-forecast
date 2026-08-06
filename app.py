import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

st.set_page_config(page_title="Demand Forecast", layout="wide")

# ---------- load model + data (cached so it doesn't reload on every click) ----------
@st.cache_resource
def load_model():
    return lgb.Booster(model_file="model.txt")

@st.cache_data
def load_data():
    df = pd.read_csv("app_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

model = load_model()
data = load_data()

FEATURES = ["dayofweek", "is_weekend", "month", "is_event", "snap_CA",
            "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28",
            "sell_price", "price_change", "price_rel"]

# ---------- sidebar: planner controls ----------
st.sidebar.title("Planner Controls")
item = st.sidebar.selectbox("Product", sorted(data["item_id"].unique()))
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 28, 28)
promo = st.sidebar.checkbox("Promo next week (10% off)")
service_z = st.sidebar.select_slider(
    "Service level", options=[1.28, 1.64, 2.05],
    format_func=lambda z: {1.28: "90%", 1.64: "95%", 2.05: "98%"}[z]
)

# ---------- recursive forecast (same logic as the notebook) ----------
def forecast(item_id, horizon, promo_discount=0.0):
    df = data[data["item_id"] == item_id].sort_values("date").copy()
    hist = df["units"].tolist()
    last_date = df["date"].max()
    last_price = df["sell_price"].iloc[-1]
    mean_price = df["sell_price"].mean()

    preds = []
    for i in range(1, horizon + 1):
        day = last_date + pd.Timedelta(days=i)
        price = last_price * (1 - promo_discount) if 7 <= i <= 14 else last_price
        row = {
            "dayofweek": day.dayofweek,
            "is_weekend": int(day.dayofweek >= 5),
            "month": day.month,
            "is_event": 0,
            "snap_CA": 0,
            "lag_7":  hist[-7],
            "lag_14": hist[-14],
            "lag_28": hist[-28],
            "roll_mean_7":  np.mean(hist[-7:]),
            "roll_mean_28": np.mean(hist[-28:]),
            "sell_price": price,
            "price_change": (price - last_price) / last_price,
            "price_rel": price / mean_price,
        }
        p = max(0, model.predict(pd.DataFrame([row])[FEATURES])[0])
        preds.append({"date": day, "pred": p})
        hist.append(p)                      # feed prediction back in
    return df, pd.DataFrame(preds)

hist_df, fc = forecast(item, horizon, 0.10 if promo else 0.0)

# ---------- main page ----------
st.title("Demand Forecasting Tool")
st.caption("LightGBM recursive forecast · trained on M5 Walmart data · CA_1 store, top-10 SKUs")

# forecast chart: last 60 days of history + forecast
recent = hist_df.tail(60)
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(recent["date"], recent["units"], label="history")
ax.plot(fc["date"], fc["pred"], label="forecast", linewidth=2)
# uncertainty band from residual std (honest, simple)
resid_std = hist_df["units"].tail(90).std()
ax.fill_between(fc["date"], (fc["pred"] - service_z * resid_std).clip(lower=0),
                fc["pred"] + service_z * resid_std, alpha=0.2, label="uncertainty")
ax.legend(); ax.set_title(item)
st.pyplot(fig)

# ---------- the planner numbers ----------
total_demand = fc["pred"].sum()
safety_stock = service_z * resid_std * np.sqrt(horizon / 7)
col1, col2, col3 = st.columns(3)
col1.metric("Forecast demand", f"{total_demand:,.0f} units")
col2.metric("Safety stock", f"{safety_stock:,.0f} units")
col3.metric("Recommended order", f"{total_demand + safety_stock:,.0f} units")