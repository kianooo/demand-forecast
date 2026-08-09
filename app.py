import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import sqlite3
import os

st.set_page_config(page_title="Demand Forecast", layout="wide")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "forecast.db")

# ---------- models ----------
@st.cache_resource
def load_models():
    return {q: lgb.Booster(model_file=os.path.join(APP_DIR, f"model_q{q}_v3.txt"))
            for q in (10, 50, 90)}

models = load_models()

# ---------- lightweight lookups (cached once) ----------
@st.cache_data
def get_catalog():
    con = sqlite3.connect(DB_PATH)
    cat = pd.read_sql(
        "SELECT DISTINCT item_id, store_id FROM sales ORDER BY item_id, store_id", con)
    con.close()
    return cat

catalog = get_catalog()

# ---------- on-demand per-series query: THE architectural change ----------
@st.cache_data
def get_series(item_id, store_id):
    """Pull one product-store series with features, straight from SQL."""
    con = sqlite3.connect(DB_PATH)
    q = """
    WITH daily AS (
        SELECT s.item_id, s.store_id, s.date, s.units,
               c.event_name_1, c.snap_CA, c.wm_yr_wk
        FROM sales s JOIN calendar c ON s.date = c.date
        WHERE s.item_id = ? AND s.store_id = ?
    )
    SELECT d.*, p.sell_price
    FROM daily d
    LEFT JOIN prices p ON p.item_id = d.item_id
        AND p.store_id = d.store_id AND p.wm_yr_wk = d.wm_yr_wk
    ORDER BY d.date
    """
    df = pd.read_sql(q, con, params=[item_id, store_id], parse_dates=["date"])
    con.close()
    df["sell_price"] = df["sell_price"].ffill()
    return df

# ---------- sidebar ----------
st.sidebar.title("Planner Controls")
store = st.sidebar.selectbox("Store", sorted(catalog["store_id"].unique()))
items_in_store = catalog[catalog["store_id"] == store]["item_id"].tolist()
item = st.sidebar.selectbox("Product", items_in_store)
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 28, 28)
promo = st.sidebar.checkbox("Promo next week (10% off)")

FEATURES3 = ["dayofweek", "is_weekend", "month", "is_event", "snap_CA",
             "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28",
             "sell_price", "price_change", "price_rel",
             "store_id", "category", "item_id"]

def forecast(series_df, item_id, store_id, horizon, promo_discount=0.0):
    hist = series_df["units"].tolist()
    prices_hist = series_df["sell_price"].tolist()
    last_date = series_df["date"].max()
    last_price = prices_hist[-1]
    mean_price = float(np.mean([p for p in prices_hist if pd.notna(p)]))
    category = item_id.split("_")[0]

    preds = []
    for i in range(1, horizon + 1):
        day = last_date + pd.Timedelta(i, unit="D")
        price = last_price * (1 - promo_discount) if 7 <= i <= 14 else last_price
        row = pd.DataFrame([{
            "dayofweek": (day.dayofweek + 1) % 7,   # convert pandas Mon=0 -> sqlite Sun=0
            "is_weekend": int(day.dayofweek >= 5),
            "month": day.month,
            "is_event": 0,
            "snap_CA": 0,
            "lag_7": hist[-7], "lag_14": hist[-14], "lag_28": hist[-28],
            "roll_mean_7": float(np.mean(hist[-7:])),
            "roll_mean_28": float(np.mean(hist[-28:])),
            "sell_price": price,
            "price_change": (price - last_price) / last_price if last_price else 0.0,
            "price_rel": price / mean_price if mean_price else 1.0,
            "store_id": store_id, "category": category, "item_id": item_id,
        }])
        for c in ["store_id", "category", "item_id"]:
            row[c] = row[c].astype("category")
        row = row[FEATURES3]
        q10 = max(0.0, models[10].predict(row)[0])
        q50 = max(0.0, models[50].predict(row)[0])
        q90 = max(0.0, models[90].predict(row)[0])
        preds.append({"date": day, "q10": q10, "q50": q50, "q90": q90})
        hist.append(q50)
    return pd.DataFrame(preds)

series = get_series(item, store)
fc = forecast(series, item, store, horizon, 0.10 if promo else 0.0)

# ---------- main ----------
st.title("Demand Forecasting Tool")
st.caption(f"LightGBM quantile forecasts · 100 products × 3 CA stores · "
           f"SQLite on-demand queries · viewing {item} @ {store}")

recent = series.tail(60)
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(recent["date"], recent["units"], label="history")
ax.plot(fc["date"], fc["q50"], label="forecast (median)", linewidth=2)
ax.fill_between(fc["date"], fc["q10"], fc["q90"], alpha=0.2, label="10th-90th pct")
ax.legend(); ax.set_title(f"{item} — {store}")
st.pyplot(fig)
st.caption("Calibration: 85.6% one-step coverage vs 80% nominal across 300 series; "
           "recursive multi-day coverage runs lower as uncertainty compounds.")

total_demand = fc["q50"].sum()
order_at_service = fc["q90"].sum()
safety_stock = order_at_service - total_demand
c1, c2, c3 = st.columns(3)
c1.metric("Expected demand (median)", f"{total_demand:,.0f} units")
c2.metric("Safety buffer", f"{safety_stock:,.0f} units")
c3.metric("Order for ~90% service", f"{order_at_service:,.0f} units")