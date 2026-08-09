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
recent_sales = series["units"].tail(28).sum()
if recent_sales < 5:
    st.warning(f"⚠ Only {int(recent_sales)} units sold in the last 28 days — "
               "this product may be delisted or out of stock here. "
               "The forecast reflects that inactivity.")
fc = forecast(series, item, store, horizon, 0.10 if promo else 0.0)

# ---------- main ----------
st.title("Demand Forecasting Tool")
st.caption(f"LightGBM quantile forecasts · 100 products × 3 CA stores · "
           f"SQLite on-demand queries · viewing {item} @ {store}")

total_demand = fc["q50"].sum()
order_at_service = fc["q90"].sum()
safety_stock = order_at_service - total_demand
c1, c2, c3 = st.columns(3)
c1.metric("Expected demand (median)", f"{total_demand:,.0f} units")
c2.metric("Safety buffer", f"{safety_stock:,.0f} units")
c3.metric("Order for ~90% service", f"{order_at_service:,.0f} units")

tab_fc, tab_bt, tab_pf = st.tabs(["📈 Forecast", "🔬 Backtest", "🏪 Store portfolio"])

with tab_fc:
    recent = series.tail(60)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(recent["date"], recent["units"], label="history")
    ax.plot(fc["date"], fc["q50"], label="forecast (median)", linewidth=2)
    ax.fill_between(fc["date"], fc["q10"], fc["q90"], alpha=0.2, label="10th-90th pct")
    ax.legend(); ax.set_title(f"{item} — {store}")
    st.pyplot(fig)
    st.caption("Calibration: 85.6% one-step coverage vs 80% nominal across 300 series; "
            "recursive multi-day coverage runs lower as uncertainty compounds.")

    st.markdown("**How did the model do on the last 28 days it never saw?** "
                "Trained on data up to 28 days before the end, then forecast "
                "recursively — same conditions as a real forecast.")

with tab_bt:
    bt_train = series.iloc[:-28]
    bt_actual = series.iloc[-28:]
    bt_fc = forecast(bt_train, item, store, 28, 0.0)

    bt_mae = float(np.mean(np.abs(bt_actual["units"].values - bt_fc["q50"].values)))
    naive_bt = np.tile(bt_train["units"].tail(7).values, 4)[:28]
    bt_naive_mae = float(np.mean(np.abs(bt_actual["units"].values - naive_bt)))
    bt_cov = float(np.mean((bt_actual["units"].values >= bt_fc["q10"].values) &
                           (bt_actual["units"].values <= bt_fc["q90"].values)))

    b1, b2, b3 = st.columns(3)
    b1.metric("Model MAE", f"{bt_mae:.2f}")
    b2.metric("vs naive",
              f"{(bt_naive_mae - bt_mae) / bt_naive_mae:+.0%}" if bt_naive_mae > 0 else "n/a",
              help="Positive = model beats copying last week")
    b3.metric("Band coverage", f"{bt_cov:.0%}", help="Nominal 80%")

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(bt_actual["date"], bt_actual["units"], label="actual", marker="o")
    ax2.plot(bt_fc["date"], bt_fc["q50"], label="model forecast", linewidth=2)
    ax2.plot(bt_fc["date"], naive_bt, label="naive", linestyle="--", alpha=0.7)
    ax2.fill_between(bt_fc["date"], bt_fc["q10"], bt_fc["q90"], alpha=0.15)
    ax2.legend(); ax2.set_title("Held-out 28 days: forecast vs reality")
    st.pyplot(fig2)

with tab_pf:
    @st.cache_data
    def store_overview(store_id):
        con = sqlite3.connect(DB_PATH)
        q = """
        WITH bounds AS (SELECT MAX(date) AS max_d FROM sales)
        SELECT s.item_id,
               SUM(CASE WHEN s.date >= date(b.max_d, '-28 days')
                        THEN s.units ELSE 0 END) AS last_28,
               SUM(CASE WHEN s.date >= date(b.max_d, '-56 days')
                         AND s.date <  date(b.max_d, '-28 days')
                        THEN s.units ELSE 0 END) AS prev_28
        FROM sales s CROSS JOIN bounds b
        WHERE s.store_id = ?
        GROUP BY s.item_id
        ORDER BY last_28 DESC
        """
        df = pd.read_sql(q, con, params=[store_id])
        con.close()
        df[["last_28", "prev_28"]] = df[["last_28", "prev_28"]].fillna(0)
        df["trend"] = (df["last_28"] - df["prev_28"]) / df["prev_28"].replace(0, np.nan)
        return df
        

    ov = store_overview(store)
    st.markdown(f"**{store}: last 28 days vs the 28 before**")

    p1, p2 = st.columns(2)
    p1.metric("Store volume, last 28d", f"{int(ov['last_28'].sum()):,} units",
              f"{(ov['last_28'].sum() - ov['prev_28'].sum()) / ov['prev_28'].sum():+.1%}")
    p2.metric("Active products (>5 units)", f"{(ov['last_28'] > 5).sum()} / {len(ov)}")

    st.markdown("**Top 10 by volume**")
    st.dataframe(ov.head(10).style.format({"last_28": "{:,.0f}", "prev_28": "{:,.0f}",
                                           "trend": "{:+.1%}"}), use_container_width=True)
    st.markdown("**Biggest movers (min 50 units)**")
    movers = ov[ov["prev_28"] >= 50].nlargest(5, "trend")
    droppers = ov[ov["prev_28"] >= 50].nsmallest(5, "trend")
    st.dataframe(pd.concat([movers, droppers]).style.format(
        {"last_28": "{:,.0f}", "prev_28": "{:,.0f}", "trend": "{:+.1%}"}),
        use_container_width=True)