import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_datareader.data as web
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import altair as alt
import warnings
import random

# -------------------- CONFIG --------------------
LOOK_BACK = 60
DEFAULT_EPOCHS = 10
CACHE_TTL_SECONDS = 60 * 60  # cache data for 1 hour

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

warnings.filterwarnings("ignore")
st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")

# -------------------- DATA FETCH (Stooq) --------------------
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_stock_history_stooq(ticker: str) -> pd.DataFrame:
    """
    Try several look-back windows (5y, 3y, 1y, max) using Stooq.
    Returns a DataFrame sorted by date (ascending) or an empty DataFrame.
    """
    ticker = ticker.strip().upper()
    now = pd.Timestamp.now()
    # candidate lookbacks (years / months)
    candidates = [
        pd.DateOffset(years=5),
        pd.DateOffset(years=3),
        pd.DateOffset(years=2),
        pd.DateOffset(years=1),
        pd.DateOffset(months=6),
        None  # fallback -> try "max" (very long start)
    ]

    for offset in candidates:
        try:
            if offset is None:
                start = pd.Timestamp(2000, 1, 1)  # fallback "max-ish"
            else:
                start = now - offset

            df = web.DataReader(ticker, "stooq", start=start, end=now)
            # Stooq returns descending order, so sort ascending to be consistent
            if df is None or df.empty:
                continue
            if "Close" not in df.columns:
                continue
            df = df.sort_index()
            # require at least some rows
            if len(df) >= 10:
                return df
        except Exception:
            # try next candidate
            continue

    return pd.DataFrame()  # nothing worked

# -------------------- MODEL TRAINING --------------------
@st.cache_resource
def train_lstm_model_cached(ticker: str, n_days: int, look_back: int = LOOK_BACK, epochs: int = DEFAULT_EPOCHS):
    """
    Train an LSTM model and cache it for (ticker, n_days, look_back, epochs).
    Returns (model, scaler, last_date).
    Raises ValueError for not-enough-data or missing data.
    """
    hist = fetch_stock_history_stooq(ticker)
    if hist.empty:
        raise ValueError("No historical data found for the ticker (Stooq).")

    closing = hist[["Close"]].dropna().values  # shape (N,1)
    N = len(closing)
    if N < look_back + n_days:
        raise ValueError(
            f"Not enough historical data for requested configuration.\n"
            f"Need at least look_back + n_days = {look_back + n_days} rows, found {N}."
        )

    # scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(closing)  # shape (N,1)

    # prepare X,y for multi-day output
    X, y = [], []
    # i is the index where y starts; last valid i is N - n_days
    for i in range(look_back, N - n_days + 1):
        X.append(scaled[i - look_back:i, 0])     # look_back values
        y.append(scaled[i:i + n_days, 0])        # next n_days values

    X = np.array(X)   # (samples, look_back)
    y = np.array(y)   # (samples, n_days)

    # reshape X for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # build model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(units=n_days))  # multi-day output

    model.compile(optimizer="adam", loss="mean_squared_error")

    # training with early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, callbacks=[early_stop])

    last_date = hist.index[-1]  # pandas Timestamp of last available trading day
    return model, scaler, last_date

# -------------------- PREDICTION WRAPPER --------------------
def predict_future_prices(ticker: str, n_days: int, look_back: int = LOOK_BACK, epochs: int = DEFAULT_EPOCHS):
    """
    Returns (predictions_df, hist_df, error_message).
    predictions_df: DataFrame with Date (business days) and Predicted Price
    hist_df: historical dataframe (reset index) for plotting
    """
    ticker = ticker.strip().upper()
    hist = fetch_stock_history_stooq(ticker)
    if hist.empty:
        return None, None, "No historical data found for the ticker (Stooq)."

    try:
        model, scaler, last_date = train_lstm_model_cached(ticker, n_days, look_back, epochs)
    except Exception as e:
        return None, None, str(e)

    closing = hist[["Close"]].dropna().values
    last_window = closing[-look_back:]  # shape (look_back,1)
    last_scaled = scaler.transform(last_window)  # shape (look_back,1)
    X_test = last_scaled.reshape(1, look_back, 1)

    pred_scaled = model.predict(X_test, verbose=0)[0]  # shape (n_days,)
    pred_prices = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    # business day future dates
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=n_days)
    predictions_df = pd.DataFrame({"Date": future_dates, "Predicted Price": pred_prices})

    hist_for_plot = hist.reset_index().copy()
    hist_for_plot["MA20"] = hist_for_plot["Close"].rolling(window=20).mean()

    return predictions_df, hist_for_plot, None

# -------------------- STREAMLIT UI --------------------
st.title("📈 LSTM Multi-day Stock Predictor")
st.write("Predict the next *business* days' closing prices.")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    ticker = st.text_input("Ticker symbol", value="AAPL").strip().upper()

with col2:
    n_days = st.slider("Days to predict (business days)", min_value=1, max_value=30, value=5)

with col3:
    epochs = st.slider("Training epochs", 1, 50, DEFAULT_EPOCHS)

fast_mode = st.checkbox("Fast mode (caps epochs)", value=False)
if fast_mode:
    epochs = min(epochs, 5)

if st.button("Predict"):
    if not ticker:
        st.warning("Enter a valid ticker symbol.")
    else:
        with st.spinner("Fetching data and training (if needed)..."):
            preds_df, hist_df, err = predict_future_prices(ticker, n_days, LOOK_BACK, epochs)

        if err:
            st.error(f"Error: {err}")
            st.stop()

        # show predictions
        st.subheader(f"{ticker} — Predicted closing prices for next {n_days} business days")
        preds_show = preds_df.copy()
        preds_show["Date"] = preds_show["Date"].dt.strftime("%Y-%m-%d")
        st.table(preds_show.set_index("Date"))

        st.download_button("Download predictions (CSV)", preds_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"{ticker}_predictions.csv")

        # Plot
        hist_plot = hist_df.copy()
        pred_plot = preds_df.copy()

        # Historical close
        close_line = alt.Chart(hist_plot).mark_line().encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Close:Q", title="Close"),
            tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Close:Q", title="Close")]
        )

        # MA20
        ma_line = alt.Chart(hist_plot).mark_line(strokeDash=[5, 3]).encode(
            x="Date:T",
            y=alt.Y("MA20:Q", title="MA(20)"),
            tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("MA20:Q", title="MA(20)")]
        )

        # Predictions (future)
        preds_chart = alt.Chart(pred_plot).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Predicted Price:Q", title="Predicted Price"),
            tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Predicted Price:Q", title="Predicted")]
        )

        combined = alt.layer(close_line, ma_line, preds_chart).properties(
            title=f"{ticker}: Historical Close (MA20) + {n_days}-day Predictions",
            width=900,
            height=450
        ).interactive()

        st.altair_chart(combined, use_container_width=True)

        st.success("Done — model is cached per (ticker, horizon, epochs).")
