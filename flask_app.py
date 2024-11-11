from flask import Flask, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

app = Flask(__name__)

def build_and_train_lstm_model(data, n_days):
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare training data
    X_train, y_train = [], []
    look_back = 60  # use last 60 days to predict the next day

    for i in range(look_back, len(scaled_data) - n_days):
        X_train.append(scaled_data[i - look_back:i, 0])
        y_train.append(scaled_data[i:i + n_days, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=n_days))  # Output layer for n future days

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, verbose=0)  # Reduced epochs for speed

    return model, scaler

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get('ticker')
        n_days = data.get('n_days', 1)  # Number of future days to predict

        if not ticker:
            return jsonify({'error': 'Ticker symbol is required'}), 400

        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1825)  # Use 5 year of data
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            return jsonify({'error': 'No data found for the specified ticker and time range'}), 404

        # Prepare data
        closing_data = stock_data[['Close']].dropna().values

        # Train LSTM model
        model, scaler = build_and_train_lstm_model(closing_data, n_days)

        # Predict future prices
        last_60_days = closing_data[-60:]  # Last 60 days to predict future
        last_60_scaled = scaler.transform(last_60_days)
        X_test = np.array([last_60_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        predicted_prices = model.predict(X_test)[0]
        predicted_prices = scaler.inverse_transform([predicted_prices])[0]  # Scale back to original

        # Prepare response
        predictions = predicted_prices.tolist()
        dates = [(end_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, n_days + 1)]
        result = {'ticker': ticker, 'predictions': dict(zip(dates, predictions))}

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
