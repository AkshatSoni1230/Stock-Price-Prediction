import streamlit as st
import requests
import pandas as pd
import altair as alt
import yfinance as yf

# Set up the Streamlit app title and description
st.title("Stock Price Prediction with LSTM")
st.write("Predict the future stock prices using LSTM for a specified number of days.")

# Input for stock ticker and days to predict
ticker = st.text_input("Enter Stock Ticker Symbol", value="AAPL")
n_days = st.slider("Select the number of days to predict", 1, 30, 5)  # Multi-day prediction

if st.button("Predict"):
    if ticker:
        # Make a request to the Flask API
        url = "http://localhost:5000/predict"
        try:
            response = requests.post(url, json={"ticker": ticker, "n_days": n_days}, timeout=120)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
            
            data = response.json()
            predictions = data['predictions']
            st.write(f"Predicted Closing Prices for {ticker} over the next {n_days} days:")
            st.write(predictions)

            # Prepare dates and predicted prices for plotting
            dates = list(predictions.keys())
            prices = list(predictions.values())

            # Fetch historical data for plotting
            stock_data = yf.download(ticker, period="1y")
            stock_data.reset_index(inplace=True)  # Reset index to have "Date" as a column
            stock_data['Moving Average'] = stock_data['Close'].rolling(window=20).mean()

            # Historical Prices Chart
            historical_chart = alt.Chart(stock_data).mark_line(color="blue").encode(
                x="Date:T",
                y="Close:Q",
                tooltip=["Date:T", "Close:Q"]
            ).properties(
                title=f"{ticker} Historical Prices",
                width=700
            ).interactive()  # Enable zooming and panning

            # Moving Average Chart
            moving_avg_chart = alt.Chart(stock_data).mark_line(color="orange", strokeDash=[5, 3]).encode(
                x="Date:T",
                y="Moving Average:Q",
                tooltip=["Date:T", "Moving Average:Q"]
            ).interactive()  # Enable zooming and panning

            # Predictions Chart
            future_dates = pd.date_range(stock_data["Date"].iloc[-1] + pd.Timedelta(days=1), periods=n_days)
            predictions_df = pd.DataFrame({"Date": future_dates, "Predicted Price": prices})
            predictions_chart = alt.Chart(predictions_df).mark_line(color="green").encode(
                x="Date:T",
                y="Predicted Price:Q",
                tooltip=["Date:T", "Predicted Price:Q"]
            ).properties(
                title=f"{ticker} Predicted Prices",
                width=700
            ).interactive()  # Enable zooming and panning

            # Display the charts in Streamlit
            st.altair_chart(historical_chart + moving_avg_chart + predictions_chart, use_container_width=True)

        except requests.exceptions.Timeout:
            st.error("The request timed out. Please try again later or reduce the number of prediction days.")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the prediction server. Ensure the Flask server is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")
    else:
        st.write("Please enter a valid ticker symbol.")

