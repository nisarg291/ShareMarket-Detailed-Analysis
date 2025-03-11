import pandas as pd
import numpy as np
import yfinance as yf
import requests
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import datetime
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
import xgboost as xgb
import os
import numpy as np
import pandas as pd
import datetime
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from textblob import TextBlob
import xgboost as xgb
import joblib  # For saving models
import matplotlib as plt
API_KEY = '63e9a0bd793f42f2b872411822f131ac'
# Initialize the SentimentIntensityAnalyzer

# Set the directory for saving CSV files
DATA_DIR = 'data'  # Change this to your desired directory
PREDICTION_DIR = 'predictions'  # Directory for saving predictions

# Ensure directories exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(PREDICTION_DIR):
    os.makedirs(PREDICTION_DIR)

sia = SentimentIntensityAnalyzer()

# Function to fetch the sentiment score from news headlines using TextBlob
def get_sentiment_score(news_headlines):
    sentiment_score = 0
    for headline in news_headlines:
        blob = TextBlob(headline)
        sentiment_score += blob.sentiment.polarity  # Sentiment polarity score
    # Return average sentiment score
    print(sentiment_score / len(news_headlines))
    return sentiment_score / len(news_headlines) if news_headlines else 0

def fetch_news_data(query, language='en', page_size=20):
    url = f'https://newsapi.org/v2/everything?q={query}&language={language}&pageSize={page_size}&apiKey={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('articles', [])
        headlines = [article['title'] for article in articles]
        print(headlines)
        return headlines
    return []

# Sentiment analysis using TextBlob
def get_news_sentiment(news_data):
    sentiment_scores = [TextBlob(headline).sentiment.polarity for headline in news_data]
    print(sentiment_scores)
    return np.mean(sentiment_scores) if sentiment_scores else 0


# Function to prepare market data for model training
def prepare_market_data(sp500, nifty500, stock_data):
    # Prepare features for training
    market_data = pd.DataFrame({
        'SP500_Open': sp500['Open'].tail(60).values.flatten(),  # Last 60 days data for S&P 500
        'Nifty500_Open': nifty500['Open'].tail(60).values.flatten(),  # Last 60 days data for Nifty500
        'SP500_Close': sp500['Close'].tail(60).values.flatten(),  # Last 60 days data for S&P 500
        'Nifty500_Close': nifty500['Close'].tail(60).values.flatten(),  # Last 60 days data for Nifty500
        'Stock_Open': stock_data['Open'].tail(60).values.flatten(),   # Last day's opening price of the stock
        'Stock_Close': stock_data['Close'].tail(60).values.flatten()  # Last day's closing price of the stock
        
    })
    return market_data

# Function to adjust predictions based on sentiment score
def adjust_prediction_based_on_sentiment(prediction, sentiment_score):
    if sentiment_score > 0:
        adjusted_prediction = prediction * (1 + 0.02)  # Positive sentiment adds 2% increase
    elif sentiment_score < 0:
        adjusted_prediction = prediction * (1 - 0.02)  # Negative sentiment subtracts 2% decrease
    else:
        adjusted_prediction = prediction  # Neutral sentiment, no change
    return adjusted_prediction

def prepare_data(df, time_steps=365):
    """Prepare data for LSTM by using 'Open' and 'Close' columns."""
    df['Date'] = pd.to_datetime(df.index)  # Ensure Date is in datetime format

    # Normalize the 'Open' and 'Close' columns using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Open', 'Close']])

    X, y = [], []
    for i in range(time_steps, len(df)):
        X.append(df_scaled[i-time_steps:i, 0:2])  # Use 'Open' and 'Close'
        y.append(df_scaled[i, 0:2])

    X, y = np.array(X), np.array(y)

    return X, y, scaler

# Define the LSTM model
def create_lstm_model(input_shape):
    """Create and compile LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=2))  # 2 output nodes for 'Open' and 'Close'

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data_for_xgboost(X):
    # Reshape 3D array (samples, timesteps, features) to 2D (samples, timesteps*features)
    return X.reshape(X.shape[0], -1)

# Train model for each stock symbol
# def train_and_predict(stock_symbols):
    
# # Define the date range (from 10 years ago to today)
#     today = datetime.datetime.now()
#     start_date = today - datetime.timedelta(days=10*365)  # 10 years back
#     end_date = today

#     # Example stock symbols list

#     for stock in stock_symbols:
#         print(f"Training model for {stock}...")

      
#         data = yf.download(f"{stock}.NS", start=start_date, end=end_date)
#         csv_filename = f"nifty_stocks/{stock}_data.csv"
#         data.to_csv(csv_filename)

#         input_csv = f"nifty_stocks/{stock}_data.csv"  # Replace with your input file path
#         df = pd.read_csv(input_csv, skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"])
#         df.to_csv(input_csv, index=False)

#         stock_data = pd.read_csv(f"nifty_stocks/{stock}_data.csv")
#         # Fetch news sentiment data
#         news_headlines = fetch_news_data(stock)
#         sentiment_score = get_sentiment_score(news_headlines)
#         print(sentiment_score)

#         if stock_data is not None:
#             # Prepare data
#             X, y, scaler = prepare_data(stock_data)
#             print(len(X), len(y))
#             # Split the data into training and testing datasets
#             train_size = int(len(X) * 0.9)
#             X_train, X_test = X[:train_size], X[train_size:]
#             y_train, y_test = y[:train_size], y[train_size:]

#             # Create and train the LSTM model
#             model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
#             model.fit(X_train, y_train, epochs=30, batch_size=32)

#             # Make predictions for the last 60 days of the data (test data)
#             predictions = model.predict(X_test)

#             # Inverse transform predictions
#             predictions_rescaled = scaler.inverse_transform(predictions)

#             # Calculate Mean Squared Error
#             mse = mean_squared_error(y_test, predictions_rescaled)
#             print(f"Mean Squared Error for {stock}: {mse}")

#             # Predict next 15 days from the last date in the dataset
#             last_data = stock_data[-180:].copy()  # Get the last 60 days
#             last_scaled = scaler.transform(last_data[['Open', 'Close']])
#             future_predictions = []

#             for _ in range(15):  # Predict for the next 15 days
#                 input_data = last_scaled.reshape(1, last_scaled.shape[0], last_scaled.shape[1])
#                 pred = model.predict(input_data)
#                 future_predictions.append(pred[0])
#                 last_scaled = np.append(last_scaled[1:], pred, axis=0)

#             # Inverse transform the predictions
#             future_predictions_rescaled = scaler.inverse_transform(future_predictions)

#             # Create a DataFrame for the predictions
#             prediction_dates = pd.date_range(start=end_date + datetime.timedelta(days=1), periods=15).strftime('%Y-%m-%d').tolist()
#             prediction_df = pd.DataFrame(future_predictions_rescaled, columns=['Open', 'Close'], index=prediction_dates)

#             # Save predictions to CSV
#             prediction_df.to_csv(os.path.join(PREDICTION_DIR, f"{stock}_predictions.csv"))
#             print(f"✅ Predictions for {stock} saved in {os.path.join(PREDICTION_DIR, f'{stock}_predictions.csv')}")
      
#                 # XGBoost Model
#             # Prepare data for XGBoost (convert 3D to 2D)
#             X_train_xgb = prepare_data_for_xgboost(X_train)
#             X_test_xgb = prepare_data_for_xgboost(X_test)
            
#             xgb_model = xgb.XGBRegressor(
#                 n_estimators=100,
#                 learning_rate=0.1,
#                 max_depth=5,
#                 random_state=42
#             )
#             xgb_model.fit(X_train_xgb, y_train)
#             xgb_predictions = xgb_model.predict(X_test_xgb)
            
#             # Inverse transform predictions and calculate MSE
#             y_test_original = y_test
            
#             xgb_pred_original = xgb_predictions
            
#             xgb_mse = mean_squared_error(y_test_original, xgb_pred_original)
                
#             # Calculate Mean Squared Error
#             print(f"Mean Squared Error for {stock}: {xgb_mse}")

#             # Predict next 15 days from the last date in the dataset
#             last_data = stock_data[-365:].copy()  # Get the last 60 days
#             last_scaled = scaler.transform(last_data[['Open', 'Close']])
#             future_predictions = []

#             for _ in range(15):  # Predict for the next 15 days
#                 input_data = last_scaled.reshape(1, last_scaled.shape[0], last_scaled.shape[1])
#                 pred = xgb_model.predict(input_data)
#                 future_predictions.append(pred[0])
#                 last_scaled = np.append(last_scaled[1:], pred, axis=0)

#             # Inverse transform the predictions
#             future_predictions_rescaled = scaler.inverse_transform(future_predictions)

#             # Create a DataFrame for the predictions
#             prediction_dates = pd.date_range(start=end_date + datetime.timedelta(days=1), periods=15).strftime('%Y-%m-%d').tolist()
#             prediction_df = pd.DataFrame(future_predictions_rescaled, columns=['Open', 'Close'], index=prediction_dates)

#             # Save predictions to CSV
#             prediction_df.to_csv(os.path.join(PREDICTION_DIR, f"{stock}_xgb_predictions.csv"))
#             print(f"✅ Predictions for {stock} saved in {os.path.join(PREDICTION_DIR, f'{stock}_xgb_predictions.csv')}")
#         # Save model
#         joblib.dump(model, f'models/{stock}_stock_model.pkl')
#         print(f"Model for {stock} saved.")


os.makedirs("models", exist_ok=True)

def train_and_predict(stock_symbols):
    today = datetime.datetime.now()
    start_date = today - datetime.timedelta(days=10*365)  # 10 years back
    end_date = today

    for stock in stock_symbols:
        print(f"Training model for {stock}...")

        # Fetch stock data
        data = yf.download(f"{stock}.NS", start=start_date, end=end_date)
        if data.empty:
            print(f"No data available for {stock}")
            continue
            
        csv_filename = f"nifty_stocks/{stock}_data.csv"
        data.to_csv(csv_filename)

        # Read and clean data
        df = pd.read_csv(csv_filename, skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"])
        df.to_csv(csv_filename, index=False)
        stock_data = pd.read_csv(csv_filename)

        # Fetch news sentiment
        news_headlines = fetch_news_data(stock)
        sentiment_score = get_sentiment_score(news_headlines)
        print(f"Sentiment score for {stock}: {sentiment_score}")

        if not stock_data.empty:
            # Prepare data with dynamic time_steps
            time_steps = min(365, len(stock_data) - 1)  # Use 365 or less if data is shorter
            X, y, scaler = prepare_data(stock_data, time_steps=time_steps)
            print(f"Data prepared for {stock}: X shape {X.shape}, y shape {y.shape}")

            # Split data
            train_size = int(len(X) * 0.9)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # LSTM Model
            model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
            model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

            # LSTM Test Predictions
            lstm_predictions = model.predict(X_test)
            lstm_predictions_rescaled = scaler.inverse_transform(lstm_predictions)
            y_test_rescaled = scaler.inverse_transform(y_test)
            lstm_mse = mean_squared_error(y_test_rescaled, lstm_predictions_rescaled)
            print(f"LSTM Mean Squared Error for {stock}: {lstm_mse}")

            # LSTM Future Predictions (15 days)
            last_data = stock_data[-time_steps:].copy()  # Use exact time_steps
            last_scaled = scaler.transform(last_data[['Open', 'Close']])
            future_predictions_lstm = []

            for _ in range(15):
                input_data = last_scaled.reshape(1, time_steps, 2)
                pred = model.predict(input_data, verbose=0)
                future_predictions_lstm.append(pred[0])
                last_scaled = np.append(last_scaled[1:], pred, axis=0)

            future_predictions_lstm_rescaled = scaler.inverse_transform(future_predictions_lstm)
            prediction_dates = pd.date_range(start=end_date + datetime.timedelta(days=1), periods=15).strftime('%Y-%m-%d').tolist()
            lstm_pred_df = pd.DataFrame(future_predictions_lstm_rescaled, columns=['Open', 'Close'], index=prediction_dates)
            lstm_pred_df.to_csv(os.path.join(PREDICTION_DIR, f"{stock}_lstm_predictions.csv"))
            print(f"✅ LSTM Predictions for {stock} saved")

            # XGBoost Model
            X_train_xgb = prepare_data_for_xgboost(X_train)
            X_test_xgb = prepare_data_for_xgboost(X_test)
            
            # Train separate models for Open and Close
            xgb_open = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            xgb_close = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            
            xgb_open.fit(X_train_xgb, y_train[:, 0])  # Train on Open prices
            xgb_close.fit(X_train_xgb, y_train[:, 1])  # Train on Close prices
            
            # XGBoost Test Predictions
            xgb_pred_open = xgb_open.predict(X_test_xgb)
            xgb_pred_close = xgb_close.predict(X_test_xgb)
            xgb_predictions = np.column_stack((xgb_pred_open, xgb_pred_close))
            xgb_predictions_rescaled = scaler.inverse_transform(xgb_predictions)
            xgb_mse = mean_squared_error(y_test_rescaled, xgb_predictions_rescaled)
            print(f"XGBoost Mean Squared Error for {stock}: {xgb_mse}")

            # XGBoost Future Predictions (15 days)
            last_xgb_data = prepare_data_for_xgboost(last_scaled.reshape(1, time_steps, 2))
            future_predictions_xgb = []

            for _ in range(15):
                pred_open = xgb_open.predict(last_xgb_data)
                pred_close = xgb_close.predict(last_xgb_data)
                pred = np.array([pred_open[0], pred_close[0]])
                future_predictions_xgb.append(pred)
                last_scaled = np.append(last_scaled[1:], [pred], axis=0)
                last_xgb_data = prepare_data_for_xgboost(last_scaled.reshape(1, time_steps, 2))

            future_predictions_xgb_rescaled = scaler.inverse_transform(future_predictions_xgb)
            xgb_pred_df = pd.DataFrame(future_predictions_xgb_rescaled, columns=['Open', 'Close'], index=prediction_dates)
            xgb_pred_df.to_csv(os.path.join(PREDICTION_DIR, f"{stock}_xgb_predictions.csv"))
            print(f"✅ XGBoost Predictions for {stock} saved")

            # Save models
            joblib.dump(model, f'models/{stock}_lstm_model.pkl')
            joblib.dump(xgb_open, f'models/{stock}_xgb_open_model.pkl')
            joblib.dump(xgb_close, f'models/{stock}_xgb_close_model.pkl')
            print(f"Models for {stock} saved.")

def main():
    end = datetime.datetime.today()
    start = datetime.datetime(end.year-100,1,1)
    print(start)
    # Get all stock symbols listed in NSE
    url = 'https://en.wikipedia.org/wiki/NIFTY_500'
    nifty500_table = pd.read_html(url)[2]  # The first table contains the data
    print(nifty500_table)
    nifty500_tickers = nifty500_table[3].tolist()
    #print(sp500_tickers)
    print(len(nifty500_tickers))
    
    train_and_predict(nifty500_tickers[1:5])

main()