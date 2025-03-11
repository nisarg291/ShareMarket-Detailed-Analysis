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

# Directory Setup
DATA_DIR = 'data'
PREDICTION_DIR = 'predictions'
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(PREDICTION_DIR):
    os.makedirs(PREDICTION_DIR)

# NewsAPI Credentials
API_KEY = '63e9a0bd793f42f2b872411822f131ac'

# Fetch news using NewsAPI
def fetch_news_data(query, language='en', page_size=10):
    url = f'https://newsapi.org/v2/everything?q={query}&language={language}&pageSize={page_size}&apiKey={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('articles', [])
        headlines = [article['title'] for article in articles]
        return headlines
    return []

# Sentiment analysis using TextBlob
def get_news_sentiment(news_data):
    sentiment_scores = [TextBlob(headline).sentiment.polarity for headline in news_data]
    return np.mean(sentiment_scores) if sentiment_scores else 0

# Fetch Stock Data from Yahoo Finance
def fetch_stock_data(stock_symbol):
    df = pd.read_csv(f"nifty_stocks/{stock_symbol}_data.csv")
    if df.empty:
        print(f"No valid data for {stock_symbol}")
        return None
    return df

# Fetch Market Data (S&P 500 and Nifty500)
def fetch_market_data():
    sp500 = yf.download('^GSPC', start='2023-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d'))['Close']
    nifty500 = yf.download('^NSEI', start='2023-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d'))['Close']
    return sp500, nifty500

# Prepare data for LSTM model (Stock Prediction Model)
def prepare_data(df, time_steps=3650):
    df['Date'] = pd.to_datetime(df.index)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Open', 'Close']])
    X, y = [], []
    for i in range(time_steps, len(df)):
        X.append(df_scaled[i-time_steps:i, 0:2])
        y.append(df_scaled[i, 0:2])
        print("X and y", np.array(X), np.array(y))
    return np.array(X), np.array(y), scaler

# Prepare features for Market Impact Model
def prepare_market_data(sp500, nifty500, days=60):
    # Ensure we're only using the last 60 days of data for the markets
    sp500_recent = sp500[-days:].values
    nifty500_recent = nifty500[-days:].values
    #sentiment_array = np.full((days,), sentiment_score)  # Repeat sentiment score for 60 days

    # Create DataFrame with 1-dimensional arrays for each column
    market_data = pd.DataFrame({
        'SP500_Close': sp500.tail(60).values.flatten(),  # Last 60 days data for S&P 500
        'Nifty500_Close': nifty500.tail(60).values.flatten()  # Last 60 days data for Nifty500
    })
    
    return market_data


#LSTM Model for Stock Price Prediction
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=2))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Impact Assessment Model (e.g., RandomForest or XGBoost)
def create_impact_model():
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    return model

# Main training and prediction logic
def train_and_predict(stock_symbols):
    for stock in stock_symbols:
        df = fetch_stock_data(stock)
        if df is None:
            continue

        # Prepare Stock Data
        X, y, scaler = prepare_data(df)

        # Fetch Market Data (S&P 500, Nifty500)
        sp500, nifty500 = fetch_market_data()

        # Fetch News Sentiment for the Stock
        news_headlines = fetch_news_data(stock)
        print(news_headlines)
        sentiment_score = get_news_sentiment(news_headlines)
        print(sentiment_score)
        # Prepare Market Data for Impact Model
        market_data = prepare_market_data(sp500, nifty500)

        # Train the Stock Prediction Model (LSTM)
        X_train, X_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
        y_train, y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]

        lstm_model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
        lstm_model.fit(X_train, y_train, epochs=30, batch_size=32)

        # Predict Stock Prices
        predictions = lstm_model.predict(X_test)
        mse = mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(predictions))
        print(f"Mean Squared Error for {stock}: {mse}")

        # Train Market Impact Model (Random Forest / XGBoost)
        impact_model = create_impact_model()
        impact_model.fit(X_train, y_train)

        # Predict Impact of Market Conditions
        market_impact = impact_model.predict(X_test)
        print(market_impact)

        mse1 = mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(market_impact))
        print(f"Mean Squared Error for {stock}: {mse1}")

        # Adjust Stock Prediction Based on Market Impact
        if mse1 <mse:
            adjusted_predictions = market_impact
        else:
            adjusted_predictions = predictions
        # Inverse Transform the Predictions
        adjusted_predictions_rescaled = scaler.inverse_transform(adjusted_predictions)

        # Create DataFrame for Future Predictions (next 15 days)
        future_predictions = []
        for i in range(15):
            future_predictions.append(adjusted_predictions_rescaled[i])

        # Save Predictions and Models
        prediction_df = pd.DataFrame(future_predictions, columns=['Open', 'Close'])
        prediction_df.to_csv(os.path.join(PREDICTION_DIR, f"{stock}_predictions.csv"))
        print(f"Predictions saved for {stock}.")

        # Save both Models (Stock Prediction Model and Market Impact Model)
        joblib.dump(lstm_model, os.path.join(MODEL_DIR, f"{stock}_lstm_model.pkl"))
        joblib.dump(impact_model, os.path.join(MODEL_DIR, f"{stock}_impact_model.pkl"))


# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import xgboost as xgb
# import matplotlib.pyplot as plt

# def create_lstm_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=input_shape))

#     model.add(LSTM(50))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')
#     return model

def prepare_data_for_xgboost(X):
    # Reshape 3D array (samples, timesteps, features) to 2D (samples, timesteps*features)
    return X.reshape(X.shape[0], -1)

def train_and_compare_models(stock_symbols):
    results = {}
    
    for stock in stock_symbols:
        print(f"\nProcessing {stock}")
        
        # Fetch and prepare stock data
        df = fetch_stock_data(stock)
        if df is None:
            continue
            
        X, y, scaler = prepare_data(df)
        
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # LSTM Model
        lstm_model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
        lstm_model.fit(X_train, y_train, epochs=30, batch_size=32)
        lstm_predictions = lstm_model.predict(X_test)
        
        # XGBoost Model
        # Prepare data for XGBoost (convert 3D to 2D)
        X_train_xgb = prepare_data_for_xgboost(X_train)
        X_test_xgb = prepare_data_for_xgboost(X_test)
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_predictions = xgb_model.predict(X_test)
        
        # Inverse transform predictions and calculate MSE
        y_test_original = scaler.inverse_transform(y_test)
        lstm_pred_original = scaler.inverse_transform(lstm_predictions)
        xgb_pred_original = scaler.inverse_transform(xgb_predictions)
        
        lstm_mse = mean_squared_error(y_test_original, lstm_pred_original)
        xgb_mse = mean_squared_error(y_test_original, xgb_pred_original)
        
        # Store results
        results[stock] = {
            'lstm_mse': lstm_mse,
            'xgb_mse': xgb_mse,
            'lstm_predictions': lstm_pred_original,
            'xgb_predictions': xgb_pred_original,
            'actual': y_test_original
        }
        
        print(f"{stock} - LSTM MSE: {lstm_mse:.4f}")
        print(f"{stock} - XGBoost MSE: {xgb_mse:.4f}")
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_original, label='Actual')
        plt.plot(lstm_pred_original, label='LSTM Prediction')
        plt.plot(xgb_pred_original, label='XGBoost Prediction')
        plt.title(f'Stock Price Prediction Comparison - {stock}')
        plt.legend()
        plt.show()

    return results

# Example usage
if __name__ == "__main__":
    stock_symbols = ["TCS", "HDFCBANK", "ICICIBANK"]  # Add your stock symbols here
    results = train_and_compare_models(stock_symbols)
    
    # Print summary
    print("\nPerformance Summary:")
    for stock, metrics in results.items():
        print(f"\n{stock}:")
        print(f"LSTM MSE: {metrics['lstm_mse']:.4f}")
        print(f"XGBoost MSE: {metrics['xgb_mse']:.4f}")
        print(f"Better model: {'LSTM' if metrics['lstm_mse'] < metrics['xgb_mse'] else 'XGBoost'}")