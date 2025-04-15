import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from textblob import TextBlob
from sklearn.metrics import mean_squared_error

# Create necessary directories
os.makedirs("predictions", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate moving averages
def calculate_moving_averages(data, windows=[15, 50]):
    for window in windows:
        data[f'MA{window}'] = data['Close'].rolling(window).mean()
    return data

# Function to calculate MACD
def calculate_macd(data):
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    data['MACD'] = macd
    data['MACD_Signal'] = signal
    return data

# Function to fetch and preprocess stock data
def fetch_and_preprocess_data(stock, start_date, end_date):
    data = yf.download(f"{stock}.NS", start=start_date, end=end_date)
    if data.empty:
        print(f"No data found for {stock}")
        return None
    data['RSI'] = calculate_rsi(data)
    data = calculate_moving_averages(data)
    data = calculate_macd(data)
    return data.dropna()

# Function to prepare data for training
def prepare_data(data, lookback=60):
    feature_cols = ['Open', 'Close', 'High', 'Low', 'Volume', 'RSI', 'MA15', 'MA50', 'MACD', 'MACD_Signal']
    
    if not all(col in data.columns for col in feature_cols):
        print("Missing feature columns.")
        return None, None, None, None
    
    features = data[feature_cols]
    targets = data['Close']
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    features_scaled = feature_scaler.fit_transform(features)
    targets_scaled = target_scaler.fit_transform(targets.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(data) - 5):
        X.append(features_scaled[i - lookback:i])
        y.append(targets_scaled[i:i + 5].flatten())  
    
    return np.array(X), np.array(y), feature_scaler, target_scaler

# Function to create CNN-LSTM model
def create_cnn_lstm_model(input_shape, output_size=5):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to train the model
def train_model(X_train, y_train):
    model = create_cnn_lstm_model((X_train.shape[1], X_train.shape[2]), output_size=5)
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)
    return model

# Function to predict next 5 days
def predict_next_5_days(model, last_data, feature_scaler, target_scaler, lookback=60):
    feature_cols = ['Open', 'Close', 'High', 'Low', 'Volume', 'RSI', 'MA15', 'MA50', 'MACD', 'MACD_Signal']
    
    if last_data.shape[0] < lookback:
        print("Not enough historical data for predictions.")
        return np.array([])

    input_data = last_data[-lookback:][feature_cols].values
    input_data_scaled = feature_scaler.transform(input_data)
    input_data_scaled = input_data_scaled[np.newaxis, :, :]

    pred_scaled = model.predict(input_data_scaled, verbose=0)[0]
    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    return pred

# Function to fetch news headlines (placeholder)
def fetch_news_data(stock):
    return [f"Positive news about {stock}", "Market uncertainty reported"]

# Function to calculate sentiment score
def get_sentiment_score(news_headlines):
    sentiments = [TextBlob(headline).sentiment.polarity for headline in news_headlines]
    return np.mean(sentiments) if sentiments else 0

# Main function to run prediction
def predict_stock(stock):
    today = pd.to_datetime("today")
    start_date = today - pd.Timedelta(days=50*365)
    
    data = fetch_and_preprocess_data(stock, start_date, today)
    if data is None:
        return
    
    lookback = 60
    X, y, feature_scaler, target_scaler = prepare_data(data, lookback)
    
    if X is None or y is None:
        print(f"Error preparing data for {stock}.")
        return
    
    train_size = int(len(X) * 0.95)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = train_model(X_train, y_train)

    # Save the trained model
    model_path = f"models/{stock}_model.h5"
    model.save(model_path)
    print(f"Model saved at {model_path}")
    
    # **Test the model on X_test and compare with y_test**
    predictions_scaled = model.predict(X_test)
    
    # Inverse transform to get actual values
    y_test_actual = target_scaler.inverse_transform(y_test)
    predictions_actual = target_scaler.inverse_transform(predictions_scaled)

    # **Evaluate performance using RMSE**
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
    
    print(f"\nTest set evaluation for {stock}:")
    print(f"RMSE for model: {rmse}")

    # Save actual vs predicted values
    results_df = pd.DataFrame({
        "Actual": y_test_actual.flatten(),
        "Predicted": predictions_actual.flatten()
    })
    results_path = f"predictions/{stock}_predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved at {results_path}")

    # **Predict future prices**
    last_data = data[-lookback:]
    future_predictions = predict_next_5_days(model, last_data, feature_scaler, target_scaler, lookback)
    
    if future_predictions.size == 0:
        print(f"Prediction failed for {stock}")
        return
    
    prediction_dates = pd.date_range(start=today + pd.Timedelta(days=1), periods=5)
    prediction_df = pd.DataFrame({'Close': future_predictions}, index=prediction_dates)
    
    future_predictions_path = f"predictions/{stock}_future_predictions.csv"
    prediction_df.to_csv(future_predictions_path)
    print(f"Future predictions saved at {future_predictions_path}")

# **Run Predictions**
predict_stock("TCS")
predict_stock("RELIANCE")
