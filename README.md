# 📈 Stock Price Prediction with News Sentiment and Deep Learning

This project is a robust stock prediction framework that uses **historical price data**, **technical indicators**, and **news sentiment** to forecast future prices for Indian stocks (NIFTY 500). It employs machine learning and deep learning models including LSTM, XGBoost, and CNN-LSTM.

---

## 🔧 Features

- 📊 **Time Series Forecasting** with LSTM and XGBoost
- 📰 **News Sentiment Analysis** using TextBlob and NewsAPI
- 📈 **Technical Indicators** (RSI, MACD, Moving Averages)
- 🧠 **Deep Learning** using Keras and TensorFlow (CNN + LSTM Hybrid)
- 🗃️ **Automatic Model Saving & Evaluation** using joblib and CSV export
- 📁 Organized into `models/`, `data/`, and `predictions/`

---

## 📂 Project Structure

| File                  | Purpose |
|-----------------------|---------|
| `main.py`             | End-to-end pipeline using LSTM/XGBoost + sentiment |
| `test (1).py`         | Model benchmarking + market impact model (XGBoost) |
| `CNN_LSTM (1).py`     | Advanced CNN-LSTM model with technical indicators |
| `test1.py`            | Linked list implementation (for data structure practice) |

---

## 🧠 Models Used

- **LSTM**: Sequential forecasting using past time steps
- **XGBoost**: Gradient Boosted Trees for tabular predictions
- **CNN-LSTM**: Feature extraction using convolution layers followed by sequence learning

---

## 🧪 Data Sources

- **Yahoo Finance**: Historical stock data
- **NewsAPI**: Real-time news headlines
- **Wikipedia**: NIFTY 500 stock symbols

---

## 🧰 Dependencies

Install all required libraries before running:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:

```bash
pip install yfinance pandas numpy keras scikit-learn joblib textblob xgboost nltk matplotlib
python -m textblob.download_corpora
```

---

## 🚀 How to Run

### ▶️ Method 1: Run `main.py` (LSTM + XGBoost + Sentiment)

```bash
python main.py
```

This will:
- Automatically fetch data for NIFTY 500 stocks
- Train LSTM and XGBoost models
- Predict next 15 days of stock prices
- Adjust predictions using sentiment analysis
- Save results in `/predictions/` and models in `/models/`

### ▶️ Method 2: Run `test (1).py` for Benchmarking

```bash
python "test (1).py"
```
This will:
- Compare LSTM vs XGBoost models
- Include market indicators like NIFTY 500 and S&P 500
- Print model performance and save predictions

### ▶️ Method 3: Run `CNN_LSTM (1).py` for Advanced Forecasting

```bash
python "CNN_LSTM (1).py"
```
This will:
- Use RSI, MACD, Moving Averages as input features
- Train a CNN-LSTM model
- Predict next 5 days of stock prices and evaluate performance

---

## 📊 Output

- **Predictions**: Saved as CSV files in `predictions/`
- **Models**: Saved using joblib in `models/`
- **Evaluation Metrics**: MSE/RMSE printed in console

---

## 📌 To Do / Future Work

- Integrate real-time dashboard using Dash/Streamlit
- Add transformer-based models (e.g., BERT for financial sentiment)
- Extend support to global stock markets

---

## 👨‍💻 Author

**Nisarg Adalja**  
🔗 [GitHub](https://github.com/nisarg291) | [LinkedIn](https://www.linkedin.com/in/nisarg-adalja-446434197/)
