# Faizan-Dashboard
Faizan Dashboard Profile 
# S&P 500 Trend Prediction with Sentiment

Pipeline:

1. `download_data.py`  
   - Downloads S&P 500 OHLCV data to `data/historical_ohlcv.parquet`.

2. `fetch_news.py`  
   - Fetches financial news and computes TextBlob sentiment.  
   - Saves daily average sentiment per ticker to `data/news_sentiment.parquet`.

3. `merge_data.py`  
   - Spark-based distributed merge of OHLCV and sentiment.  
   - Outputs `data/master_dataset.parquet`.

4. `feature_engineering.py`  
   - Computes technical indicators, creates Trend label, performs feature selection.  
   - Outputs:
     - `data/features_dataset.parquet`
     - `data/feature_importance.csv`.

5. `train_lstm.py`  
   - Trains LSTM for trend prediction:
     - Experiment 1: without sentiment
     - Experiment 2: with sentiment  
   - Outputs:
     - `data/results_without_sentiment.parquet`
     - `data/results_with_sentiment.parquet`.

6. `Faizan-Dashboard.py`  
   - Streamlit dashboard for visualization of:
     - Trend prediction performance (DA, MAE, RMSE)
     - Feature importance
     - Daily sentiment
     - Demo price + indicators chart.
