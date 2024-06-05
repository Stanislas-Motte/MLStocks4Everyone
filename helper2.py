# Imports
import datetime as dt
import os
from pathlib import Path

import streamlit as st

# Import pandas
import pandas as pd
import pandas_ta as ta

# Import yfinance
import yfinance as yf

# Import a bunch of machine learning models from sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

#Import time series models
from statsmodels.tsa.arima.model import ARIMA

# Define training and testing size
TRAIN_SIZE = 0.7
TEST_SIZE = 0.3

# Create a dictionary of models
models_dict = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Elastic Net": ElasticNet(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Machine": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "ARIMA": ARIMA(),
}

#Define a dictionary for our saved models

# Create function to fetch stock name and id
#def fetch_stocks():
    # Load the data
    # df = pd.read_csv(Path.cwd() / "data" / "equity_issuers.csv")

    # Filter the data
    #df = df[["Security Code", "Issuer Name"]]

    # Create a dictionary
    #stock_dict = dict(zip(df["Security Code"], df["Issuer Name"]))

    # Return the dictionary
    #return stock_dict

# Create function to fetch periods and intervals
def fetch_periods_intervals():
    # Create dictionary for periods and intervals
    periods = {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }

    # Return the dictionary
    return periods

# Function to fetch the stock history
def fetch_stock_history(stock_ticker, period, interval):
    # Pull the data for the first security
    stock_data = yf.Ticker(stock_ticker)

    # Extract full of the stock
    stock_data_history = stock_data.history(period=period, interval=interval)[
        ["Open", "High", "Low", "Close"]
    ]

    #Keep only Close and date columns

    # Return the stock data
    return stock_data_history


import yfinance as yf
from datetime import datetime
def get_historical_prices(ticker, start_date, end_date):
    # Convert start_date and end_date to string format
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Get data on the ticker
    tickerData = yf.Ticker(ticker)

    # Get the historical prices for the ticker
    tickerDf = tickerData.history(period='1d', start=start_date_str, end=end_date_str)

    return tickerDf[['Close']]

get_historical_prices('AAPL', datetime(2020, 1, 1), datetime(2021, 1, 1))

def set_up_df(stock_ticker, time = '5y', interval = '1d'):

    df = fetch_stock_history(stock_ticker, time, interval)
    #df = df.loc[::-1].copy()

    #### Create all indicatiors

    # #Moving averages
    df['25_Day_EMA'] = ta.ema(df['Close'], length=25)
    df['50_Day_EMA'] = ta.ema(df['Close'], length=50)
    df['100_Day_EMA'] = ta.ema(df['Close'], length=100)
    df['200_Day_EMA'] = ta.ema(df['Close'], length=200)

    df['dist_from_25_EMA'] = (df['Close'] - df['25_Day_EMA']) / df['Close']
    df['dist_from_50_EMA'] = (df['Close'] - df['50_Day_EMA']) / df['Close']
    df['dist_from_100_EMA'] = (df['Close'] - df['100_Day_EMA']) / df['Close']
    df['dist_from_200_EMA'] = (df['Close'] - df['200_Day_EMA']) / df['Close']

    # #Relative Streingth Index
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_21'] = ta.rsi(df['Close'], length=21)
    df['RSI_28'] = ta.rsi(df['Close'], length=28)

    df['RSI_dist_from_50'] = df['RSI_14'] - 50
    df['RSI_dist_from_70'] = df['RSI_14'] - 70
    df['RSI_dist_from_30'] = df['RSI_14'] - 30

    df['next_day_close'] = df['Close'].shift(-1)
    df['next day return'] = (df['next_day_close'] - df['Close']) / df['Close']

    #Stochastic Oscillator
    df['stochastic_oscillator'] = (df['Close'] - df['Close'].rolling(window=14).min()) / (df['Close'].rolling(window=14).max() - df['Close'].rolling(window=14).min())

    #Parabolic SAR
    #df['Parabolic_SAR'] = ta.psar(df['High'], df['Low'], df['Close'], step=0.02, max_step=0.2, fillna = False)

    #Average True Range
    #df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])

    # #Code other indicators
    # df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    # df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
    # df['DPO'] = ta.dpo(df['Close'], length=20)
    # df['KST'] = ta.kst(df['Close'], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15)
    # df['KST_SIG'] = ta.kst_sig(df['Close'], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15)
    # df['KST_DIFF'] = ta.kst_diff(df['Close'], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15)
    # df['MACD_SIG'] = ta.macd_signal(df['Close'], fast=12, slow=26, signal=9)
    # df['MACD_DIFF'] = ta.macd_diff(df['Close'], fast=12, slow=26, signal=9)

    #Create all indicators
    #df['AVG_PRICE'] = df['Close'].rolling(window=2).mean()
    #df['25_Day_EMA'] = ta.ema(df['Close'], length=25)
    #df['50_Day_EMA'] = ta.ema(df['Close'], length=50)
    #df['stochastic_oscillator'] = (df['Close'] - df['Close'].rolling(window=14).min()) / (df['Close'].rolling(window=14).max() - df['Close'].rolling(window=14).min())
    #df['RSI'] = ta.rsi(df['Close'])
    #df['Tenkan_Sen'] = (df['Close'].rolling(window=9).max() + df['Close'].rolling(window=9).min()) / 2
    #df['Kijun_Sen'] = (df['Close'].rolling(window=26*days_in_week).max() + df['Close'].rolling(window=26*days_in_week).min()) / 2
    #df['Senkou_Span_A'] = (df['Tenkan_Sen'] + df['Kijun_Sen']) / 2
    #df['Senkou_Span_B'] = (df['Close'].rolling(window=52*days_in_week).max() + df['Close'].rolling(window=52*days_in_week).min()) / 2
    #df['Chikou_Span'] = df['Close'].shift(-26)

    #Create the y variable
    df['daily_return'] = (df['Close'].shift(-1) / df['Close'] -1)*100
    df['daily_return'] = df['daily_return']

    #Create X
    df = df.dropna()
    X = pd.DataFrame(df)
    #Drop multiple columns at once
    X = pd.DataFrame(df.drop(['Close', 'High', 'Low', 'Open', 'daily_return'], axis=1))

    #Y
    y = pd.DataFrame(df['daily_return'])

    return X, y


# Function to generate the stock prediction
def generate_stock_prediction(stock_ticker, model):

    #Treat time series diffrently
    if model == 'ARIMA':

        train_prices_df = get_historical_prices(stock_ticker, start_train_date, start_end_date)
        test_prices_df = get_historical_prices(stock_ticker, start_test_date, end_test_date)
        # Return the required data
        return train_prices_df, test_prices_df, price_predictions_df

    model = models_dict[model]

    #If the model is an ARIMA model, we will use the ARIMA model and return separately

    # Try to generate the predictions
    try:
        #Get X and y
        X = set_up_df(stock_ticker, time = '5y', interval = '1d')[0]
        y = set_up_df(stock_ticker, time = '5y', interval = '1d')[1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

        #Train the sklearn model on the X_train and y_train
        model.fit(X_train, y_train)

        #Get the earliest date in the X_train index
        start_train_date = X_train.index[0]
        start_end_date = X_train.index[-1]

        start_test_date = X_test.index[0]
        end_test_date = X_test.index[-1]

        #Get stock_data_close from start of train set to end of train set by filtering y_train
        train_prices_df = get_historical_prices(stock_ticker, start_train_date, start_end_date)
        test_prices_df = get_historical_prices(stock_ticker, start_test_date, end_test_date)

        #Get the first price of the test set
        first_price = test_prices_df['Close'][0]

        #Predict the X_test
        predictions = model.predict(X_test)

        #Get the factor to multiply the price by by doing cumprod predictions
        factor = (1 + predictions/100).cumprod()
        price_predictions = first_price * factor

        #Add the dates of the test set to the price predictions in a  df
        price_predictions_df = pd.DataFrame(price_predictions, index = X_test.index, columns = ['Close'])

        # Return the required data
        return train_prices_df, test_prices_df, price_predictions_df

    # If error occurs
    except:
        # Return None
        return None, None, None
