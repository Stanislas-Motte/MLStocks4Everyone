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


def set_up_df(stock_data_close):

    days_in_week = 7

    df = stock_data_close
    #df = df.loc[::-1].copy()

    df.reset_index(inplace=True)

    #Create all indicators
    df['AVG_PRICE'] = df['price'].rolling(window=2).mean()
    df['25_Day_EMA'] = ta.ema(df['price'], length=25)
    df['50_Day_EMA'] = ta.ema(df['price'], length=50)
    df['stochastic_oscillator'] = (df['price'] - df['price'].rolling(window=14).min()) / (df['price'].rolling(window=14).max() - df['price'].rolling(window=14).min())
    df['RSI'] = ta.rsi(df['price'])
    df['Tenkan_Sen'] = (df['price'].rolling(window=9).max() + df['price'].rolling(window=9).min()) / 2
    #df['Kijun_Sen'] = (df['price'].rolling(window=26*days_in_week).max() + df['price'].rolling(window=26*days_in_week).min()) / 2
    #df['Senkou_Span_A'] = (df['Tenkan_Sen'] + df['Kijun_Sen']) / 2
    #df['Senkou_Span_B'] = (df['price'].rolling(window=52*days_in_week).max() + df['price'].rolling(window=52*days_in_week).min()) / 2
    #df['Chikou_Span'] = df['price'].shift(-26)

    #Create the y variable
    df['daily_return'] = (df['price'].shift(-1) / df['price'] -1)*100

    df['daily_return'] = df['daily_return']

    #Create X and y
    df = df.dropna()
    X = pd.DataFrame(df)
    X = pd.DataFrame(df.drop(['price'], axis=1))
    X = X.drop(['daily_return'], axis=1)
    #X = X.drop(['index'], axis = 1)
    y = pd.DataFrame(df['daily_return'])

    #Reset index as the date
    X.set_index('DATETIME', inplace=True)
    #Drop old index
    X = X.drop(['index'], axis = 1)


    return X, y

# Function to fetch the stock history
def fetch_stock_history(stock_ticker, period, interval):
    # Pull the data for the first security
    stock_data = yf.Ticker(stock_ticker)

    # Extract full of the stock
    stock_data_history = stock_data.history(period=period, interval=interval)[
        ["Open", "High", "Low", "Close"]
    ]

    # Return the stock data
    return stock_data_history


# Function to generate the stock prediction
def generate_stock_prediction(stock_ticker, model):

    model = models_dict[model]
    # Try to generate the predictions
    try:
        # Pull the data for the first security
        stock_data = yf.Ticker(stock_ticker)

        # Extract the data for last 1yr with 1d interval
        stock_data_hist = stock_data.history(period="2y", interval="1d")

        # Clean the data for to keep only the required columns
        stock_data_close = stock_data_hist[["Close"]]

        # Change frequency to day
        stock_data_close = stock_data_close.asfreq("D", method="ffill")

        # Fill missing values
        stock_data_close = stock_data_close.ffill()

        stock_data_close.reset_index(inplace=True)

        stock_data_close.columns = ['DATETIME', 'price']

        #Create all the relevant features
        X, y = set_up_df(stock_data_close)



        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

        #Train the sklearn model on the X_train and y_train
        model.fit(X_train, y_train)

        #Get the earliest date in the X_train index
        start_train_date = X_train.index[0]
        start_end_date = X_train.index[-1]

        start_test_date = X_test.index[0]
        end_test_date = X_test.index[-1]

        #Get stock_data_close from start of train set to end of train set
        stock_data_close_train = stock_data_close[stock_data_close['DATETIME'] >= start_train_date]

        #Get stock_data close from start of test set to end of test set
        stock_data_close_test = stock_data_close[stock_data_close['DATETIME'] >= start_test_date]

        #Get the first price of the test set
        first_price = stock_data_close_test['price'].iloc[0]

        #Predict the X_test
        predictions = model.predict(X_test)

        #Get the factor to multiply the price by by doing cumprod predictions
        factor = (1 + predictions/100).cumprod()
        price_predictions = first_price * factor

        #Add the dates of the test set to the price predictions in a  df
        price_predictions_df = pd.DataFrame(price_predictions, index = X_test.index, columns = ['price'])

        # Return the required data
        return stock_data_close_train, stock_data_close_test, price_predictions_df

    # If error occurs
    except:
        # Return None
        return None, None, None
