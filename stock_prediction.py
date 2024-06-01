# Imports
import plotly.graph_objects as go
import streamlit as st

# Import helper functions
from helper2 import *

# Import a bunch of machine learning models from sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import yfinance as yf
import time

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
    "XGBoost": XGBRegressor()
}

#Create a dict of top 10 stocks including FAANG
stock_dict = {
    "Apple Inc. 🍏": "AAPL",
    "Amazon.com Inc.📦": "AMZN",
    "Alphabet Inc. 🔤": "GOOGL",
    #"Facebook Inc.": "FB", # because stock info is NaN
    "Netflix Inc.🍿": "NFLX",
    "Tesla Inc.⚡": "TSLA",
    "Microsoft Corporation 🌐": "MSFT",
    "Alibaba Group Holding Limited 🧞‍♂️": "BABA",
    "NVIDIA Corporation 📟": "NVDA",
    "PayPal Holdings Inc. 💵": "PYPL",
    "GameStop Corp. 🎮": "GME",
    }

# Configure the page
st.set_page_config(
    page_title="Stock Price Predictor 3000",
    page_icon="📈",
)

#####Sidebar Start#####

# Add a sidebar
st.sidebar.markdown("## **User Input Features**")

# Add a dropdown for selecting the stock
st.sidebar.markdown("### **Select stock**")
stock = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))


# Add a dropdown for selecting the model
st.sidebar.markdown("### **Select a Model**")
model = st.sidebar.selectbox("Choose a model", list(models_dict.keys())) # need to create a dict for models instead of stocks.

# # Add a selector for stock exchange
# st.sidebar.markdown("### **Select stock exchange**")
# stock_exchange = st.sidebar.radio("Choose a stock exchange", ("BSE", "NSE"), index=0)

# # Build the stock ticker
# stock_ticker = f"{stock_dict[stock]}.{'BO' if stock_exchange == 'BSE' else 'NS'}"

stock_ticker = stock_dict[stock]
# ticker = yf.Ticker(stock_ticker)
# address = ticker.info['address1']
# city = ticker.info['city']
# state = ticker.info['state']
# country = ticker.info['country']
# industry = ticker.info['industry']
# sector = ticker.info['sectorDisp']
# CEO = ticker.info['companyOfficers'][0]['name']


# st.subheader(f"About {stock}")
# #put the stock name in the center of the list
# st.markdown(f"<h3 style='text-align: center;'></h3>", unsafe_allow_html=True)

# col1, col2 = st.columns(2)
# with col1:
#      st.markdown(f"**Address:** {address}, {city}, {state}, {country}")
#      st.markdown(f"**Industry:** {industry}")
# with col2:
#      st.markdown(f"**Sector:** {sector}")
#      st.markdown(f"**CEO:** {CEO}")



# Add a disabled input for stock ticker
st.sidebar.markdown("### **Stock ticker**")
st.sidebar.text_input(
    label="Stock ticker code", placeholder=stock_ticker, disabled=True
)

# Fetch and store periods and intervals
periods = fetch_periods_intervals()

# Add a selector for period
st.sidebar.markdown("### **Select period**")
period = st.sidebar.selectbox("Choose a period", list(periods.keys()))

# Add a selector for interval
st.sidebar.markdown("### **Select interval**")
interval = st.sidebar.selectbox("Choose an interval", periods[period])

#####Title#####

# Add title to the app
st.markdown("# **Stock Predictor 3000**")
col1, col2 = st.columns([1, 3])
# Display HTML with JavaScript to control GIF animation
# Display HTML with CSS animation to control GIF animation


# GIF
with col1:
    gif_url = "https://media.giphy.com/media/JpG2A9P3dPHXaTYrwu/giphy.gif"
    st.image(gif_url, use_column_width=True)
    time.sleep(3)
# Add a subtitle to the app
st.markdown("##### **Enhance Investment Decisions through Data-Driven Forecasting 💰**")


stock_ticker = stock_dict[stock]
ticker = yf.Ticker(stock_ticker)
address = ticker.info['address1']
city = ticker.info['city']
state = ticker.info['state']
country = ticker.info['country']
industry = ticker.info['industry']
sector = ticker.info['sectorDisp']
CEO = ticker.info['companyOfficers'][0]['name']


#st.subheader(f"About {stock}")
#put the stock name in the center of the list
st.markdown(f"<h2 style='text-align: left; color: blue;'>About {stock}</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
     st.markdown(f"**Address:** {address}, {city}, {state}, {country}")
     st.markdown(f"**Industry:** {industry}")
with col2:
     st.markdown(f"**Sector:** {sector}")
     st.markdown(f"**CEO:** {CEO}")

#####Title End#####


# Fetch the stock historical data
stock_data = fetch_stock_history(stock_ticker, period, interval)

#####Historical Data Graph#####

# Add a title to the historical data graph
st.markdown("## **Historical Data**")

# Create a plot for the historical data
fig = go.Figure(
    data=[
        go.Candlestick(
            x=stock_data.index,
            open=stock_data["Open"],
            high=stock_data["High"],
            low=stock_data["Low"],
            close=stock_data["Close"],
        )
    ]
)

# Customize the historical data graph
fig.update_layout(xaxis_rangeslider_visible=False)

# Use the native streamlit theme.
st.plotly_chart(fig, use_container_width=True)

#####Historical Data Graph End#####
#####Stock Prediction Graph#####

# Unpack the data
stock_data_close_train, stock_data_close_test, price_predictions_df = generate_stock_prediction(stock_ticker, model)

#st.write(forecast)
#st.write(predictions)


# Check if the data is not None
if price_predictions_df is not None:
    # Add a title to the stock prediction graph
    st.markdown("## **Stock Prediction**")

    fig = go.Figure(

        data=[
            go.Scatter(
                x=stock_data_close_train.index,
                y=stock_data_close_train["price"],
                name="Train",
                mode="lines",
                line=dict(color="blue"),
            ),
            go.Scatter(
                x=stock_data_close_test.index,
                y=stock_data_close_test["price"],
                name="Test",
                mode="lines",
                line=dict(color="orange"),
            ),
            go.Scatter(
                x=stock_data_close_test.index,
                y=price_predictions_df['price'],
                name="Forecast",
                mode="lines",
                line=dict(color="red"),
            ),
        ]
    )

    # Create a plot for the stock prediction
    # fig = go.Figure(

    #     data=[
    #         go.Scatter(
    #             x=train_df.index,
    #             y=train_df["price"],
    #             name="Train",
    #             mode="lines",
    #             line=dict(color="blue"),
    #         ),
    #         go.Scatter(
    #             x=test_df.index,
    #             y=test_df["price"],
    #             name="Test",
    #             mode="lines",
    #             line=dict(color="orange"),
    #         ),
    #         go.Scatter(
    #             x=forecast.index,
    #             y=forecast,
    #             name="Forecast",
    #             mode="lines",
    #             line=dict(color="red"),
    #         ),
    #         go.Scatter(
    #             x=test_df.index,
    #             y=predictions,
    #             name="Test Predictions",
    #             mode="lines",
    #             line=dict(color="green"),
    #         ),
    #     ]
    # )

    # Customize the stock prediction graph
    fig.update_layout(xaxis_rangeslider_visible=False)

    # Use the native streamlit theme.
    st.plotly_chart(fig, use_container_width=True)
    #Trouble shoot the bug we had to sshow if the model recommends to buy or sell stock
        # st.write(stock_data_close_test['price'].values[0])


def return_buy_sell_message():
    if price_predictions_df is not None and stock_data_close_test is not None:
        if price_predictions_df['price'][0] - stock_data_close_test['price'].values[0] > 0:
            return 'Buy This Stock 🫡'
        else:
            return 'Sell This Stock ⚠️'
    else:
        return None

# Get the recommendation message
recommendation_message = return_buy_sell_message()

# Display the message with center alignment
if recommendation_message is not None:
    st.markdown(f"<div style='text-align: center;'><u>The model recommends that you should {recommendation_message}</u></div>", unsafe_allow_html=True)
else:
    # Add a title to the stock prediction graph
    st.markdown("## **Stock Prediction**")
    # Add a message to the stock prediction graph
    st.markdown("### **No data available for the selected stock**")
