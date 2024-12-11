import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import MACD
from ta.momentum import RSIIndicator
from utils import plot_stock_data

# Configure the Streamlit app with a title, layout, icon, and initial sidebar state
st.set_page_config(page_title="Equity Trading",
                   layout="centered",
                   page_icon="💰",
                   initial_sidebar_state="expanded")

# Streamlit app
st.title(":blue[Equity Trading Strategies]")
st.write("This app is designed to assist investors and traders in assessing the effectiveness of various trading strategies by backtesting historical stock data, supporting informed decision-making and strategy optimization.")

# sidebar widgets
sidebar_widget = st.sidebar
exchange = sidebar_widget.selectbox(label="Choose a market",
                                    options=["SGX", "NYSE"])

# Load the CSV file containing stock symbols and names
# Create a dictionary for the dropdown menu
try:
    stocks_df = pd.read_csv(f"./resource/{exchange}.csv")
except FileNotFoundError:
    st.error(
        "The csv file was not found. Please ensure the file is in the same directory as the script.")
    stocks_df = pd.DataFrame(columns=['Symbol', 'Name'])

stock_options = dict(zip(stocks_df['Symbol'], stocks_df['Name']))

# User inputs
# Create a multiselect dropdown menu for stock selection
tickers = sidebar_widget.multiselect("Choose one or more counters", list(
    stock_options.keys()), format_func=lambda x: f"{x} - {stock_options[x]}")

# Define strategy parameters and their default states
strategy_params = {
    "Moving Average Crossover": {
        "sma_period": {"min_value": 1, "value": 50},
        "ema_period": {"min_value": 1, "value": 20},
    },
    "Bollinger Bands": {
        "bb_period": {"min_value": 1, "value": 20},
        "bb_std": {"min_value": 0.01, "value": 2.0},
    },
    "MACD": {
        "macd_fast": {"min_value": 1, "value": 12},
        "macd_slow": {"min_value": 1, "value": 26},
        "macd_signal": {"min_value": 1, "value": 9},
    },
    "RSI": {
        "rsi_period": {"min_value": 1, "value": 14},
    },
}

if len(tickers):
    strategy = sidebar_widget.selectbox(label="Choose a trading strategy", options=[
        "", "Moving Average Crossover", "Bollinger Bands", "MACD", "RSI"], )

    if strategy != "":
        # Create a selectbox for time period selection
        period = sidebar_widget.selectbox("Select Time Period:", [
            "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], index=5)

        # Get the parameters for the selected strategy
        params = strategy_params.get(strategy, {})

        # Initialize variables to hold the parameter values
        sma_period = None
        ema_period = None
        bb_period = None
        bb_std = None
        macd_fast = None
        macd_slow = None
        macd_signal = None
        rsi_period = None

        # Create number inputs for each parameter based on the selected strategy
        if "sma_period" in params:
            sma_period = sidebar_widget.number_input(
                "Enter SMA Period (days):", **params["sma_period"])
        if "ema_period" in params:
            ema_period = sidebar_widget.number_input(
                "Enter EMA Period (days):", **params["ema_period"])
        if "bb_period" in params:
            bb_period = sidebar_widget.number_input(
                "Enter Bollinger Bands Period (days):", **params["bb_period"])
        if "bb_std" in params:
            bb_std = sidebar_widget.number_input(
                "Enter Bollinger Bands Standard Deviations:", **params["bb_std"])
        if "macd_fast" in params:
            macd_fast = sidebar_widget.number_input(
                "Enter MACD Fast Period (days):", **params["macd_fast"])
        if "macd_slow" in params:
            macd_slow = sidebar_widget.number_input(
                "Enter MACD Slow Period (days):", **params["macd_slow"])
        if "macd_signal" in params:
            macd_signal = sidebar_widget.number_input(
                "Enter MACD Signal Period (days):", **params["macd_signal"])
        if "rsi_period" in params:
            rsi_period = sidebar_widget.number_input(
                "Enter RSI Period (days):", **params["rsi_period"])

        # Button to trigger the analysis
        if sidebar_widget.button("Analyze"):

            for ticker in tickers:
                st.divider()
                st.subheader(stock_options[ticker])
                plot_stock_data(ticker, period, sma_period, ema_period, bb_period,
                                bb_std, macd_fast, macd_slow, macd_signal, rsi_period, strategy)
