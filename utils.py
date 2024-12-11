import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import MACD
from ta.momentum import RSIIndicator


def plot_stock_data(ticker, period, sma_period, ema_period, bb_period, bb_std, macd_fast, macd_slow, macd_signal, rsi_period, strategy):
    # Fetch the stock data using yfinance
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period=period)

    # Check if the fetched data is empty
    if hist.empty:
        st.error(f"No data found for {ticker} over the specified period.")
        return

    # Initialize signals and positions
    hist['Signal'] = 0.0
    hist['Position'] = 0.0

    if strategy == "Moving Average Crossover":
        # Ensure parameters are not None
        if sma_period is None or ema_period is None:
            st.error(
                "SMA and EMA periods must be specified for Moving Average Crossover.")
            return

        # Calculate Simple Moving Average (SMA) and Exponential Moving Average (EMA)
        hist['SMA'] = hist['Close'].rolling(window=sma_period).mean()
        hist['EMA'] = hist['Close'].ewm(span=ema_period, adjust=False).mean()

        # Ensure we have enough data to calculate signals
        if len(hist) < max(sma_period, ema_period):
            st.error(
                f"Not enough data to calculate {sma_period}-day SMA or {ema_period}-day EMA for {ticker}.")
            return

        # Generate buy and sell signals for SMA/EMA crossover
        hist.loc[hist.index[sma_period:], 'Signal'] = np.where(
            hist.loc[hist.index[sma_period:],
                     'EMA'] > hist.loc[hist.index[sma_period:], 'SMA'], 1.0, 0.0
        )
        hist['Position'] = hist['Signal'].diff()

        # Plot the closing price, SMA, and EMA
        plt.figure(figsize=(14, 7))
        plt.plot(hist['Close'], label=f'{ticker} - Close Price', alpha=0.5)
        plt.plot(
            hist['SMA'], label=f'{ticker} - {sma_period}-day SMA', alpha=0.75)
        plt.plot(
            hist['EMA'], label=f'{ticker} - {ema_period}-day EMA', alpha=0.75)

        # Plot buy and sell signals for SMA/EMA crossover
        plt.plot(hist[hist['Position'] == 1].index,
                 hist['EMA'][hist['Position'] == 1],
                 '^', markersize=10, color='g', lw=0, label=f'{ticker} - Buy Signal')
        plt.plot(hist[hist['Position'] == -1].index,
                 hist['EMA'][hist['Position'] == -1],
                 'v', markersize=10, color='r', lw=0, label=f'{ticker} - Sell Signal')

        # Add title and legend to the plot
        plt.title(f'{ticker} - Close Price, SMA, and EMA')
        plt.legend()
        st.pyplot(plt)

    elif strategy == "Bollinger Bands":
        # Ensure parameters are not None
        if bb_period is None or bb_std is None:
            st.error(
                "Bollinger Bands period and standard deviations must be specified.")
            return

        # Calculate Bollinger Bands
        hist['Middle_Band'] = hist['Close'].rolling(window=bb_period).mean()
        hist['Upper_Band'] = hist['Middle_Band'] + \
            (hist['Close'].rolling(window=bb_period).std() * bb_std)
        hist['Lower_Band'] = hist['Middle_Band'] - \
            (hist['Close'].rolling(window=bb_period).std() * bb_std)

        # Generate buy and sell signals for Bollinger Bands
        hist.loc[hist['Close'] < hist['Lower_Band'], 'Signal'] = 1.0
        hist.loc[hist['Close'] > hist['Upper_Band'], 'Signal'] = -1.0
        hist['Position'] = hist['Signal'].diff()

        # Plot the closing price and Bollinger Bands
        plt.figure(figsize=(14, 7))
        plt.plot(hist['Close'], label=f'{ticker} - Close Price', alpha=0.5)
        plt.plot(hist['Middle_Band'],
                 label=f'{ticker} - {bb_period}-day Middle Band', alpha=0.75)
        plt.plot(hist['Upper_Band'],
                 label=f'{ticker} - Upper Band', alpha=0.75)
        plt.plot(hist['Lower_Band'],
                 label=f'{ticker} - Lower Band', alpha=0.75)

        # Plot buy and sell signals for Bollinger Bands
        plt.plot(hist[hist['Position'] == 1].index,
                 hist['Lower_Band'][hist['Position'] == 1],
                 '^', markersize=10, color='g', lw=0, label=f'{ticker} - Buy Signal')
        plt.plot(hist[hist['Position'] == -1].index,
                 hist['Upper_Band'][hist['Position'] == -1],
                 'v', markersize=10, color='r', lw=0, label=f'{ticker} - Sell Signal')

        # Add title and legend to the plot
        plt.title(f'{ticker} - Close Price and Bollinger Bands')
        plt.legend()
        st.pyplot(plt)

    elif strategy == "MACD":
        # Ensure parameters are not None
        if macd_fast is None or macd_slow is None or macd_signal is None:
            st.error("MACD fast, slow, and signal periods must be specified.")
            return

        # Calculate MACD
        macd_indicator = MACD(hist['Close'], window_slow=macd_slow,
                              window_fast=macd_fast, window_sign=macd_signal)
        hist['MACD'] = macd_indicator.macd()
        hist['Signal_Line'] = macd_indicator.macd_signal()
        hist['MACD_Histogram'] = macd_indicator.macd_diff()

        # Generate buy and sell signals for MACD
        hist['Signal'] = 0.0
        hist['Signal'][hist['MACD'] > hist['Signal_Line']] = 1.0
        hist['Signal'][hist['MACD'] < hist['Signal_Line']] = -1.0
        hist['Position'] = hist['Signal'].diff()

        # Plot the closing price, MACD, Signal Line, and MACD Histogram
        plt.figure(figsize=(14, 7))
        plt.plot(hist['Close'], label=f'{ticker} - Close Price', alpha=0.5)

        # Plot MACD and Signal Line
        plt.figure(figsize=(14, 7))
        plt.plot(hist['MACD'], label=f'{ticker} - MACD Line', alpha=0.75)
        plt.plot(hist['Signal_Line'],
                 label=f'{ticker} - Signal Line', alpha=0.75)

        # Plot MACD Histogram
        plt.bar(hist.index, hist['MACD_Histogram'],
                label=f'{ticker} - MACD Histogram', color='purple', alpha=0.5)

        # Plot buy and sell signals for MACD
        plt.plot(hist[hist['Position'] == 2].index,
                 hist['Signal_Line'][hist['Position'] == 2],
                 '^', markersize=10, color='g', lw=0, label=f'{ticker} - Buy Signal')
        plt.plot(hist[hist['Position'] == -2].index,
                 hist['Signal_Line'][hist['Position'] == -2],
                 'v', markersize=10, color='r', lw=0, label=f'{ticker} - Sell Signal')

        # Add title and legend to the plot
        plt.title(f'{ticker} - MACD, Signal Line, and Histogram')
        plt.legend()
        st.pyplot(plt)

    elif strategy == "RSI":
        # Ensure parameters are not None
        if rsi_period is None:
            st.error("RSI period must be specified.")
            return

        # Calculate RSI
        rsi_indicator = RSIIndicator(hist['Close'], window=rsi_period)
        hist['RSI'] = rsi_indicator.rsi()

        # Generate buy and sell signals for RSI
        hist['Signal'] = 0.0
        hist['Signal'][hist['RSI'] < 30] = 1.0  # Buy signal
        hist['Signal'][hist['RSI'] > 70] = -1.0  # Sell signal
        hist['Position'] = hist['Signal'].diff()

        # Plot the closing price and RSI
        plt.figure(figsize=(14, 7))
        plt.plot(hist['Close'], label=f'{ticker} - Close Price', alpha=0.5)

        # Plot RSI
        plt.figure(figsize=(14, 7))
        plt.plot(hist['RSI'], label=f'{ticker} - RSI', alpha=0.75)
        plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')

        # Plot buy and sell signals for RSI
        plt.plot(hist[hist['Position'] == 1].index,
                 hist['RSI'][hist['Position'] == 1],
                 '^', markersize=10, color='g', lw=0, label=f'{ticker} - Buy Signal')
        plt.plot(hist[hist['Position'] == -1].index,
                 hist['RSI'][hist['Position'] == -1],
                 'v', markersize=10, color='r', lw=0, label=f'{ticker} - Sell Signal')

        # Add title and legend to the plot
        plt.title(f'{ticker} - Close Price and RSI')
        plt.legend()
        st.pyplot(plt)

    else:
        st.error("Invalid strategy selected.")
        return

    # Backtest the strategy
    backtest_results = backtest_strategy(hist, ticker, strategy)
    st.write(backtest_results)


# Function to backtest the strategy


def backtest_strategy(hist, ticker, strategy):
    # Initialize variables
    # Calculate daily returns
    hist['Returns'] = hist['Close'].pct_change()
    # Calculate strategy returns by multiplying daily returns by the previous day's signal
    hist['Strategy_Returns'] = hist['Returns'] * hist['Signal'].shift(1)
    # Calculate cumulative returns for the strategy
    hist['Cumulative_Returns'] = (1 + hist['Strategy_Returns']).cumprod()
    # Calculate cumulative returns for the benchmark (buy and hold)
    hist['Cumulative_Benchmark_Returns'] = (1 + hist['Returns']).cumprod()

    # Calculate performance metrics
    # Total return for the strategy
    total_return = hist['Cumulative_Returns'].iloc[-1] - 1
    # Total return for the benchmark
    benchmark_return = hist['Cumulative_Benchmark_Returns'].iloc[-1] - 1
    # Number of trades (each trade consists of a buy and a sell signal)
    num_trades = hist['Position'].abs().sum() / 2
    # Win rate (percentage of profitable trades)
    win_rate = (hist['Strategy_Returns'] > 0).sum() / \
        (hist['Strategy_Returns'] != 0).sum()
    # Average profit per trade
    avg_profit = hist[hist['Strategy_Returns'] > 0]['Strategy_Returns'].mean()
    # Average loss per trade
    avg_loss = hist[hist['Strategy_Returns'] < 0]['Strategy_Returns'].mean()

    # Plot cumulative returns
    plt.figure(figsize=(14, 7))
    plt.plot(hist['Cumulative_Returns'],
             label=f'{ticker} - Strategy Cumulative Returns', alpha=0.75)
    plt.plot(hist['Cumulative_Benchmark_Returns'],
             label=f'{ticker} - Benchmark Cumulative Returns', alpha=0.75)
    plt.title(f'{ticker} - Cumulative Returns')
    plt.legend()
    st.pyplot(plt)

    # Create a dictionary to store performance metrics
    performance_metrics = f"""
       :blue[**{strategy.upper()} ANALYSIS**] 

        **Total Return : {total_return:.2%}**. It represents the percentage increase or decrease in the value of the investment from the start to the end of the period.

        **Benchmark Return : {benchmark_return:.2%}**, which in this case is the closing price of the stock without any strategy. It serves as a reference point to compare the performance of the strategy against simply holding the stock.

        **Total Trades : {int(num_trades)}**.  It includes both buy and sell actions taken executed during the backtest.

        **Win Rate : {win_rate:.2%}**. It indicates how often :blue[{strategy}] strategy made a profit.

        **Average Profit : {avg_profit:.2%}**. It provides insight into the typical profit generated by successful trades.

        **Average Loss per Trade: {avg_loss:.2%}**. Average loss incurred on losing trades. It helps to understand the typical loss experienced by unsuccessful trades.

    """

    return performance_metrics
