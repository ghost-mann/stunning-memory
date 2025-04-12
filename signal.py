import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def load_data(filepath):
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return data

def add_technical_indicators(data):
    # copy to avoid modifying original
    df = data.copy()

    # simple moving average
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['CLose'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    # exponential moving average
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] = df['MACD_Signal']

    # RSI
    delta = df['CLose'].diff()
    gain = delta.where(delta > 0, 0)
    loss = delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    # Calculate RSI using SMA for first average, then EMA for subsequent values
    for i in range(14, len(df)):
        if i == 14:
            avg_gain[i] = gain[1:15].mean()
            avg_loss[i] = loss[1:15].mean()
        else:
            avg_gain[i] = (avg_gain[i - 1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i - 1] * 13 + loss[i]) / 14

        # calculate RS and RSI
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).max()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

        # Average True Range(ATR)
        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
        low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))

        true_range = pd.DataFrame({
            'high_low': high_low,
            'high_close_prev': high_close_prev,
            'low_close_prev': low_close_prev,
        }).max(axis=1)

        df['ATR'] = true_range.rolling(window=14).mean()

        # stochastic oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['CLose'] - low_14) / (high_14 - low_14))
        df['%D'] = df['%K'].rolling(window=3).mean()

        return df


    def generate_signals(df):
        # generate signals based on technical indicators
        signals = pd.DataFrame(index=df.index)
        signals['price'] = df['Close']
        signals['signal'] = 0 # 0=neutral 1=buy -1=sell

        # MA crossover signals
        signals.loc[df['SMA2O'] > df['SMA50'], 'ma_cross_signal'] = 1
        signals.loc[df['SMA20'] < df['SMA50'], 'ma_cross_signal'] = -1

        # MACD crossing signals
        signals.loc[df['MACD'] > df['MACD_Signal'], 'macd_signal'] = 1
        signals.loc[df['MACD'] < df['MACD_Signal'], 'macd_signal'] = -1

        # RSI signals
        signals.loc[df['RSI'] < 30, 'rsi_signal'] = 1
        signals.loc[df['RSI'] > 70 , 'rsi_signal'] = -1

        # Bollinger Bands signals
        signals.loc[df['Close'] < df['BB_Lower'], 'bb_signal'] = 1  # Price below lower band (potential buy)
        signals.loc[df['Close'] > df['BB_Upper'], 'bb_signal'] = -1  # Price above upper band (potential sell)

        # Combined signal - simple average of all signals
        signal_columns = ['ma_cross_signal', 'macd_signal', 'rsi_signal', 'bb_signal']
        signals['combined_signal'] = signals[signal_columns].mean(axis=1)

        # Convert to discrete signals (-1, 0, 1)
        signals['signal'] = np.where(signals['combined_signal'] > 0.5, 1,
                                     np.where(signals['combined_signal'] < -0.5, -1, 0))

        # Add signal strength (absolute value of combined signal)
        signals['signal_strength'] = np.abs(signals['combined_signal'])

        return signals


def plot_signals(data, signals, last_days=60):

    # plot the price chart with buy/sell signals
    plt.figure(figsize=(12, 8))

    # get the last N days of data
    data = data.iloc[-last_days:]
    signals = signals.iloc[-last_days:]

    # plot price
    plt.plot(data.index, data['CLose'], label='XAUUSD Price', color='blue')

    # plot moving average
    plt.plot(data.index, data['SMA20'], label='SMA 20', color='orange', alpha=0.7)
    plt.plot(data.index, data['SMA50'] , label='SMA 50', color='green', alpha=0.7)

    # plot bollinger bands
    plt.plot(data.index, data['BB_Upper'], '--', color='gray', alpha=0.6)
    plt.plot(data.index, data['BB_Lower'], '--', color='gray', alpha=0.6)

    # plot buy signals
    buy_signals = signals[signals['signal'] == 1]
    plt.scatter(buy_signals.index, buy_signals['price'],
                color='green', s=100, marker='^', label='Buy Signal')

    # Plot sell signals
    sell_signals = signals[signals['signal'] == -1]
    plt.scatter(sell_signals.index, sell_signals['price'],
                color='red', s=100, marker='v', label='Sell Signal')

    plt.title('XAU/USD Price with Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/xauusd_signals_{datetime.now().strftime('%Y%m%d')}.png")
    plt.close()


