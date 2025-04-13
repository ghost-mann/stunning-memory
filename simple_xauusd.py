import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Fetch data
print("Fetching XAU/USD data...")
data = yf.download("GC=F", period="1y", interval="1d")
print(f"Downloaded {len(data)} rows of XAU/USD data")

# Save data
data_file = f"data/xauusd_simple_{datetime.now().strftime('%Y%m%d')}.csv"
data.to_csv(data_file)
print(f"Data saved to {data_file}")

# Add some indicators
print("Calculating indicators...")
df = data.copy()
df['SMA20'] = df['Close'].rolling(window=20).mean()
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA12'] - df['EMA26']

# Create a simple plot
print("Creating plot...")
plt.figure(figsize=(12, 6))
plt.plot(df.index[-60:], df['Close'].iloc[-60:], label='Price', color='blue')
plt.plot(df.index[-60:], df['SMA20'].iloc[-60:], label='SMA20', color='orange')
plt.plot(df.index[-60:], df['SMA50'].iloc[-60:], label='SMA50', color='green')
plt.title('XAU/USD Price with Moving Averages')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"plots/xauusd_simple_{datetime.now().strftime('%Y%m%d')}.png")
print(f"Plot saved to plots/xauusd_simple_{datetime.now().strftime('%Y%m%d')}.png")

# Generate a simple signal
print("\nSimple signal:")
last_price = df['Close'].iloc[-1]
sma20 = df['SMA20'].iloc[-1]
sma50 = df['SMA50'].iloc[-1]

if sma20 > sma50:
    print(f"BUY signal: SMA20 ({sma20:.2f}) is above SMA50 ({sma50:.2f})")
else:
    print(f"SELL signal: SMA20 ({sma20:.2f}) is below SMA50 ({sma50:.2f})")

print(f"Current price: ${last_price:.2f}")