import yfinance as yf
import pandas as pd
from datetime import timedelta, datetime
import os

def fetch_xauusd_data(period="1y", interval="1d"):
    """
    Fetch XAU/USD data using Yahoo Finance.

    Parameters:
    - period: Time period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    - interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
    - DataFrame with OHLC data
    """
    # GLD is a gold ETF used as a proxy for XAU/USD
    data = yf.download("GC=F", period=period, interval=interval)

    # Alternatively, you can specify start and end dates
    # start_date = datetime.now() - timedelta(days=365)
    # end_date = datetime.now()
    # data = yf.download("GC=F", start=start_date, end=end_date, interval=interval)

    # Clean the data
    data = data.dropna()

    return data

def save_data(data, filename="xauusd_data.csv"):
    """Save the data to a CSV file"""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    data.to_csv(filepath)
    print(f"Data saved to {filepath}")
    return filepath

if __name__ == "__main__":
    # fetch daily data for the past year
    xauusd_data = fetch_xauusd_data(period="1y", interval="1d")

    # print data summary
    print(f"Downloaded {len(xauusd_data)} rows of XAU/USD data")
    print(xauusd_data.head())
    # save to csv
    save_data(xauusd_data, filename=f"xauusd_data_{datetime.now().strftime('%Y%m%d')}.csv")