import yfinance as yf
import pandas as pd
import numpy as np
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import logging
from datetime import datetime

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'logs/xauusd_monitor_{datetime.now().strftime("%Y%m%d")}.log',
    filemode='a'
)


def setup_email_config():
    """
    Set up email configuration for notifications
    Note: For security, use environment variables or a config file
    """
    # Change these to your email settings
    config = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'smtp_username': os.environ.get('EMAIL_USERNAME', ''),
        'smtp_password': os.environ.get('EMAIL_PASSWORD', ''),  # App password for Gmail
        'sender_email': os.environ.get('SENDER_EMAIL', ''),
        'recipient_email': os.environ.get('RECIPIENT_EMAIL', '')
    }
    return config


def send_email_notification(config, subject, message):
    # send email notification with the signal

    try:
        # create message
        msg = MIMEMultipart()
        msg['From'] = config['send_email']
        msg['To'] = config['recipient_email']
        msg['Subject'] = subject

        # add message body
        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['smtp_username'], config['smtp_password'])
        server.send_message(msg)
        server.quit()

        logging.info(f"Email notification sent: {subject}")
        return True
    except Exception as e:
        logging.error(f"Failed to send email: {str{e} }")
        return False


def get_real_time_xauusd():
    # get the latest XAU/USD price data
    try:
        data = yf.download("GC=F", period="2d", interval="1m")
        return data
    except Exception as e:
        logging.error(f"Failed to fetch data: {str{e} }")
        return None


def calculate_indicators(data):
    """Calculate technical indicators for the latest data"""
    if data is None or len(data) < 50:
        logging.warning("Not enough data to calculate indicators")
        return None

    df = data.copy()

    # Simple Moving Averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

    return df.dropna()


def check_for_signals(df):
    """Check for trading signals in real-time data"""
    if df is None or df.empty:
        return None

    # Get the latest data point
    latest = df.iloc[-1]

    signals = []
    signal_strength = 0

    # Check Moving Average Crossover
    if df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1] and df['SMA20'].iloc[-2] <= df['SMA50'].iloc[-2]:
        signals.append("SMA20 crossed above SMA50 (BULLISH)")
        signal_strength += 1
    elif df['SMA20'].iloc[-1] < df['SMA50'].iloc[-1] and df['SMA20'].iloc[-2] >= df['SMA50'].iloc[-2]:
        signals.append("SMA20 crossed below SMA50 (BEARISH)")
        signal_strength -= 1

    # Check MACD
    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]:
        signals.append("MACD crossed above signal line (BULLISH)")
        signal_strength += 1
    elif df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]:
        signals.append("MACD crossed below signal line (BEARISH)")
        signal_strength -= 1

    # Check RSI
    if latest['RSI'] < 30:
        signals.append(f"RSI is oversold at {latest['RSI']:.2f} (BULLISH)")
        signal_strength += 1
    elif latest['RSI'] > 70:
        signals.append(f"RSI is overbought at {latest['RSI']:.2f} (BEARISH)")
        signal_strength -= 1

    # Check Bollinger Bands
    if latest['Close'] < latest['BB_Lower']:
        signals.append("Price below lower Bollinger Band (BULLISH)")
        signal_strength += 1
    elif latest['Close'] > latest['BB_Upper']:
        signals.append("Price above upper Bollinger Band (BEARISH)")
        signal_strength -= 1

    # Determine overall signal
    if signal_strength > 1:
        overall_signal = "STRONG BUY"
    elif signal_strength == 1:
        overall_signal = "BUY"
    elif signal_strength == 0:
        overall_signal = "NEUTRAL"
    elif signal_strength == -1:
        overall_signal = "SELL"
    else:  # signal_strength < -1
        overall_signal = "STRONG SELL"

    if signals:
        return {
            'timestamp': df.index[-1],
            'price': latest['Close'],
            'signals': signals,
            'overall_signal': overall_signal,
            'signal_strength': signal_strength
        }
    else:
        return None


def format_signal_message(signal_data):
    """Format signal data into a readable message"""
    message = f"""
XAU/USD TRADING SIGNAL ALERT
---------------------------
Time: {signal_data['timestamp']}
Current Price: ${signal_data['price']:.2f}

OVERALL SIGNAL: {signal_data['overall_signal']}
Signal Strength: {abs(signal_data['signal_strength'])}

Technical Indicators:
"""

    for signal in signal_data['signals']:
        message += f"- {signal}\n"

    message += f"\nThis is an automated alert. Please conduct your own analysis before trading."

    return message


def monitor_xauusd(interval_minutes=5, email_config=None):
    """
    Monitor XAU/USD in real-time and send notifications when signals occur

    Parameters:
    - interval_minutes: How often to check for signals (in minutes)
    - email_config: Email configuration for notifications
    """
    # Create log directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    print(f"Starting XAU/USD monitoring (checking every {interval_minutes} minutes)...")
    logging.info(f"XAU/USD monitoring started. Interval: {interval_minutes} minutes")

    last_signal_time = None

    try:
        while True:
            current_time = datetime.now()
            print(f"Checking for signals at {current_time.strftime('%Y-%m-%d %H:%M:%S')}...")

            # Get latest data
            data = get_real_time_xauusd()

            if data is not None and not data.empty:
                # Calculate indicators
                df_with_indicators = calculate_indicators(data)

                # Check for signals
                signal_data = check_for_signals(df_with_indicators)

                if signal_data:
                    signal_time = signal_data['timestamp']

                    # Check if this is a new signal (avoid duplicates)
                    if last_signal_time is None or signal_time > last_signal_time:
                        print(f"Signal detected: {signal_data['overall_signal']}")
                        logging.info(f"Signal detected: {signal_data['overall_signal']}")

                        # Format message
                        message = format_signal_message(signal_data)
                        print("\n" + message + "\n")

                        # Send email notification if configured
                        if email_config:
                            subject = f"XAU/USD {signal_data['overall_signal']} Signal Alert"
                            send_email_notification(email_config, subject, message)

                        # Update last signal time
                        last_signal_time = signal_time
                else:
                    print("No new signals detected.")
            else:
                print("Failed to get data or not enough data points.")

            # Wait for the next check
            print(f"Next check in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        logging.info("XAU/USD monitoring stopped by user.")
    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error(f"Monitoring error: {str(e)}")


if __name__ == "__main__":
    # Set up email config (optional)
    # Comment this out if you don't want email notifications
    email_config = setup_email_config()

    # Start monitoring (check every 5 minutes)
    monitor_xauusd(interval_minutes=5, email_config=email_config)
