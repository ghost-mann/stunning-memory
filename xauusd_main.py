import os
import argparse
import pandas as pd
from datetime import datetime
import subprocess
import sys
import logging

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'logs/xauusd_main_{datetime.now().strftime("%Y%m%d")}.log',
    filemode='a'
)


def check_requirements():
    """Check if all required libraries are installed"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'yfinance', 'scikit-learn'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")

        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}")
                return False

    return True


def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'signals', 'plots', 'logs', 'backtest']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("Directory structure set up successfully.")


def run_data_collection(period="1y", interval="1d"):
    """Run the data collection script"""
    try:
        # Change this part
        # from xauusd_data_collection import fetch_xauusd_data, save_data

        # Instead, use direct import
        import xauusd_data_collection

        print(f"Fetching XAU/USD data (period: {period}, interval: {interval})...")
        data = xauusd_data_collection.fetch_xauusd_data(period=period, interval=interval)

        if data is not None and not data.empty:
            filename = f"xauusd_data_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = xauusd_data_collection.save_data(data, filename=filename)
            print(f"Data collection complete. Saved to {filepath}")
            return filepath
        else:
            print("Data collection failed: No data returned")
            return None
    except Exception as e:
        print(f"Error in data collection: {str(e)}")
        logging.error(f"Data collection error: {str(e)}")
        return None

def run_signal_analysis(data_filepath):
    """Run the signal analysis script"""
    try:
        from xauusd_signal_generation import (
            load_data, add_technical_indicators,
            generate_signals, plot_signals, get_latest_signals
        )

        print(f"Loading data from {data_filepath}...")
        data = load_data(data_filepath)

        print("Adding technical indicators...")
        data_with_indicators = add_technical_indicators(data)

        print("Generating trading signals...")
        signals = generate_signals(data_with_indicators)

        print("Creating signal plot...")
        plot_signals(data_with_indicators, signals)

        # Display and return latest signals
        latest_signals = get_latest_signals(signals)
        print("\nLatest XAU/USD Trading Signals:")
        print(latest_signals)

        # Save signals to CSV
        signals_file = f"signals/xauusd_signals_{datetime.now().strftime('%Y%m%d')}.csv"
        signals.to_csv(signals_file)
        print(f"\nDetailed signals saved to {signals_file}")

        return latest_signals
    except Exception as e:
        print(f"Error in signal analysis: {str(e)}")
        logging.error(f"Signal analysis error: {str(e)}")
        return None


def run_monitoring(interval_minutes=5):
    """Run the real-time monitoring script"""
    try:
        from xauusd_monitoring import monitor_xauusd, setup_email_config

        # Set up email config (optional)
        # Comment this out if you don't want email notifications
        email_config = setup_email_config()

        # Start monitoring
        print(f"Starting real-time monitoring (interval: {interval_minutes} minutes)...")
        monitor_xauusd(interval_minutes=interval_minutes, email_config=email_config)
    except Exception as e:
        print(f"Error in monitoring: {str(e)}")
        logging.error(f"Monitoring error: {str(e)}")


def run_backtest(data_filepath):
    """Run backtest on historical data"""
    try:
        # Import needed functions from the signal generation script
        from xauusd_signal_generation import (
            load_data, add_technical_indicators, generate_signals
        )

        print("Running backtest on historical data...")

        # Load and process data
        data = load_data(data_filepath)
        data_with_indicators = add_technical_indicators(data)
        signals = generate_signals(data_with_indicators)

        # Initialize backtest parameters
        initial_capital = 10000
        position = 0
        capital = initial_capital
        trades = []

        # Run through the signals
        for i in range(1, len(signals)):
            yesterday = signals.iloc[i - 1]
            today = signals.iloc[i]
            price = today['price']

            # Buy signal (if not already in position)
            if yesterday['signal'] == 1 and position == 0:
                # Calculate how much gold we can buy (simplified)
                position = capital / price
                capital = 0
                trades.append({
                    'date': today.name,
                    'action': 'BUY',
                    'price': price,
                    'position': position,
                    'capital': capital
                })

            # Sell signal (if in position)
            elif yesterday['signal'] == -1 and position > 0:
                # Sell all gold
                capital = position * price
                position = 0
                trades.append({
                    'date': today.name,
                    'action': 'SELL',
                    'price': price,
                    'position': position,
                    'capital': capital
                })

        # Close out any remaining position at the last price
        if position > 0:
            capital = position * signals.iloc[-1]['price']

        # Calculate results
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            total_trades = len(trades_df)
            profit = capital - initial_capital
            profit_percent = (profit / initial_capital) * 100

            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']

            if not sell_trades.empty and not buy_trades.empty:
                win_count = sum(sell_trades['capital'].reset_index(drop=True) >
                                buy_trades['price'].reset_index(drop=True) *
                                buy_trades['position'].reset_index(drop=True))

                win_rate = (win_count / len(sell_trades)) * 100 if len(sell_trades) > 0 else 0
            else:
                win_rate = 0

            # Print results
            print("\nBacktest Results:")
            print(f"Initial Capital: ${initial_capital:.2f}")
            print(f"Final Capital: ${capital:.2f}")
            print(f"Profit/Loss: ${profit:.2f} ({profit_percent:.2f}%)")
            print(f"Total Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.2f}%")

            # Save backtest results
            os.makedirs("backtest", exist_ok=True)
            backtest_file = f"backtest/xauusd_backtest_{datetime.now().strftime('%Y%m%d')}.csv"
            trades_df.to_csv(backtest_file)
            print(f"\nBacktest details saved to {backtest_file}")

            # Create and save summary
            summary = {
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'initial_capital': initial_capital,
                'final_capital': capital,
                'profit': profit,
                'profit_percent': profit_percent,
                'total_trades': total_trades,
                'win_rate': win_rate
            }

            summary_df = pd.DataFrame([summary])
            summary_file = f"backtest/xauusd_backtest_summary_{datetime.now().strftime('%Y%m%d')}.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"Backtest summary saved to {summary_file}")

            return summary
        else:
            print("No trades were executed during the backtest period.")
            return None
    except Exception as e:
        print(f"Error in backtest: {str(e)}")
        logging.error(f"Backtest error: {str(e)}")
        return None


def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description='XAU/USD Forex Trading Signal System')

    parser.add_argument('--action', type=str, required=True,
                        choices=['setup', 'collect', 'analyze', 'monitor', 'backtest', 'all'],
                        help='Action to perform')

    parser.add_argument('--period', type=str, default='1y',
                        help='Data period for collection (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)')

    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (e.g., 1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk, 1mo)')

    parser.add_argument('--monitor-interval', type=int, default=5,
                        help='Interval for real-time monitoring in minutes')

    args = parser.parse_args()

    # Check requirements first
    if not check_requirements():
        print("Failed to install required packages. Exiting.")
        return

    # Process the requested action
    if args.action == 'setup' or args.action == 'all':
        setup_directories()

    data_filepath = None

    if args.action == 'collect' or args.action == 'all':
        data_filepath = run_data_collection(period=args.period, interval=args.interval)

    if args.action == 'analyze' or args.action == 'all':
        if data_filepath is None:
            # Try to find the most recent data file
            data_dir = "data"
            if os.path.exists(data_dir):
                data_files = [f for f in os.listdir(data_dir) if f.startswith("xauusd_data_")]
                if data_files:
                    data_filepath = os.path.join(data_dir, sorted(data_files)[-1])
                    print(f"Using most recent data file: {data_filepath}")

        if data_filepath:
            run_signal_analysis(data_filepath)
        else:
            print("No data file available for analysis. Run data collection first.")

    if args.action == 'backtest' or args.action == 'all':
        if data_filepath is None:
            # Try to find the most recent data file
            data_dir = "data"
            if os.path.exists(data_dir):
                data_files = [f for f in os.listdir(data_dir) if f.startswith("xauusd_data_")]
                if data_files:
                    data_filepath = os.path.join(data_dir, sorted(data_files)[-1])
                    print(f"Using most recent data file: {data_filepath}")

        if data_filepath:
            run_backtest(data_filepath)
        else:
            print("No data file available for backtesting. Run data collection first.")

    if args.action == 'monitor' or args.action == 'all':
        run_monitoring(interval_minutes=args.monitor_interval)


if __name__ == "__main__":
    main()