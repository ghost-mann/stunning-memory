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
        logging.error(f"Failed to send email: {str{e}}")
        return False

def get_real_time_xauusd():
    # get the latest XAU/USD price data
    try:
        data = yf.download("GC=F", period="2d", interval="1m")
        return data
    except Exception as e:
        logging.error(f"Failed to fetch data: {str{e}}")
        return None