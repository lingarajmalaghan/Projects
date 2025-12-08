# stock_analysis.py
"""
Stock Analysis - simple, clean, reproducible.
Change TICKER / PERIOD / START / END below as needed.
"""

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


TICKER = "TCS.NS"            
USE_PERIOD = True          
PERIOD = "4mo"             
START_DATE = "2025-06-01"  
END_DATE = "2024-10-01"    
MA_SHORT = 10              
MA_LONG = 50               
VOL_WINDOW = 30            
OUT_FOLDER = "outputs"     
SHOW_PLOTS = True          
SAVE_PLOTS = True
# -----------------------------------------------------------------

def fetch_data(ticker):
    """Fetch data using yfinance. Returns DataFrame with Date as column."""
    print(f"Fetching data for {ticker} ...")
    if USE_PERIOD:
        df = yf.download(ticker, period=PERIOD, progress=False)
    else:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

    if df is None or df.empty:
        raise ValueError("No data returned. Check ticker, period/dates, or internet connection.")
    df = df.reset_index()  
    return df

def preprocess(df):
    """Ensure Date column, sort, use Adj Close, drop NaNs if needed."""
    df = df.copy()
   
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    
    if 'Adj Close' in df.columns:
        df['Price'] = df['Adj Close']
    else:
        df['Price'] = df['Close']

    return df

def compute_indicators(df):
    """Add returns, moving averages, volatility, cumulative returns, drawdown, signals."""
    df = df.copy()
   
    df['Daily Return'] = df['Price'].pct_change()

   
    df[f'MA{MA_SHORT}'] = df['Price'].rolling(window=MA_SHORT).mean()
    df[f'MA{MA_LONG}'] = df['Price'].rolling(window=MA_LONG).mean()

   
    df[f'Vol_{VOL_WINDOW}d'] = df['Daily Return'].rolling(window=VOL_WINDOW).std()
    df[f'AnnVol_{VOL_WINDOW}d'] = df[f'Vol_{VOL_WINDOW}d'] * np.sqrt(252)  

   
    df['Cumulative Return'] = (1 + df['Daily Return'].fillna(0)).cumprod() - 1

    
    df['Rolling Max'] = df['Price'].cummax()
    df['Drawdown'] = (df['Price'] - df['Rolling Max']) / df['Rolling Max']

    
    df['Signal'] = np.where(df[f'MA{MA_SHORT}'] > df[f'MA{MA_LONG}'], 1, 0)
    df['Signal'] = df['Signal'].fillna(0).astype(int)
    df['Position'] = df['Signal'].diff()  

    return df

def get_summary(df):
    """Produce readable summary stats (strings)."""
    s = {}
   
    dr = df['Daily Return'].dropna()
    s['Average Daily Return'] = dr.mean()
    s['Std Daily Return'] = dr.std()
    s['Annualized Return (approx)'] = (1 + dr.mean())**252 - 1 
    s['Annualized Volatility (std*sqrt(252))'] = dr.std() * np.sqrt(252)
    
    s['Total Return'] = df['Cumulative Return'].iloc[-1]
    
    s['Latest 30d Vol'] = df[f'Vol_{VOL_WINDOW}d'].iloc[-1]
    s['Latest Annualized 30d Vol'] = df[f'AnnVol_{VOL_WINDOW}d'].iloc[-1]
    
    s['Latest Price'] = df['Price'].iloc[-1]
    s['MA Signal (short>long)'] = bool(df['Signal'].iloc[-1])
    
    last_buy = df.loc[df['Position'] == 1, 'Date']
    last_sell = df.loc[df['Position'] == -1, 'Date']
    s['Last Buy Signal'] = last_buy.iloc[-1].strftime('%Y-%m-%d') if not last_buy.empty else "None"
    s['Last Sell Signal'] = last_sell.iloc[-1].strftime('%Y-%m-%d') if not last_sell.empty else "None"

    return s

def save_outputs(df, out_folder=OUT_FOLDER, ticker=TICKER):
    os.makedirs(out_folder, exist_ok=True)
    
    full_path = os.path.join(out_folder, f"{ticker}_full.csv")
    df.to_csv(full_path, index=False)
   
    last_30 = df.tail(30)
    last_path = os.path.join(out_folder, f"{ticker}_last30.csv")
    last_30.to_csv(last_path, index=False)
    print(f"Saved full data to {full_path}")
    print(f"Saved last 30 trading days to {last_path}")
    return full_path, last_path

def plot_charts(df, out_folder=OUT_FOLDER, ticker=TICKER):
    os.makedirs(out_folder, exist_ok=True)

   
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Price'], label='Price')
    plt.plot(df['Date'], df[f'MA{MA_SHORT}'], label=f'MA{MA_SHORT}')
    plt.plot(df['Date'], df[f'MA{MA_LONG}'], label=f'MA{MA_LONG}')
    plt.title(f"{ticker} Price and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    price_png = os.path.join(out_folder, f"{ticker}_price_ma.png")
    if SAVE_PLOTS: plt.savefig(price_png, bbox_inches='tight')
    if SHOW_PLOTS: plt.show()
    plt.close()

    
    plt.figure(figsize=(8,5))
    df['Daily Return'].hist(bins=50)
    plt.title(f"{ticker} Daily Returns Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    ret_png = os.path.join(out_folder, f"{ticker}_returns_hist.png")
    if SAVE_PLOTS: plt.savefig(ret_png, bbox_inches='tight')
    if SHOW_PLOTS: plt.show()
    plt.close()

    
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df['Cumulative Return'])
    plt.title(f"{ticker} Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    cum_png = os.path.join(out_folder, f"{ticker}_cumulative.png")
    if SAVE_PLOTS: plt.savefig(cum_png, bbox_inches='tight')
    if SHOW_PLOTS: plt.show()
    plt.close()

    print(f"Plots saved to {out_folder}")

def main():
    try:
        df_raw = fetch_data(TICKER)
    except Exception as e:
        print("Error fetching data:", e)
        sys.exit(1)

    df = preprocess(df_raw)
    df = compute_indicators(df)

    
    last_30 = df.tail(30)

   
    summary = get_summary(df)
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    
    save_outputs(df)

    
    os.makedirs(OUT_FOLDER, exist_ok=True)
    with open(os.path.join(OUT_FOLDER, f"{TICKER}_summary.txt"), 'w') as f:
        f.write("Summary for " + TICKER + "\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print("Text summary saved.")

    
    if SHOW_PLOTS or SAVE_PLOTS:
        plot_charts(df)

if __name__ == "__main__":
    main()
