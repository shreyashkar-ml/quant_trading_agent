import pytz
import requests
import yfinance
import threading
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Tuple, Dict

from utils import load_pickle, save_pickle, Alpha
from alpha1 import Alpha1

def get_ndxt30_tickers():
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get("https://finance.yahoo.com/quote/%5ENDXT/components/", headers=headers)
        soup = BeautifulSoup(res.content, "lxml")
        table = soup.find_all('table')[0]
        df = pd.read_html(StringIO(str(table)))
        tickers = list(df[0].Symbol)
        return tickers
    except Exception as e:
        print(f"Failed to fetch NDXT tickers: {e}")
        return []

# Fetch historical data for a single ticker
def get_history(ticker: str, period_start: datetime, period_end, tries = 0, granularity="1d"):
    
    if tries >= 2:
        print(f"Max retries reached for {ticker}. Returning empty DataFrame.")
        return pd.DataFrame()
    
    try:
        ticker_obj = yfinance.Ticker(ticker)
        if not ticker_obj.info or ticker_obj.info.get('regularMarketPrice') is None:
            print(f"Skipping {ticker}: Not found or delisted in yfinance")
            return pd.DataFrame()

        # Fetch history if ticker is valid
        df = ticker_obj.history(
            start=period_start, 
            end=period_end,
            interval=granularity,
            auto_adjust=True
        ).reset_index()

        if df.empty:
            print(f"No data returned for {ticker}")
            return pd.DataFrame()

        df = df.rename(columns={
            "Date":"datetime",
            "Open":"open",
            "High":"high",
            "Low":"low",
            "Close":"close",
            "Volume":"volume"
        })
        df = df.drop(columns=["Dividends", "Stock Splits"])
        df = df.set_index("datetime",drop=True)
        print(f"Successfully fetched data for {ticker}: {len(df)} rows")
        return df
    
    except Exception as e:
        print(f"Error fetching {ticker} (try {tries+1}: {e})")
        return get_history(ticker, period_start, period_end, tries+1, granularity)

# historical data for multiple tickers using threading
def get_histories(tickers: List[str], period_starts: List[datetime], period_ends: List[datetime], granularity: str ="1d") -> Tuple[List[str], List[pd.DataFrame]]:
    dfs = [None]*len(tickers)
    lock = threading.Lock()

    def _helper(i: int) -> None:
        print(tickers[i])
        df = get_history(
            tickers[i], 
            period_starts[i], 
            period_ends[i], 
            granularity=granularity
        )
        with lock:
            dfs[i]  = df
    threads = [threading.Thread(target=_helper,args=(i,)) for i in range(len(tickers))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Filter out empty DataFrame and corresponding tickers
    valid_tickers = []
    valid_dfs = []
    for ticker, df in zip(tickers, dfs):
        if df is not None and not df.empty:
            valid_tickers.append(ticker)
            valid_dfs.append(df)

    return valid_tickers, valid_dfs

def get_ticker_dfs(start: datetime, end: datetime) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    data = load_pickle("dataset.obj")
    if data is not None:
        tickers, ticker_dfs = data
        print(f"Loaded {len(tickers)} tickers from pickle file")
        return tickers, ticker_dfs
    
    print("Fetching fresh data...")
    tickers = get_ndxt30_tickers()
    if not tickers:
        print("No tickers fetched. Exiting.")
        return [], {}
    
    starts = [start]*len(tickers)
    ends = [end]*len(tickers)
    tickers, dfs = get_histories(tickers, starts, ends, granularity="1d")
    ticker_dfs = {ticker: df for ticker, df in zip(tickers, dfs) if not df.empty}

    if ticker_dfs:
        save_pickle("dataset.obj", (tickers, ticker_dfs))
    else:
        print("No valid data to save.")

    return tickers, ticker_dfs


if __name__ == "__main__":
    period_start = datetime(2015, 1, 1, tzinfo=pytz.utc)
    period_end = datetime.now(pytz.utc)
    tickers, ticker_dfs = get_ticker_dfs(start=period_start, end=period_end)
    alpha1 = Alpha1(insts=tickers, dfs = ticker_dfs, start=period_start, end=period_end)
    portfolio_df1 = alpha1.run_simulation()
    print(portfolio_df1)