import pandas as pd
import yfinance
import threading
from datetime import datetime
from typing import List, Tuple, Dict
import pickle
import lzma
from bs4 import BeautifulSoup
import requests
from io import StringIO

def load_pickle(path: str) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    try:
        with lzma.open(path, "rb") as fp:
            return pickle.load(fp)
    except FileNotFoundError:
        print(f"Pickle file {path} not found. Fetching fresh data.")
        return None
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

def save_pickle(path: str, obj: Tuple[List[str], Dict[str, pd.DataFrame]]) -> None:
    try:
        with lzma.open(path, "wb") as fp:
            pickle.dump(obj, fp)
        print(f"Saved data to {path}")
    except Exception as e:
        print(f"Error saving pickle file: {e}")

def get_sp500_data(start, end):
    """Fetch S&P 500 data and compute 200-day moving average."""
    sp500 = yfinance.Ticker("^GSPC")
    df = sp500.history(start=start, end=end)
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    df = df[["Close"]].rename(columns={"Close": "close"})
    df["ma200"] = df["close"].rolling(200).mean()
    return df

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

def get_history(ticker: str, period_start: datetime, period_end, tries=0, granularity="1d"):
    if tries >= 2:
        print(f"Max retries reached for {ticker}. Returning empty DataFrame.")
        return pd.DataFrame()

    try:
        ticker_obj = yfinance.Ticker(ticker)
        if not ticker_obj.info or ticker_obj.info.get('regularMarketPrice') is None:
            print(f"Skipping {ticker}: Not found or delisted in yfinance")
            return pd.DataFrame()

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
            "Date": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        df = df.drop(columns=["Dividends", "Stock Splits"])
        # Normalize to date-only and remove timezone
        df['datetime'] = pd.to_datetime(df['datetime']).dt.normalize().dt.tz_localize(None)
        df = df.set_index("datetime", drop=True)
        df["eligible"] = True
        print(f"Successfully fetched data for {ticker}: {len(df)} rows")
        return df

    except Exception as e:
        print(f"Error fetching {ticker} (try {tries+1}: {e})")
        return get_history(ticker, period_start, period_end, tries+1, granularity)

def get_histories(tickers: List[str], period_starts: List[datetime], period_ends: List[datetime], granularity: str = "1d") -> Tuple[List[str], List[pd.DataFrame]]:
    dfs = [None] * len(tickers)
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
            dfs[i] = df

    threads = [threading.Thread(target=_helper, args=(i,)) for i in range(len(tickers))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

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
        for ticker in ticker_dfs:
            if "eligible" not in ticker_dfs[ticker].columns:
                ticker_dfs[ticker]["eligible"] = True
            # Ensure index is date-only and timezone-naive
            ticker_dfs[ticker].index = ticker_dfs[ticker].index.normalize().tz_localize(None)
        return tickers, ticker_dfs

    print("Fetching fresh data...")
    tickers = get_ndxt30_tickers()
    if not tickers:
        print("No tickers fetched. Exiting.")
        return [], {}

    starts = [start] * len(tickers)
    ends = [end] * len(tickers)
    tickers, dfs = get_histories(tickers, starts, ends, granularity="1d")
    ticker_dfs = {ticker: df for ticker, df in zip(tickers, dfs) if not df.empty}

    if ticker_dfs:
        save_pickle("dataset.obj", (tickers, ticker_dfs))
    else:
        print("No valid data to save.")

    return tickers, ticker_dfs