import pytz
import requests
import yfinance
import threading
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime
from utils import load_pickle, save_pickle
from typing import List, Tuple, Dict

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

def get_history(ticker, period_start, period_end, tries = 0, granularity="1d"):
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
    except Exception as err:
        if tries < 5:
            return get_history(ticker, period_start, period_end, tries + 1, granularity="1d")
        return pd.DataFrame()

    df = df.rename(columns={
        "Date":"datetime",
        "Open":"open",
        "High":"high",
        "Low":"low",
        "Close":"close",
        "Volume":"volume"
    })
    if df.empty:
        return pd.DataFrame()
    
    # df["datetime"] = df["datetime"].dt.tz_localize(pytz.utc)
    df = df.drop(columns=["Dividends", "Stock Splits"])
    df = df.set_index("datetime",drop=True)
    input(df)
    #index datetime open high low close volume

def get_histories(tickers, period_starts, period_ends, granularity="1d"):
    dfs = [None]*len(tickers)
    def _helper(i):
        print(tickers[i])
        df = get_history(
            tickers[i], 
            period_starts[i], 
            period_ends[i], 
            granularity=granularity
        )
        dfs[i]  = df
    threads = [threading.Thread(target=_helper,args=(i,)) for i in range(len(tickers))]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    tickers = [tickers[i] for i in range(len(tickers)) if not dfs[i].empty]
    dfs = [df for df in dfs if not df.empty]
    return tickers, dfs

def get_ticker_dfs(start, end):
    try:
        tickers, ticker_dfs = load_pickle("dataset.obj")
    except Exception as err:
        tickers = get_ndxt30_tickers()
        starts = [start]*len(tickers)
        ends = [end]*len(tickers)
        tickers, dfs = get_histories(tickers, starts, ends, granularity="1d")
        ticker_dfs = {ticker:df for ticker,df in zip(tickers,dfs)}
        save_pickle("dataset.obj", (tickers, ticker_dfs))
    return tickers, ticker_dfs


period_start = datetime(2015, 1, 1, tzinfo=pytz.utc)
period_end = datetime.now(pytz.utc)
tickers, ticker_dfs = get_ticker_dfs(start=period_start,end=period_end)
print(ticker_dfs)