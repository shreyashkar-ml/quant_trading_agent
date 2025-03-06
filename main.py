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

if __name__ == "__main__":
    period_start = datetime(2015, 1, 1, tzinfo=pytz.utc)
    period_end = datetime.now(pytz.utc)
    tickers, ticker_dfs = get_ticker_dfs(start=period_start, end=period_end)
    alpha1 = Alpha1(insts=tickers, dfs = ticker_dfs, start=period_start, end=period_end)
    portfolio_df1 = alpha1.run_simulation()
    print(portfolio_df1)