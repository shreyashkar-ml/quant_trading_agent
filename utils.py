import lzma
import random
import numpy as np
import pandas as pd
import dill as pickle
from typing import List, Tuple, Dict

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

def get_pnl_stats(date,prev,portfolio_df,insts,idx,dfs):
    day_pnl = 0
    nominal_ret = 0
    for inst in insts:
        units = portfolio_df.loc[idx-1, "{} units".format(inst)]
        if units != 0:
            delta = dfs[inst].loc[date, "close"] - dfs[inst].loc[prev, "close"]
            inst_pnl = delta * units
            day_pnl += inst_pnl
            nominal_ret += portfolio_df.loc[idx-1, "{} w".format(inst)] * dfs[inst].loc[date, "ret"]
    capital_ret = nominal_ret * portfolio_df.loc[idx-1, "leverage"]
    portfolio_df.loc[idx,"capital"] = portfolio_df.loc[idx - 1, "capital"] + day_pnl
    portfolio_df.loc[idx, "day_pnl"] = day_pnl
    portfolio_df.loc[idx, "nominal_ret"] = nominal_ret
    portfolio_df.loc[idx, "capital_ret"] = capital_ret
    return day_pnl, capital_ret

class Alpha():

    def __init__(self, insts, dfs, start, end):
        self.insts = insts
        self.dfs = dfs
        self.start = start
        self.end = end

    def init_portfolio_settings(self, trade_range):
        portfolio_df = pd.DataFrame(index=trade_range) \
                        .reset_index() \
                        .rename(columns={"index":"datetime"})
        portfolio_df.loc[0,"capital"] = 10000
        return portfolio_df
    
    def compute_meta_info(self, trade_range):
        # Normalize trade_range to date-only UTC
        trade_range = pd.DatetimeIndex([d.date() for d in trade_range]).tz_localize("UTC")
        for inst in self.insts:
            # Normalize the existing DataFrame index to date-only UTC
            self.dfs[inst].index = pd.DatetimeIndex([d.date() for d in self.dfs[inst].index]).tz_localize("UTC")
            df = pd.DataFrame(index=trade_range)
            self.dfs[inst] = df.join(self.dfs[inst]).fillna(method="ffill").fillna(method="bfill")
            self.dfs[inst]["ret"] = -1 * self.dfs[inst]["close"]/self.dfs[inst]["close"].shift(1)
            sampled = self.dfs[inst]["close"] != self.dfs[inst]["close"].shift(1).fillna(method="bfill")
            eligible = sampled.rolling(5).apply(lambda x: int(np.any(x))).fillna(0)
            self.dfs[inst]["eligible"] = eligible.astype(int) & (self.dfs[inst]["close"] > 0).astype(int)
        return

    def run_simulation(self):
        print("running backtest")
        date_range = pd.date_range(start=self.start, end=self.end, freq="D")
        self.compute_meta_info(trade_range=date_range)
        portfolio_df = self.init_portfolio_settings(trade_range=date_range)
        for i in portfolio_df.index:
            date = portfolio_df.loc[i, "datetime"]

            eligibles = [inst for inst in self.insts if self.dfs[inst].loc[date, "eligible"]]
            non_eligibles = [inst for inst in self.insts if inst not in eligibles]

            if i != 0:
                date_prev = portfolio_df.loc[i-1,"datetime"]
                day_pnl, capital_ret = get_pnl_stats(
                    date=date,
                    prev=date_prev,
                    portfolio_df=portfolio_df,
                    insts=self.insts,
                    idx=i,
                    dfs=self.dfs
                )

            alpha_scores = {}
            for inst in eligibles:
                alpha_scores[inst] = random.uniform(0,1)
            alpha_scores = {k:v for k,v in sorted(alpha_scores.items(), key = lambda pair: pair[1])}
            alpha_long = list(alpha_scores.keys())[-int(len(eligibles)/4):]
            alpha_short = list(alpha_scores.keys())[:int(len(eligibles)/4)]

            for inst in non_eligibles:
                portfolio_df.loc[i, "{} w".format(inst)] = 0
                portfolio_df.loc[i, "{} units".format(inst)] = 0

            nominal_tot = 0
            for inst in eligibles:
                forecast = 1 if inst in alpha_long else (-1 if inst in alpha_short else 0)
                dollar_allocation = portfolio_df.loc[i, "capital"] / (len(alpha_long) + len(alpha_short))
                position = forecast * dollar_allocation / self.dfs[inst].loc[date, "close"]
                portfolio_df.loc[i, inst + " units"] = position
                nominal_tot += abs(position * self.dfs[inst].loc[date, "close"])

            for inst in eligibles:
                units = portfolio_df.loc[i, inst + " units"]
                nominal_inst = units * self.dfs[inst].loc[date, "close"]
                inst_w = nominal_inst / nominal_tot
                portfolio_df.loc[i, inst + " w"] = inst_w

            portfolio_df.loc[i, "nominal"] = nominal_tot
            portfolio_df.loc[i, "leverage"] = nominal_tot / portfolio_df.loc[i, "capital"]
            if i%100 == 0: print(portfolio_df.loc[i])
            return portfolio_df