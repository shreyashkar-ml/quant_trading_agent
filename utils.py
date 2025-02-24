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
            input(self.dfs[inst])
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
                # compute pnl
                pass

            alpha_scores = {}
            for inst in eligibles:
                alpha_scores[inst] = random.uniform(0,1)
            print(alpha_scores)
            # compute alpha signals

            # compute positions and other informations
