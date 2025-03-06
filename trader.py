import numpy as np
import pandas as pd

class Trader:
    def __init__(self, tickers, dfs, start, end, alphas):
        self.tickers = tickers
        self.dfs = dfs
        self.start = start
        self.end = end
        self.alphas = alphas
        self.portfolio = {}
        self.cash = 100000
        self.equity = []

        trade_range = pd.date_range(self.start, self.end, freq='B')
        for alpha in self.alphas:
            alpha.pre_compute(trade_range)
            alpha.post_compute(trade_range)