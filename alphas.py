import pandas as pd
import numpy as np

class Alpha:
    def __init__(self, insts, dfs, start, end, name):
        self.insts = insts
        self.dfs = dfs
        self.start = start
        self.end = end
        self.name = name

    def pre_compute(self, trade_range):
        pass

    def post_compute(self, trade_range):
        pass

class Alpha1(Alpha):
    def __init__(self, insts, dfs, start, end, name="alpha1"):
        super().__init__(insts, dfs, start, end, name)

    def pre_compute(self, trade_range):
        for inst in self.insts:
            df = self.dfs[inst]
            op1 = df['volume']
            op2 = (df['close'] - df['low']) - (df['high'] - df['close'])
            op3 = df['high'] - df['low']
            op4 = op1 * op2 / op3.replace(0, np.nan)
            df['op4'] = op4

    def post_compute(self, trade_range):
        temp_df = pd.DataFrame(index=trade_range)
        for inst in self.insts:
            temp_df[inst] = self.dfs[inst]['op4']
        temp_df = temp_df.replace(np.inf, 0).replace(-np.inf, 0)
        zscore = lambda x: (x - np.mean(x)) / np.std(x) if np.std(x) > 0 else x * 0
        cszcre_df = temp_df.fillna(method="ffill").apply(zscore, axis=1)
        for inst in self.insts:
            self.dfs[inst][self.name] = cszcre_df[inst].rolling(12).mean() * -1
            self.dfs[inst][self.name] = self.dfs[inst][self.name].fillna(0)  # Ensure no NaN

class Alpha2(Alpha):
    def __init__(self, insts, dfs, start, end, name="alpha2"):
        super().__init__(insts, dfs, start, end, name)

    def post_compute(self, trade_range):
        for inst in self.insts:
            df = self.dfs[inst]
            alpha = -1 * (1 - (df['open'] / df['close'])).rolling(12).mean()
            self.dfs[inst][self.name] = alpha.fillna(0)  # Ensure no NaN

class Alpha3(Alpha):
    def __init__(self, insts, dfs, start, end, name="alpha3"):
        super().__init__(insts, dfs, start, end, name)

    def post_compute(self, trade_range):
        for inst in self.insts:
            df = self.dfs[inst]
            fast = (df['close'].rolling(10).mean() > df['close'].rolling(50).mean()).astype(int)
            medium = (df['close'].rolling(20).mean() > df['close'].rolling(100).mean()).astype(int)
            slow = (df['close'].rolling(50).mean() > df['close'].rolling(200).mean()).astype(int)
            self.dfs[inst][self.name] = (fast + medium + slow).fillna(0)  # Ensure no NaN

class RegimeSwitchingAlpha(Alpha):
    def __init__(self, insts, dfs, start, end, sp500_df, name="regime_switching"):
        super().__init__(insts, dfs, start, end, name)
        self.sp500_df = sp500_df.reindex(dfs[list(dfs.keys())[0]].index, method="ffill")  # Align with ticker data
        self.alpha1 = Alpha1(insts, dfs, start, end)
        self.alpha2 = Alpha2(insts, dfs, start, end)
        self.alpha3 = Alpha3(insts, dfs, start, end)

    def pre_compute(self, trade_range):
        self.alpha1.pre_compute(trade_range)
        self.alpha2.pre_compute(trade_range)
        self.alpha3.pre_compute(trade_range)

    def post_compute(self, trade_range):
        self.alpha1.post_compute(trade_range)
        self.alpha2.post_compute(trade_range)
        self.alpha3.post_compute(trade_range)
        for date in trade_range:
            for inst in self.insts:
                if date not in self.dfs[inst].index:
                    continue  # Skip if no data for this ticker on this date
                if date in self.sp500_df.index and not pd.isna(self.sp500_df.loc[date, "ma200"]):
                    if self.sp500_df.loc[date, "close"] > self.sp500_df.loc[date, "ma200"]:
                        val = self.dfs[inst].loc[date, self.alpha3.name]
                        self.dfs[inst].loc[date, self.name] = val if not pd.isna(val) else 0
                    else:
                        a1 = self.dfs[inst].loc[date, self.alpha1.name]
                        a2 = self.dfs[inst].loc[date, self.alpha2.name]
                        avg = (a1 + a2) / 2 if not (pd.isna(a1) or pd.isna(a2)) else 0
                        self.dfs[inst].loc[date, self.name] = avg
                else:
                    self.dfs[inst].loc[date, self.name] = 0  # No S&P 500 data or early period