import numpy as np
import pandas as pd

class Alpha:
    def __init__(self, insts, dfs, start, end, name):
        self.insts = insts
        self.dfs = dfs
        self.start = start
        self.end = end
        self.name = name
        for inst in self.insts:
            if "eligible" not in self.dfs[inst].columns:
                self.dfs[inst]["eligible"] = True

    def pre_compute(self, trade_range):
        pass

    def post_compute(self, trade_range):
        pass

class Alpha1(Alpha):
    def __init__(self, insts, dfs, start, end, name="alpha1"):
        super().__init__(insts, dfs, start, end, name)

    def pre_compute(self, trade_range):
        self.op4s = {}
        for inst in self.insts:
            inst_df = self.dfs[inst]
            op1 = inst_df.volume
            op2 = (inst_df.close - inst_df.low) - (inst_df.high - inst_df.close)
            op3 = inst_df.high - inst_df.low
            op4 = op1 * op2 / op3
            self.op4s[inst] = op4

    def post_compute(self, trade_range):
        temp = []
        for inst in self.insts:
            self.dfs[inst][self.name] = self.op4s[inst]
            temp.append(self.dfs[inst][self.name])

        temp_df = pd.concat(temp, axis=1)
        temp_df.columns = self.insts
        temp_df = temp_df.replace([np.inf, -np.inf], 0)
        zscore = lambda x: (x - np.mean(x)) / np.std(x) if np.std(x) > 0 else x * 0
        cszcre_df = temp_df.fillna(method="ffill").apply(zscore, axis=1)
        for inst in self.insts:
            self.dfs[inst][self.name] = cszcre_df[inst].rolling(12).mean() * -1
            self.dfs[inst]["eligible"] &= ~pd.isna(self.dfs[inst][self.name])

class Alpha2(Alpha):
    def __init__(self, insts, dfs, start, end, name="alpha2"):
        super().__init__(insts, dfs, start, end, name)

    def pre_compute(self, trade_range):
        self.alphas = {}
        for inst in self.insts:
            inst_df = self.dfs[inst]
            alpha = -1 * (1 - (inst_df.open / inst_df.close)).rolling(12).mean()
            self.alphas[inst] = alpha

    def post_compute(self, trade_range):
        for inst in self.insts:
            self.dfs[inst][self.name] = self.alphas[inst]
            self.dfs[inst][self.name] = self.dfs[inst][self.name].fillna(method="ffill")
            self.dfs[inst]["eligible"] &= ~pd.isna(self.dfs[inst][self.name])

class Alpha3(Alpha):
    def __init__(self, insts, dfs, start, end, name="alpha3"):
        super().__init__(insts, dfs, start, end, name)

    def pre_compute(self, trade_range):
        for inst in self.insts:
            inst_df = self.dfs[inst]
            # Compute moving averages
            ma10 = inst_df.close.rolling(10).mean()
            ma50 = inst_df.close.rolling(50).mean()
            ma20 = inst_df.close.rolling(20).mean()
            ma100 = inst_df.close.rolling(100).mean()
            ma200 = inst_df.close.rolling(200).mean()
            # Compute crossover signals as series
            fast = (ma10 > ma50).astype(int)
            medium = (ma20 > ma100).astype(int)
            slow = (ma50 > ma200).astype(int)
            # Sum the signals into alpha
            alpha = fast + medium + slow
            self.dfs[inst][self.name] = alpha

    def post_compute(self, trade_range):
        for inst in self.insts:
            self.dfs[inst][self.name] = self.dfs[inst][self.name].fillna(method="ffill")
            self.dfs[inst]["eligible"] &= ~pd.isna(self.dfs[inst][self.name])