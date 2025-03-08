import pandas as pd
import numpy as np

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
        self.trade_dates = None

        if not tickers or not dfs:
            print("Warning: No tickers or dataframes provided.")
            return

        # Use date-only trade range, timezone-naive
        trade_range = pd.date_range(self.start, self.end, freq='B').normalize().tz_localize(None)
        for alpha in self.alphas:
            alpha.pre_compute(trade_range)
            alpha.post_compute(trade_range)

    def run_backtest(self):
        if not self.tickers or not self.dfs:
            print("Cannot run backtest: No valid tickers or data.")
            return

        # Use the union of all dates, already normalized and timezone-naive from utils.py
        all_dates = pd.Index([])
        for df in self.dfs.values():
            all_dates = all_dates.union(df.index)
        # Convert to DatetimeIndex explicitly, should already be naive
        all_dates = pd.DatetimeIndex(all_dates)
        self.trade_dates = all_dates[(all_dates >= self.start) & (all_dates <= self.end)]
        
        print(f"Running backtest over {len(self.trade_dates)} dates.")
        print(f"Sample trade date: {self.trade_dates[0] if not self.trade_dates.empty else 'None'}")
        print(f"Trade dates dtype: {self.trade_dates.dtype}")

        for date in self.trade_dates:
            print(f"Processing date: {date}")
            available_tickers = [ticker for ticker in self.tickers if date in self.dfs[ticker].index]
            if not available_tickers:
                print(f"No tickers available for {date}, skipping.")
                continue
            equity = self.cash + sum(
                self.portfolio.get(ticker, 0) * self.dfs[ticker].loc[date, 'close']
                for ticker in available_tickers
            )
            self.equity.append(equity)
            signals = self.generate_signals(date)
            self.manage_portfolio(signals, date, equity)

    def generate_signals(self, date):
        alpha_values = {}
        for alpha in self.alphas:
            alpha_name = alpha.name
            values = {}
            for ticker in self.tickers:
                if date in self.dfs[ticker].index and self.dfs[ticker].loc[date, 'eligible']:
                    values[ticker] = self.dfs[ticker].loc[date, alpha_name]
            alpha_values[alpha_name] = values

        standardized = {}
        for alpha_name, values in alpha_values.items():
            if values:
                vals = list(values.values())
                mean = np.mean(vals)
                std = np.std(vals)
                if std > 0:
                    standardized[alpha_name] = {ticker: (v - mean) / std for ticker, v in values.items()}
                else:
                    standardized[alpha_name] = {ticker: 0 for ticker in values}

        composite_alpha = {}
        for ticker in self.tickers:
            if all(ticker in standardized[alpha_name] for alpha_name in standardized):
                composite_alpha[ticker] = sum(
                    standardized[alpha_name][ticker] for alpha_name in standardized
                )

        if composite_alpha:
            sorted_tickers = sorted(composite_alpha, key=composite_alpha.get)
            n = len(sorted_tickers)
            long_count = max(1, n // 4)
            short_count = max(1, n // 4)
            signals = {}
            for i, ticker in enumerate(sorted_tickers):
                if i < short_count:
                    signals[ticker] = -1
                elif i >= n - long_count:
                    signals[ticker] = 1
                else:
                    signals[ticker] = 0
            return signals
        return {}

    def manage_portfolio(self, signals, date, equity):
        long_tickers = [ticker for ticker, signal in signals.items() if signal == 1]
        short_tickers = [ticker for ticker, signal in signals.items() if signal == -1]
        n_long = len(long_tickers)
        n_short = len(short_tickers)

        w_long = 1 / n_long if n_long > 0 else 0
        w_short = -1 / n_short if n_short > 0 else 0

        for ticker in long_tickers:
            if date in self.dfs[ticker].index:
                price = self.dfs[ticker].loc[date, 'close']
                target_shares = (w_long * equity) / price
                current_shares = self.portfolio.get(ticker, 0)
                trade_shares = target_shares - current_shares
                cost = trade_shares * price
                self.cash -= cost
                self.portfolio[ticker] = current_shares + trade_shares

        for ticker in short_tickers:
            if date in self.dfs[ticker].index:
                price = self.dfs[ticker].loc[date, 'close']
                target_shares = (w_short * equity) / price
                current_shares = self.portfolio.get(ticker, 0)
                trade_shares = target_shares - current_shares
                cost = trade_shares * price
                self.cash -= cost
                self.portfolio[ticker] = current_shares + trade_shares

        for ticker in list(self.portfolio.keys()):
            if ticker not in long_tickers and ticker not in short_tickers and date in self.dfs[ticker].index:
                shares = self.portfolio[ticker]
                price = self.dfs[ticker].loc[date, 'close']
                self.cash += shares * price
                del self.portfolio[ticker]

    def get_pnl_stats(self):
        if not self.equity or len(self.equity) < 2:
            return {"error": "Insufficient equity data for stats"}

        equity_series = pd.Series(self.equity, index=self.trade_dates[:len(self.equity)])
        daily_returns = equity_series.pct_change().dropna()
        cumulative_returns = (equity_series / equity_series.iloc[0] - 1) * 100

        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        annualized_return = mean_daily_return * 252
        annualized_volatility = std_daily_return * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0

        rolling_max = equity_series.cummax()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100

        stats = {
            "Total Return (%)": cumulative_returns.iloc[-1],
            "Annualized Return (%)": annualized_return * 100,
            "Annualized Volatility (%)": annualized_volatility * 100,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown (%)": max_drawdown,
            "Daily Returns": daily_returns,
            "Cumulative Returns": cumulative_returns,
            "Equity Curve": equity_series
        }
        return stats