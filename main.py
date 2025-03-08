import pandas as pd
from datetime import datetime
from utils import get_ticker_dfs, get_sp500_data
from alphas import Alpha1, Alpha2, Alpha3, RegimeSwitchingAlpha
from trader import Trader

start = pd.Timestamp(datetime(2015, 1, 1)).normalize()
end = pd.Timestamp(datetime(2023, 12, 31)).normalize()

sp500_df = get_sp500_data(start, end)
tickers, dfs = get_ticker_dfs(start, end)

if not tickers or not dfs:
    print("No data available to proceed with backtest.")
else:
    alpha1 = Alpha1(tickers, dfs, start, end, name="alpha1")
    alpha2 = Alpha2(tickers, dfs, start, end, name="alpha2")
    alpha3 = Alpha3(tickers, dfs, start, end, name="alpha3")
    regime_alpha = RegimeSwitchingAlpha(tickers, dfs, start, end, sp500_df, name="regime_switching")

    strategies = {
        "Alpha1": [alpha1],
        "Alpha2": [alpha2],
        "Alpha3": [alpha3],
        "Combined Alpha": [alpha1, alpha2, alpha3],
        "Regime Switching Alpha": [regime_alpha]
    }

    results = {}
    for strategy_name, alphas in strategies.items():
        print(f"\n=== Running {strategy_name} Backtest ===")
        trader = Trader(tickers, dfs, start, end, alphas)
        trader.run_backtest()
        if trader.equity:
            results[strategy_name] = trader.get_pnl_stats()
        else:
            results[strategy_name] = {"error": "No equity data generated"}

    for strategy, stats in results.items():
        print(f"\n=== {strategy} Results ===")
        if "error" in stats:
            print(stats["error"])
        else:
            print(f"Final Portfolio Equity: ${stats['Equity Curve'].iloc[-1]:,.2f}")
            print("PnL Statistics:")
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.2f}")
                elif key not in ["Daily Returns", "Cumulative Returns", "Equity Curve"]:
                    print(f"{key}: {value}")