import pandas as pd
from datetime import datetime
from utils import get_ticker_dfs
from alphas import Alpha1, Alpha2, Alpha3
from trader import Trader

# Define the backtest period as date-only, timezone-naive
start = pd.Timestamp(datetime(2015, 1, 1)).normalize()  # datetime64[ns]
end = pd.Timestamp(datetime(2024, 1, 1)).normalize()

tickers, dfs = get_ticker_dfs(start, end)

if not tickers or not dfs:
    print("No data available to proceed with backtest.")
else:
    # Define all alphas
    alpha1 = Alpha1(tickers, dfs, start, end, name="alpha1")
    alpha2 = Alpha2(tickers, dfs, start, end, name="alpha2")
    alpha3 = Alpha3(tickers, dfs, start, end, name="alpha3")
    all_alphas = [alpha1, alpha2, alpha3]

    # Dictionary to store results
    results = {}

    # Run backtest for combined alpha
    print("\n=== Running Combined Alpha Backtest ===")
    trader_combined = Trader(tickers, dfs, start, end, all_alphas)
    trader_combined.run_backtest()
    if trader_combined.equity:
        results["Combined Alpha"] = trader_combined.get_pnl_stats()
    else:
        results["Combined Alpha"] = {"error": "No equity data generated"}

    # Run backtests for individual alphas
    for alpha in all_alphas:
        alpha_name = alpha.name.capitalize()
        print(f"\n=== Running {alpha_name} Backtest ===")
        trader_individual = Trader(tickers, dfs, start, end, [alpha])  # Pass single alpha in a list
        trader_individual.run_backtest()
        if trader_individual.equity:
            results[alpha_name] = trader_individual.get_pnl_stats()
        else:
            results[alpha_name] = {"error": "No equity data generated"}

    # Display all results
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