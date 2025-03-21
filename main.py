import pandas as pd
from datetime import datetime
from utils import get_ticker_dfs, get_sp500_data, fetch_date_range
from alphas import MeanReversalAlpha, PriceRatioMeanReversalAlpha, MomentumAlpha, AdaptiveRegimeAlpha
from trader import Trader

start, end = fetch_date_range(
    '2015-01-01', '2023-12-31'
)

sp500_df = get_sp500_data(start, end)
tickers, dfs = get_ticker_dfs(start, end)           # Include user_tickers here in arguemnt as fetched from front-end

if not tickers or not dfs:
    print("No data available to proceed with backtest.")
else:
    alpha1 = MeanReversalAlpha(tickers, dfs, start, end, name="MeanReversalAlpha")
    alpha2 = PriceRatioMeanReversalAlpha(tickers, dfs, start, end, name="PriceRatioMeanAlpha")
    alpha3 = MomentumAlpha(tickers, dfs, start, end, name="MomentumAlpha")
    regime_alpha = AdaptiveRegimeAlpha(tickers, dfs, start, end, sp500_df, name="regime_switching")

    strategies = {
        "MeanReversalAlpha": [alpha1],
        "PriceRatioMeanAlpha": [alpha2],
        "MomentumAlpha": [alpha3],
        "Combined Alpha": [alpha1, alpha2, alpha3],
        "Regime Switching Alpha": [regime_alpha]
    }

    all_strategy_results = []
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

        strategy_dict = {"strategy_name": strategy}

        print(f"\n=== {strategy} Results ===")
        if "error" in stats:
            print(stats["error"])
            strategy_dict["error"] = stats["error"]
        else:
            final_equity = stats['Equity Curve'].iloc[-1]
            print(f"Final Portfolio Equity: ${final_equity:,.2f}")

            strategy_dict["final_equity"] = float(final_equity)

            print("PnL Statistics:")

            stats_dict = {}
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.2f}")
                    stats_dict[key] = float(value)
                elif key not in ["Daily Returns", "Cumulative Returns", "Equity Curve"]:
                    print(f"{key}: {value}")
                    stats_dict[key] = float(value)

            strategy_dict["statistics"] = stats_dict

        all_strategy_results.append(strategy_dict)

# print(all_strategy_results)
# Example output: front-end should be suited to this structure
# [
#     {'strategy_name': 'MeanReversalAlpha', 
#      'final_equity': 255261.4984118666, 
#      'statistics': {
#          'Total Return (%)': 155.26149841186657, 
#          'Annualized Return (%)': 12.62290828616337, 
#          'Annualized Volatility (%)': 20.90165548403795, 
#          'Sharpe Ratio': 0.6039190673582366, 
#          'Max Drawdown (%)': -39.42536228201063}
#     }, 
#     {'strategy_name': 'PriceRatioMeanAlpha', 
#      'final_equity': 168163.07443351566, 
#      'statistics': {
#          'Total Return (%)': 68.16307443351566, 
#          'Annualized Return (%)': 9.375223190706846, 
#          'Annualized Volatility (%)': 26.83762092481679, 
#          'Sharpe Ratio': 0.34933138138327163, 
#          'Max Drawdown (%)': -63.17033298084911}
#     }, 
#     {'strategy_name': 'MomentumAlpha', 
#      'final_equity': 74736.92082579329, 
#      'statistics': {
#          'Total Return (%)': -25.263079174206705, 
#          'Annualized Return (%)': -0.6929658693556769, 
#          'Annualized Volatility (%)': 22.57215181624784, 
#          'Sharpe Ratio': -0.030700035822764038, 
#          'Max Drawdown (%)': -58.10945023173749}
#     }, 
#     {'strategy_name': 'Combined Alpha', 
#      'final_equity': 83718.99592249075, 
#      'statistics': {
#          'Total Return (%)': -16.28100407750925, 
#          'Annualized Return (%)': 0.8024958618085644, 
#          'Annualized Volatility (%)': 23.59653669787269, 
#          'Sharpe Ratio': 0.03400905277260083, 
#          'Max Drawdown (%)': -76.77629843594688}
#     }, 
#     {'strategy_name': 'Regime Switching Alpha', 
#      'final_equity': 151563.94572384172, 
#      'statistics': {
#          'Total Return (%)': 51.563945723841734, 
#          'Annualized Return (%)': 7.221421172487232, 
#          'Annualized Volatility (%)': 22.7608264958023, 
#          'Sharpe Ratio': 0.3172741189261758, 
#          'Max Drawdown (%)': -49.59445872660554}
#     }
# ]