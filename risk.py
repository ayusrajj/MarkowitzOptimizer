import numpy as np
import pandas as pd
from typing import Dict

class RiskEngine:
    def __init__(self):
        pass

    def calculate_advanced_metrics(
        self, price_df: pd.DataFrame, weights_dict: Dict[str, float], 
        expected_return: float, risk_free_rate: float
    ) -> dict:
        """
        Calculate Value-at-Risk (VaR), Maximum Drawdown, Sortino Ratio, 
        and Historical Cumulative Returns.
        """
        tickers = list(weights_dict.keys())
        weights = np.array([weights_dict[t] for t in tickers])

        # Calculate daily log returns
        returns_df = np.log(price_df / price_df.shift(1)).dropna()
        
        # Calculate daily portfolio returns
        returns_matrix = returns_df[tickers].to_numpy()
        portfolio_returns = np.dot(returns_matrix, weights)

        # Historical Simulation VaR
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)

        # Cumulative Returns
        # Start at 1.0
        cumulative = np.exp(np.cumsum(portfolio_returns))
        
        # Maximum Drawdown
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        # Sortino Ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        sortino_ratio = (expected_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0

        # Calmar Ratio
        calmar_ratio = expected_return / max_drawdown if max_drawdown > 0 else 0.0

        # Tail Ratio
        var_5 = np.percentile(portfolio_returns, 5)
        var_95_positive = np.percentile(portfolio_returns, 95)
        tail_ratio = abs(var_95_positive / var_5) if var_5 != 0 else 0.0

        # Daily Win Rate
        win_rate = np.sum(portfolio_returns > 0) / len(portfolio_returns) if len(portfolio_returns) > 0 else 0.0

        # Map dates to cumulative returns for graphing
        dates = returns_df.index.strftime('%Y-%m-%d').tolist()
        hist_returns = {dates[i]: float(cumulative[i]) for i in range(len(dates))}

        return {
            "value_at_risk_95": float(abs(var_95)),
            "value_at_risk_99": float(abs(var_99)),
            "maximum_drawdown": float(max_drawdown),
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
            "tail_ratio": float(tail_ratio),
            "win_rate": float(win_rate),
            "historical_returns": hist_returns
        }

    def stress_test(self, price_df: pd.DataFrame, weights_dict: Dict[str, float], shock_factor: float = -0.20) -> float:
        """
        Basic stress test simulation applying a macroeconomic shock to the portfolio.
        Returns the expected portfolio drop.
        This is a placeholder for a more complex stress vector engine.
        """
        # If we had historical macroeconomic shock dates, we would slice price_df.
        # For now, apply a generic shock vector based on beta/correlation if needed, 
        # or simply return the deterministic shock.
        return abs(shock_factor)

