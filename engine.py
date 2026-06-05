import numpy as np
import pandas as pd
from typing import List, Any
from strategies import AllocationStrategy

class PortfolioEngineContext:
    def __init__(self, strategy: AllocationStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: AllocationStrategy):
        self._strategy = strategy

    def compute(self, price_df: pd.DataFrame, tickers: List[str], risk_free_rate: float) -> dict:
        """
        Compute optimal portfolio allocation given historical price data.
        Converts prices to historical log returns smoothly.
        """
        # Calculate log returns
        # Shift(1) gives the previous day's price
        returns_df = np.log(price_df / price_df.shift(1)).dropna()
        
        # Ensure the columns match the tickers exactly in order
        returns_matrix = returns_df[tickers].to_numpy()
        
        # Delegate down to selected abstraction strategy pattern
        return self._strategy.allocate(returns_matrix, tickers, risk_free_rate)
