from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class PortfolioRequest(BaseModel):
    tickers: List[str] = Field(..., min_items=2, description="List of asset tickers to optimize.")
    start_date: str = Field(..., description="ISO format start date YYYY-MM-DD")
    end_date: str = Field(..., description="ISO format end date YYYY-MM-DD")
    strategy: str = Field("MAX_SHARPE", description="Allocation strategy: MAX_SHARPE, MIN_VOLATILITY")
    risk_free_rate: Optional[float] = Field(0.05, description="Annualized risk-free rate percentage.")

class PortfolioResponse(BaseModel):
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    value_at_risk_95: float
    value_at_risk_99: float
    maximum_drawdown: float
    sortino_ratio: float
    calmar_ratio: float
    tail_ratio: float
    win_rate: float
    historical_returns: Dict[str, float]
    execution_time_ms: float
