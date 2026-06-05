import yfinance as yf
import pandas as pd
from cachetools import TTLCache, cached
from typing import List
from exceptions import DataInvariantError
import numpy as np

# Cache up to 100 queries, TTL = 24 hours (86400 seconds)
cache = TTLCache(maxsize=100, ttl=86400)

@cached(cache)
def fetch_historical_prices(tickers_tuple: tuple, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical closing prices for the given tickers.
    Uses TTLCache to avoid external API calls for identical requests within 24 hours.
    Returns a DataFrame with tickers as columns and dates as index.
    """
    tickers = list(tickers_tuple)
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # yfinance returns MultiIndex columns if multiple tickers are requested.
    # We want just the 'Close' or 'Adj Close' prices.
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            prices = data['Adj Close']
        else:
            prices = data['Close']
    else:
        # Single ticker case (though our API enforces min_items=2)
        prices = data[['Adj Close']] if 'Adj Close' in data.columns else data[['Close']]
        prices.columns = tickers

    # Forward fill and backward fill to handle minor trading gaps or holidays
    prices = prices.ffill().bfill()

    # Validate data integrity
    validate_data(prices)
    
    return prices

def validate_data(prices: pd.DataFrame):
    if prices.empty:
        raise DataInvariantError("Fetched pricing data is empty. Check if dates are valid.")
    
    if prices.isnull().values.any():
        # Identify which specific tickers have missing data
        missing_tickers = prices.columns[prices.isnull().all()].tolist()
        if missing_tickers:
            raise DataInvariantError(f"No historical data found for tickers: {missing_tickers}. Please check if they are valid.")
        raise DataInvariantError("Historical input pricing windows contain unresolvable missing data arrays (NaNs).")
    
    if np.isinf(prices.values).any():
        raise DataInvariantError("Historical input pricing windows contain infinite fields.")
        
    # Check for flat values across an asset (which would lead to zero variance and singular covariance matrices)
    variances = prices.var()
    if (variances == 0).any():
        flat_assets = variances[variances == 0].index.tolist()
        raise DataInvariantError(f"Constant flat values found across assets: {flat_assets}.")
