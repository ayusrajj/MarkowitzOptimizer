from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
from typing import List

from exceptions import NonPositiveDefiniteError

def _make_positive_semi_definite(matrix: np.ndarray) -> np.ndarray:
    """
    Ensure the covariance matrix is positive semi-definite.
    Negative Data Assertions: Trigger eigenvalue normalization automatically.
    """
    try:
        np.linalg.cholesky(matrix)
        return matrix
    except np.linalg.LinAlgError:
        # Matrix is not positive definite, normalize using eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        # Set negative eigenvalues to a small positive number (or 0)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        # Reconstruct the matrix
        new_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Ensure symmetry
        return (new_matrix + new_matrix.T) / 2


class AllocationStrategy(ABC):
    @abstractmethod
    def allocate(self, returns: np.ndarray, tickers: List[str], risk_free_rate: float) -> dict:
        pass


class MaxSharpeStrategy(AllocationStrategy):
    def allocate(self, returns: np.ndarray, tickers: List[str], risk_free_rate: float) -> dict:
        num_assets = len(tickers)
        avg_returns = np.mean(returns, axis=0) * 252
        
        raw_cov_matrix = np.cov(returns, rowvar=False) * 252
        cov_matrix = _make_positive_semi_definite(raw_cov_matrix)

        def negative_sharpe(weights):
            p_return = np.sum(avg_returns * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if p_vol == 0:
                return 0
            return -(p_return - risk_free_rate) / p_vol

        # System Design Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}) # Fully invested
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))           # No short-selling
        initial_weights = np.array(num_assets * [1.0 / num_assets])

        optimized = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not optimized.success:
            raise ValueError(f"Optimization failed: {optimized.message}")
            
        weights_dict = dict(zip(tickers, optimized.x.tolist()))
        p_return = np.sum(avg_returns * optimized.x)
        p_vol = np.sqrt(np.dot(optimized.x.T, np.dot(cov_matrix, optimized.x)))
        
        return {
            "weights": weights_dict,
            "expected_return": float(p_return),
            "expected_volatility": float(p_vol),
            "sharpe_ratio": float((p_return - risk_free_rate) / p_vol) if p_vol > 0 else 0.0
        }


class MinVolStrategy(AllocationStrategy):
    def allocate(self, returns: np.ndarray, tickers: List[str], risk_free_rate: float) -> dict:
        num_assets = len(tickers)
        avg_returns = np.mean(returns, axis=0) * 252
        
        raw_cov_matrix = np.cov(returns, rowvar=False) * 252
        cov_matrix = _make_positive_semi_definite(raw_cov_matrix)

        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}) 
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))           
        initial_weights = np.array(num_assets * [1.0 / num_assets])

        optimized = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not optimized.success:
            raise ValueError(f"Optimization failed: {optimized.message}")

        weights_dict = dict(zip(tickers, optimized.x.tolist()))
        p_return = np.sum(avg_returns * optimized.x)
        p_vol = np.sqrt(np.dot(optimized.x.T, np.dot(cov_matrix, optimized.x)))
        
        return {
            "weights": weights_dict,
            "expected_return": float(p_return),
            "expected_volatility": float(p_vol),
            "sharpe_ratio": float((p_return - risk_free_rate) / p_vol) if p_vol > 0 else 0.0
        }
