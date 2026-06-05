class DataInvariantError(Exception):
    """Raised when historical input pricing windows contain missing data, infinite fields, or constant flat values."""
    pass

class NonPositiveDefiniteError(Exception):
    """Raised when the calculated covariance matrix yields non-positive semi-definite properties."""
    pass
