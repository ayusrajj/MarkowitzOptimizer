import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import httpx

from schemas import PortfolioRequest, PortfolioResponse
from data import fetch_historical_prices
from exceptions import DataInvariantError, NonPositiveDefiniteError
from strategies import MaxSharpeStrategy, MinVolStrategy
from engine import PortfolioEngineContext
from risk import RiskEngine

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Advanced Multi-Asset Portfolio Optimization & Risk Engine",
    description="Quantitative portfolio asset allocation and validation system.",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers
@app.exception_handler(DataInvariantError)
async def data_invariant_exception_handler(request, exc: DataInvariantError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_type": "DataInvariantError"},
    )

@app.exception_handler(NonPositiveDefiniteError)
async def non_positive_definite_exception_handler(request, exc: NonPositiveDefiniteError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_type": "NonPositiveDefiniteError"},
    )

@app.get("/search")
async def search_ticker(q: str = Query(..., min_length=1)):
    """Proxy endpoint to fetch ticker suggestions from Yahoo Finance"""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}&quotesCount=5&newsCount=0"
    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for quote in data.get("quotes", []):
                if quote.get("quoteType") in ("EQUITY", "ETF", "MUTUALFUND"):
                    results.append({
                        "symbol": quote.get("symbol"),
                        "name": quote.get("shortname", quote.get("longname", "")),
                        "exchange": quote.get("exchDisp", "")
                    })
            return {"results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error fetching ticker suggestions")

@app.post("/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(request: PortfolioRequest):
    start_time = time.time()
    
    # 1. Data Acquisition
    try:
        # Tuple for cache hashing
        tickers_tuple = tuple(request.tickers)
        price_df = fetch_historical_prices(tickers_tuple, request.start_date, request.end_date)
    except Exception as e:
        if isinstance(e, DataInvariantError):
            raise e
        raise HTTPException(status_code=400, detail=f"Failed to fetch data: {str(e)}")

    # 2. Strategy Selection
    strategy_map = {
        "MAX_SHARPE": MaxSharpeStrategy(),
        "MIN_VOLATILITY": MinVolStrategy()
    }
    
    strategy = strategy_map.get(request.strategy.upper())
    if not strategy:
        raise HTTPException(status_code=400, detail=f"Unsupported strategy: {request.strategy}")

    # 3. Optimization Execution
    engine = PortfolioEngineContext(strategy=strategy)
    try:
        allocation_result = engine.compute(price_df, request.tickers, request.risk_free_rate)
    except Exception as e:
        if isinstance(e, NonPositiveDefiniteError):
            raise e
        raise HTTPException(status_code=500, detail=f"Optimization engine error: {str(e)}")

    # 4. Risk Engine
    risk_engine = RiskEngine()
    try:
        metrics = risk_engine.calculate_advanced_metrics(
            price_df, 
            allocation_result["weights"],
            expected_return=allocation_result["expected_return"],
            risk_free_rate=request.risk_free_rate
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk engine error: {str(e)}")

    execution_time_ms = (time.time() - start_time) * 1000

    return PortfolioResponse(
        weights=allocation_result["weights"],
        expected_return=allocation_result["expected_return"],
        expected_volatility=allocation_result["expected_volatility"],
        sharpe_ratio=allocation_result["sharpe_ratio"],
        value_at_risk_95=metrics["value_at_risk_95"],
        value_at_risk_99=metrics["value_at_risk_99"],
        maximum_drawdown=metrics["maximum_drawdown"],
        sortino_ratio=metrics["sortino_ratio"],
        calmar_ratio=metrics["calmar_ratio"],
        tail_ratio=metrics["tail_ratio"],
        win_rate=metrics["win_rate"],
        historical_returns=metrics["historical_returns"],
        execution_time_ms=execution_time_ms
    )
