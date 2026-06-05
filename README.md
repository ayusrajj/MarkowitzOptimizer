# Advanced Multi-Asset Portfolio Optimization & Risk Engine 📈

A production-grade quantitative portfolio asset allocation and validation system. It elevates standard single-script mathematical computations into a full-stack, decoupled application featuring a Python/FastAPI backend and a premium React/Vite frontend inspired by modern fintech applications (like Groww).

![Dashboard Preview](docs/dashboard_1.png)

## Features 🚀

- **Quantitative Engine**: Computes optimal asset allocation weights using the Markowitz Modern Portfolio Theory.
- **Multiple Strategies**: Supports `Maximum Sharpe Ratio` and `Minimum Volatility` objective functions.
- **Advanced Risk Metrics**: Calculates Expected Return, Volatility, Value-at-Risk (95% & 99%), Maximum Drawdown, Sortino Ratio, Calmar Ratio, Tail Ratio, and historical Win Rate.
- **Graceful Error Handling**: Automatically interpolates minor gaps in historical trading data using forward/backward fills, while actively intercepting and gracefully raising errors for invalid tickers or completely non-positive definite covariance matrices.
- **Stunning Fintech UI**: A custom-designed React dashboard heavily inspired by Groww, featuring a deep dark-mode aesthetic with vibrant `#00D09C` teal accents. Includes interactive Recharts for cumulative returns and asset allocation donuts.
- **Yahoo Finance Autocomplete**: Dynamically search for tickers with intelligent live-fetching from Yahoo Finance.
- **Theme Toggle**: Switch seamlessly between premium Dark and Light modes.

![Allocation Breakdown](docs/dashboard_2.png)

## Architecture 🏗️

The project is heavily decoupled, strictly adhering to SOLID principles:
- `engine.py`: Defines the `PortfolioEngineContext`, enforcing the Strategy pattern.
- `strategies.py`: Implements `MaxSharpeStrategy` and `MinVolStrategy` using `scipy.optimize`. Includes eigenvalue normalization for strict mathematical compliance.
- `risk.py`: Handles historical simulations and quantitative risk modeling.
- `data.py`: Wrapper for `yfinance` utilizing memory caching (`cachetools`) to prevent repetitive identical API calls.
- `main.py`: The FastAPI controller coordinating the engine components.
- `/frontend`: An ultra-fast Vite SPA built with React and raw, carefully crafted CSS.

## Getting Started ⚙️

### Prerequisites
- Python 3.9+
- Node.js 18+

### Backend Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ayusrajj/MarkowitzOptimizer.git
   cd MarkowitzOptimizer
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
   *The backend will run on http://127.0.0.1:8000*

### Frontend Setup
1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   *The frontend will run on http://localhost:5173*

## Testing 🧪
The core backend quantitative logic is covered by Pytest unit tests. To run them:
```bash
PYTHONPATH=. pytest tests/
```

## Credits 👨‍💻
Made with ❤️ by [Ayush Raj](https://www.linkedin.com/in/ayusrajj/), IIT Guwahati.
