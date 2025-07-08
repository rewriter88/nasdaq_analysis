# Nasdaq Top-N Momentum Strategy Analysis with FMP API

## Overview

This program implements a sophisticated momentum strategy that dynamically selects the top N Nasdaq stocks based on market capitalization and tests their performance over time. It uses the Financial Modeling Prep (FMP) API to get accurate historical market cap data, enabling analysis of periods longer than 5 years with proper API credentials.

## Key Features

- **Real Market Cap Data**: Uses FMP API for accurate historical market capitalization data
- **Multiple Portfolio Sizes**: Tests portfolios of 2, 3, 4, 5, 6, 8, and 10 stocks simultaneously
- **Dynamic Rebalancing**: Automatically rebalances when top-N composition changes
- **Comprehensive Analysis**: Includes performance metrics, risk analysis, and visualizations
- **Intelligent Caching**: Minimizes API calls by caching responses locally
- **Professional Visualizations**: Creates publication-quality charts and analysis

## Performance Metrics Calculated

- Total Return and Compound Annual Growth Rate (CAGR)
- Volatility and Sharpe Ratio
- Maximum Drawdown
- Rolling Returns Analysis
- Risk-Return Comparison vs. QQQ Benchmark

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get FMP API Key

1. Go to [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs)
2. Sign up for a free account
3. Get your API key from the dashboard

**Free Tier Limits:**
- 5 years of historical data
- 250 requests per day
- Perfect for testing and moderate analysis

**Premium Tiers:**
- 20+ years of historical data  
- Unlimited requests
- Recommended for extensive backtesting

### 3. Configure API Key

**Option A: Use Configuration File (Recommended)**

1. Copy the template:
   ```bash
   cp config_template.py config.py
   ```

2. Edit `config.py` and set your API key:
   ```python
   FMP_API_KEY = "your_actual_api_key_here"
   ```

**Option B: Edit the Main Script**

Alternatively, you can directly edit `nasdaq_fmp_analysis.py` and replace:
```python
FMP_API_KEY = "YOUR_FMP_API_KEY_HERE"
```

### 4. Run the Analysis

```bash
python nasdaq_fmp_analysis.py
```

## Configuration Options

You can customize the analysis by modifying these parameters in `config.py`:

```python
# Analysis period
START_DATE = "2019-01-01"
END_DATE = "2025-01-01"

# Portfolio sizes to test
PORTFOLIO_SIZES = [2, 3, 4, 5, 6, 8, 10]

# Cache settings
CACHE_DIR = "fmp_cache"
REQUEST_DELAY = 0.1  # seconds between API requests
```

## Program Flow

1. **Data Collection**:
   - Fetches current Nasdaq 100 constituents from FMP
   - Downloads historical market cap data for each symbol
   - Caches all data locally to minimize future API calls

2. **Strategy Construction**:
   - For each date, ranks all stocks by market capitalization
   - Builds Top-N portfolios (equal-weighted)
   - Tracks composition changes and rebalancing dates

3. **Backtesting**:
   - Simulates portfolio performance with realistic rebalancing
   - Allows weights to drift between rebalancing dates
   - Downloads price data from Yahoo Finance for execution

4. **Analysis & Visualization**:
   - Calculates comprehensive performance metrics
   - Creates professional visualizations
   - Compares all strategies against QQQ benchmark

## Output

The program generates:

- **Console Output**: Detailed performance metrics for each strategy
- **Visualization**: Multi-panel chart showing:
  - Cumulative performance (log scale)
  - Rolling 60-day returns
  - Drawdown analysis
  - Risk-return scatter plot
- **Chart File**: High-resolution PNG saved as `nasdaq_momentum_analysis_[dates].png`

## Sample Results

```
PERFORMANCE SUMMARY
============================================================

QQQ:
  Total Return: 156.3%
  CAGR: 12.4%
  Volatility: 18.2%
  Sharpe Ratio: 0.68
  Max Drawdown: -32.1%
  Final Value: $256.30

Top 6:
  Total Return: 203.7%
  CAGR: 15.1%
  Volatility: 21.4%
  Sharpe Ratio: 0.71
  Max Drawdown: -28.9%
  Final Value: $303.70
```

## Understanding the Strategy

This momentum strategy is based on the principle that the largest companies by market cap tend to have strong momentum characteristics. The strategy:

1. **Selects Winners**: Focuses on the largest N companies by market cap
2. **Momentum Capture**: Benefits from continued growth of large, successful companies
3. **Automatic Rebalancing**: Adapts to changing market leadership
4. **Concentration Effect**: Smaller portfolios (Top 2-6) often outperform due to concentration in mega-caps

## Troubleshooting

**API Key Issues:**
- Ensure your API key is correctly set
- Check that you haven't exceeded daily request limits
- Verify your account status on the FMP website

**Data Issues:**
- Some symbols may not have complete historical data
- The program gracefully handles missing data
- Check the console output for any warnings

**Performance Issues:**
- First run may be slow due to API calls
- Subsequent runs use cached data and are much faster
- Consider reducing the date range or portfolio sizes for testing

## Files Created

- `fmp_cache/`: Directory containing cached API responses
- `nasdaq_momentum_analysis_[dates].png`: Performance visualization
- `config.py`: Your API configuration (if created)

## Extending the Analysis

You can easily modify the program to:

- Test different rebalancing frequencies
- Add transaction costs
- Implement different weighting schemes (market cap vs. equal weight)
- Test other universes (S&P 500, Russell 2000, etc.)
- Add more sophisticated risk management

## Support

For issues with:
- **FMP API**: Contact Financial Modeling Prep support
- **Yahoo Finance data**: Check yfinance library documentation
- **Script functionality**: Review the code comments and error messages

## Disclaimer

This tool is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough due diligence before making investment decisions.
