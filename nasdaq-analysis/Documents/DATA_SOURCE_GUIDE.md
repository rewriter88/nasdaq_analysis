# Data Source Configuration Guide

This project now supports multiple data sources for cross-validation and comparison.

## Quick Start

1. **Configure Data Source** in `config.py`:
   ```python
   DATA_SOURCE = "FMP"    # or "YAHOO"
   ```

2. **Run Analysis**:
   ```bash
   python run_analysis.py
   ```

## Data Sources

### FMP (Financial Modeling Prep) - Recommended
- **File**: `nasdaq_fmp_analysis_corrected.py`
- **Features**: Premium data, accurate market cap calculation, survivorship bias correction
- **Requirements**: API key in `config.py`
- **Cache**: `fmp_cache_corrected/`

### Yahoo Finance - For Comparison
- **File**: `nasdaq_yahoo_analysis.py`  
- **Features**: Free data, good for verification
- **Requirements**: None (uses yfinance)
- **Cache**: `yahoo_cache/`

## Chart Modes

Set `CHART_DISPLAY_MODE` in `config.py`:
- `"simple"`: Single performance chart
- `"full"`: 4-chart comprehensive analysis

## Manual Execution

You can also run scripts directly:
```bash
python nasdaq_fmp_analysis_corrected.py    # FMP data
python nasdaq_yahoo_analysis.py            # Yahoo data
```

## Verification Workflow

1. Run with FMP data: `DATA_SOURCE = "FMP"`
2. Run with Yahoo data: `DATA_SOURCE = "YAHOO"`  
3. Compare results for validation

The launcher (`run_analysis.py`) automatically selects the correct script based on your `DATA_SOURCE` setting.
