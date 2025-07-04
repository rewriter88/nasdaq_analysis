# FMP API Configuration Template
# Copy this file to config.py and set your actual API key

# Financial Modeling Prep API Key
# Get your free API key at: https://financialmodelingprep.com/developer/docs
# Free tier: 5 years of historical data, 250 requests/day
# Premium tiers: 20+ years of data, unlimited requests

FMP_API_KEY = "YOUR_FMP_API_KEY_HERE"

# Optional: Analysis Parameters
START_DATE = "2019-01-01"
END_DATE = "2025-01-01"
PORTFOLIO_SIZES = [2, 3, 4, 5, 6, 8, 10]

# Cache settings
CACHE_DIR = "fmp_cache"
REQUEST_DELAY = 0.1  # seconds between API requests
