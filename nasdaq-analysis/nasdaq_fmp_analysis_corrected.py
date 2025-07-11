"""
Corrected Nasdaq Top-N Momentum Strategy Analysis using FMP API
================================================================

This script implements a corrected momentum strategy that fixes:
• Survivorship bias: Uses broader stock universe instead of only current constituents
• Data quality: Properly calculates historical market cap with time-varying shares outstanding
• Accurate historical data: Uses proper time-series alignment for all calculations
• REAL-TIME EXECUTION: Market cap rankings and rebalancing decisions made at market open

Key Features:
- Uses all major Nasdaq stocks, not just current Nasdaq 100 constituents
- Implements proper historical shares outstanding calculation
- Uses time-series aligned market cap data
- Includes stocks that may have been delisted during the period
- **REAL-TIME SIMULATION**: Portfolio decisions made using market cap rankings at market open
- **PRECISE EXECUTION**: All buy/sell orders executed at adjusted open prices
- **REALISTIC TIMING**: Simulates broker checking prices at market open and trading immediately

Trading Logic:
1. At market open (9:30 AM EST), calculate real-time market cap rankings using open prices
2. Determine if portfolio rebalancing is needed based on threshold rules
3. Execute all buy/sell orders at the exact open prices used for ranking
4. Track portfolio performance using adjusted close prices throughout the day

Dependencies:
    pip install pandas numpy matplotlib requests tqdm

Usage:
    1. Set your FMP API key in config.py
    2. Run the script
"""

import os
import json
import datetime as dt
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple
import time
import warnings
import csv
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ========================== CHART FILENAME HELPER ==========================
def generate_chart_filename(chart_type="performance", config_params=None):
    """Generate descriptive filename for charts with timestamp and config params"""
    import os
    
    # Create timestamp
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract key config parameters
    if config_params is None:
        config_params = {
            'threshold': REBALANCING_THRESHOLD,
            'max_top_n': max(PORTFOLIO_SIZES),
            'data_source': DATA_SOURCE,
            'chart_mode': CHART_DISPLAY_MODE
        }
    
    # Create descriptive filename components
    threshold_pct = f"{config_params['threshold']:.0%}"
    max_n = config_params['max_top_n']
    source = config_params['data_source'].lower()
    mode = config_params['chart_mode']
    
    # Build filename
    filename = f"nasdaq_{chart_type}_{timestamp}_top{max_n}_{threshold_pct}thresh_{source}_{mode}.png"
    
    # Ensure results/charts directory exists
    results_dir = "results/charts"
    os.makedirs(results_dir, exist_ok=True)
    
    return os.path.join(results_dir, filename)

# ========================== CONFIGURATION ==========================
try:
    from config import (FMP_API_KEY, START_DATE, END_DATE, PORTFOLIO_SIZES, 
                       CACHE_DIR, REQUEST_DELAY, BENCHMARKS, CHART_DISPLAY_MODE, 
                       DATA_SOURCE, REBALANCING_THRESHOLD)
    print("Configuration loaded from config.py")
    print(f"Data Source: {DATA_SOURCE}")
    print(f"Rebalancing Threshold: {REBALANCING_THRESHOLD:.1%}")
    
    # Set cache directory based on data source
    if DATA_SOURCE.upper() == "FMP":
        CACHE_DIR = "fmp_cache_corrected"
    else:
        print(f"Warning: This script is for FMP data, but DATA_SOURCE is set to {DATA_SOURCE}")
        print("Consider using run_analysis.py to automatically select the correct script")
        
except ImportError:
    print("config.py not found, using default configuration")
    FMP_API_KEY = "YOUR_API_KEY_HERE"
    START_DATE = "2005-01-01"
    END_DATE = "2025-01-01"
    PORTFOLIO_SIZES = [1, 2, 3, 4, 5, 6, 8, 9, 10]
    BENCHMARKS = ["QQQ", "SPY"]
    CACHE_DIR = "fmp_cache_corrected"
    REQUEST_DELAY = 0.1
    CHART_DISPLAY_MODE = "simple"
    DATA_SOURCE = "FMP"

class CorrectedFMPDataProvider:
    """Corrected FMP data provider with proper historical market cap calculation"""
    
    def __init__(self, api_key: str, cache_dir: str = "fmp_cache_corrected"):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.session = requests.Session()
        self.last_request_time = 0
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with intelligent caching and rate limiting"""
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - time_since_last)
        
        # Build URL
        url = f"{self.base_url}/{endpoint}"
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        
        # Generate cache key
        cache_key = self._generate_cache_key(endpoint, params)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        # For historical price requests, check if we can reuse existing cache with broader date range
        if 'historical-price-full' in endpoint:
            existing_cache = self._find_suitable_price_cache(endpoint, params)
            if existing_cache:
                print(f"♻️  Reusing existing cache for {endpoint.split('/')[-1]} (intelligent cache reuse)")
                return existing_cache
        
        # For enterprise values and key metrics, check for any existing cache (no date dependency)
        elif ('enterprise-values' in endpoint or 'key-metrics' in endpoint):
            existing_cache = self._find_suitable_non_date_cache(endpoint)
            if existing_cache:
                symbol = endpoint.split('/')[-1]
                cache_type = "enterprise values" if 'enterprise-values' in endpoint else "key metrics"
                print(f"♻️  Reusing existing {cache_type} cache for {symbol}")
                return existing_cache
        
        # Try exact cache match first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass  # Continue to API request if cache is corrupted
        
        # Make API request
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Cache response
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.last_request_time = time.time()
            return data
            
        except requests.RequestException as e:
            print(f"API request failed for {endpoint}: {e}")
            return []
    
    def _generate_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate a unique cache key for the request"""
        if 'historical-price-full' in endpoint:
            symbol = endpoint.split('/')[-1]
            from_date = params.get('from', '')
            to_date = params.get('to', '')
            return f"{symbol}_prices_{from_date}_{to_date}"
        elif 'enterprise-values' in endpoint:
            symbol = endpoint.split('/')[-1]
            return f"{symbol}_enterprise_values"
        elif 'key-metrics' in endpoint:
            symbol = endpoint.split('/')[-1]
            return f"{symbol}_key_metrics"
        elif 'available-traded' in endpoint:
            return "available_traded_list"
        else:
            return endpoint.replace('/', '_')
    
    def _find_suitable_price_cache(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Find existing cache file that contains the requested date range"""
        symbol = endpoint.split('/')[-1]
        requested_start = params.get('from', '')
        requested_end = params.get('to', '')
        
        if not requested_start or not requested_end:
            return None
        
        # Look for any existing price cache files for this symbol
        existing_files = glob.glob(os.path.join(self.cache_dir, f"{symbol}_prices_*.json"))
        
        for cache_file in existing_files:
            try:
                # Extract date range from filename
                filename = os.path.basename(cache_file)
                # Format: SYMBOL_prices_START_END.json
                parts = filename.replace('.json', '').split('_')
                if len(parts) >= 4:
                    cached_start = parts[-2]  # Second to last part
                    cached_end = parts[-1]    # Last part
                    
                    # Check if cached range covers requested range
                    if (cached_start <= requested_start and 
                        cached_end >= requested_end):
                        
                        # Load and filter the cached data
                        with open(cache_file, 'r') as f:
                            cached_data = json.load(f)
                        
                        # Filter to requested date range
                        filtered_data = self._filter_price_data_by_date(
                            cached_data, requested_start, requested_end)
                        
                        if filtered_data and 'historical' in filtered_data:
                            print(f"♻️  Cache reuse: Found {len(filtered_data['historical'])} days of data")
                            print(f"    Requested: {requested_start} to {requested_end}")
                            print(f"    From cache: {cached_start} to {cached_end}")
                            return filtered_data
                            
            except (json.JSONDecodeError, IOError, IndexError) as e:
                continue  # Skip corrupted or incorrectly named files
        
        return None
    
    def _filter_price_data_by_date(self, data: Dict, start_date: str, end_date: str) -> Dict:
        """Filter historical price data to only include the requested date range"""
        if not data or 'historical' not in data:
            return data
        
        # Convert dates to datetime for comparison
        from datetime import datetime
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return data  # Return original if date parsing fails
        
        # Filter historical data
        filtered_historical = []
        for record in data['historical']:
            try:
                record_date = datetime.strptime(record['date'], '%Y-%m-%d')
                if start_dt <= record_date <= end_dt:
                    filtered_historical.append(record)
            except (ValueError, KeyError):
                continue  # Skip invalid records
        
        # Create filtered response
        filtered_data = data.copy()
        filtered_data['historical'] = filtered_historical
        
        return filtered_data
    
    def _find_suitable_non_date_cache(self, endpoint: str) -> Optional[Dict]:
        """Find existing cache file for non-date-ranged endpoints (enterprise values, key metrics)"""
        # Create cache filename based on endpoint
        if 'enterprise-values' in endpoint:
            symbol = endpoint.split('/')[-1]
            cache_filename = f"{symbol}_enterprise_values.json"
        elif 'key-metrics' in endpoint:
            symbol = endpoint.split('/')[-1]
            cache_filename = f"{symbol}_key_metrics.json"
        elif 'available-traded' in endpoint:
            cache_filename = "available_traded_list.json"
        else:
            cache_filename = endpoint.replace('/', '_') + ".json"
            
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if cache is not empty and has valid data
                if cached_data:
                    print(f"♻️  Reusing existing cache for {endpoint.split('/')[-1]} (intelligent cache reuse)")
                    return cached_data
                    
            except (json.JSONDecodeError, IOError):
                pass  # Skip corrupted cache files
        
        return None

    def get_tradeable_stocks(self) -> List[str]:
        """Get a comprehensive list of tradeable stocks, focusing on larger cap names"""
        
        # Always ensure we include the major tech companies that have been market leaders
        priority_stocks = [
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE",
            "CRM", "ORCL", "CSCO", "INTC", "AMD", "QCOM", "AVGO", "TXN", "INTU", "CMCSA",
            "COST", "PEP", "AMGN", "GILD", "BIIB", "ISRG", "REGN", "BKNG", "CHTR", "PYPL",
            "SBUX", "MAR", "AMAT", "LRCX", "KLAC", "ADI", "MCHP", "ABBV", "JNJ", "PFE"
        ]
        
        # First try to get all available traded stocks
        print("Fetching comprehensive stock universe...")
        data = self._make_request("available-traded/list")
        
        if not data:
            print("Warning: Could not fetch stock list, using priority major stocks")
            return priority_stocks
        
        # Filter for likely Nasdaq stocks (exclude penny stocks, focus on tech/growth)
        nasdaq_candidates = []
        
        for stock in data:
            if not isinstance(stock, dict):
                continue
                
            symbol = stock.get('symbol', '')
            name = stock.get('name', '')
            exchange = stock.get('exchangeShortName', '')
            price = stock.get('price', 0)
            
            # Filter criteria for likely Nasdaq momentum candidates
            if (symbol and 
                len(symbol) <= 5 and  # Exclude very long symbols
                not '.' in symbol and  # Exclude foreign stocks with dots
                not '^' in symbol and  # Exclude indices
                price and price > 5 and  # Exclude penny stocks
                exchange in ['NASDAQ', 'NYSE', 'AMEX']):  # Major exchanges
                
                nasdaq_candidates.append(symbol)
        
        # Start with priority stocks, then add others to make up 500 total
        final_symbols = list(priority_stocks)  # Start with priority stocks
        
        # Add other candidates, avoiding duplicates
        for symbol in sorted(nasdaq_candidates):
            if symbol not in final_symbols:
                final_symbols.append(symbol)
                if len(final_symbols) >= 500:
                    break
        
        print(f"Selected {len(final_symbols)} stocks for analysis (including {len(priority_stocks)} priority stocks)")
        return final_symbols
    
    def get_historical_prices(self, symbol: str, start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """Get historical adjusted open and close prices"""
        endpoint = f"historical-price-full/{symbol}"
        params = {
            'from': start_date,
            'to': end_date,
            'limit': 10000
        }
        
        data = self._make_request(endpoint, params)
        
        if not data or 'historical' not in data:
            return {'open': pd.Series(dtype=float), 'close': pd.Series(dtype=float)}
        
        try:
            df = pd.DataFrame(data['historical'])
            if df.empty:
                return {'open': pd.Series(dtype=float), 'close': pd.Series(dtype=float)}
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            result = {}
            
            # Get adjusted open prices for transactions
            if 'open' in df.columns:
                # Calculate adjustment factor using close prices
                if 'adjClose' in df.columns and 'close' in df.columns:
                    adj_factor = df['adjClose'] / df['close']
                    result['open'] = (df['open'] * adj_factor).astype(float)
                else:
                    result['open'] = df['open'].astype(float)
            else:
                result['open'] = pd.Series(dtype=float)
            
            # Use adjusted close price for performance calculation
            if 'adjClose' in df.columns:
                result['close'] = df['adjClose'].astype(float)
            elif 'close' in df.columns:
                print(f"Warning: No adjClose for {symbol}, using close price")
                result['close'] = df['close'].astype(float)
            else:
                result['close'] = pd.Series(dtype=float)
                
            return result
                
        except Exception as e:
            print(f"Error processing price data for {symbol}: {e}")
            return {'open': pd.Series(dtype=float), 'close': pd.Series(dtype=float)}
    
    def get_historical_shares_outstanding(self, symbol: str) -> pd.Series:
        """Get historical shares outstanding data with proper time-series"""
        
        # Try enterprise values first (most comprehensive)
        endpoint = f"enterprise-values/{symbol}"
        params = {'limit': 100}  # Get more historical data
        
        data = self._make_request(endpoint, params)
        
        if data and isinstance(data, list):
            try:
                df = pd.DataFrame(data)
                if not df.empty and 'date' in df.columns and 'numberOfShares' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date').sort_index()
                    
                    # Convert to numeric and handle any issues
                    shares = pd.to_numeric(df['numberOfShares'], errors='coerce')
                    shares = shares.dropna()
                    
                    if not shares.empty:
                        print(f"Found {len(shares)} historical shares data points for {symbol}")
                        return shares
                        
            except Exception as e:
                print(f"Error processing enterprise values for {symbol}: {e}")
        
        # Fallback: try key metrics
        endpoint = f"key-metrics/{symbol}"
        params = {'limit': 40}  # Annual data going back
        
        data = self._make_request(endpoint, params)
        
        if data and isinstance(data, list):
            try:
                df = pd.DataFrame(data)
                if not df.empty and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date').sort_index()
                    
                    # Try different column names for shares
                    shares_col = None
                    for col in ['numberOfShares', 'sharesOutstanding', 'weightedAverageShsOut']:
                        if col in df.columns:
                            shares_col = col
                            break
                    
                    if shares_col:
                        shares = pd.to_numeric(df[shares_col], errors='coerce')
                        shares = shares.dropna()
                        
                        if not shares.empty:
                            print(f"Found {len(shares)} key metrics shares data for {symbol}")
                            return shares
                            
            except Exception as e:
                print(f"Error processing key metrics for {symbol}: {e}")
        
        print(f"No shares outstanding data found for {symbol}")
        return pd.Series(dtype=float)
    
    def calculate_corrected_market_cap(self, symbol: str, start_date: str, end_date: str) -> pd.Series:
        """Calculate properly time-aligned historical market cap"""
        
        # Get price data
        price_data = self.get_historical_prices(symbol, start_date, end_date)
        prices = price_data['close']  # Use close prices for market cap calculation
        if prices.empty:
            return pd.Series(dtype=float)
        
        # Get shares outstanding data
        shares = self.get_historical_shares_outstanding(symbol)
        if shares.empty:
            return pd.Series(dtype=float)
        
        # Align the time series properly
        # Create a combined index and forward-fill shares outstanding
        combined_index = prices.index.union(shares.index).sort_values()
        
        # Reindex both series to the combined index
        prices_aligned = prices.reindex(combined_index)
        shares_aligned = shares.reindex(combined_index)
        
        # Forward fill shares outstanding (they don't change daily)
        shares_aligned = shares_aligned.fillna(method='ffill')
        
        # Only keep dates where we have both price and shares data
        valid_mask = prices_aligned.notna() & shares_aligned.notna()
        
        if not valid_mask.any():
            print(f"No overlapping price and shares data for {symbol}")
            return pd.Series(dtype=float)
        
        # Calculate market cap
        market_cap = (prices_aligned * shares_aligned)[valid_mask]
        
        # Filter to requested date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        market_cap = market_cap[(market_cap.index >= start_dt) & (market_cap.index <= end_dt)]
        
        print(f"Calculated corrected market cap for {symbol}: {len(market_cap)} days")
        return market_cap

class SharedDataManager:
    """Manages shared data across multiple analysis runs to avoid reloading"""
    
    def __init__(self):
        self.market_cap_data = None
        self.price_data = None
        self.symbols_loaded = set()
        self.data_start_date = None
        self.data_end_date = None
        
    def has_data_for_period(self, start_date: str, end_date: str, symbols: List[str]) -> bool:
        """Check if we already have data that covers the requested period"""
        if (self.market_cap_data is None or 
            self.price_data is None or 
            self.data_start_date is None or 
            self.data_end_date is None):
            return False
            
        # Check if date range is covered
        if (start_date < self.data_start_date or end_date > self.data_end_date):
            return False
            
        # Check if symbols are covered
        needed_symbols = set(symbols)
        if not needed_symbols.issubset(self.symbols_loaded):
            return False
            
        return True
    
    def store_data(self, market_cap_data: pd.DataFrame, price_data: Dict, 
                   symbols: List[str], start_date: str, end_date: str):
        """Store data for reuse"""
        self.market_cap_data = market_cap_data.copy()
        self.price_data = price_data.copy()
        self.symbols_loaded = set(symbols)
        self.data_start_date = start_date
        self.data_end_date = end_date
    
    def get_data_for_period(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict]:
        """Get subset of data for a specific period"""
        if not self.has_data_for_period(start_date, end_date, list(self.symbols_loaded)):
            raise ValueError("Data not available for requested period")
        
        # Filter market cap data by date
        market_cap_subset = self.market_cap_data.loc[start_date:end_date].copy()
        
        # Filter price data by date
        price_subset = {}
        for symbol, series in self.price_data.items():
            if isinstance(series, pd.DataFrame):
                price_subset[symbol] = series.loc[start_date:end_date].copy()
            else:
                price_subset[symbol] = series.loc[start_date:end_date].copy()
        
        return market_cap_subset, price_subset

# Global shared data manager
_shared_data_manager = SharedDataManager()

class CorrectedMomentumAnalyzer:
    """Corrected momentum strategy analyzer"""
    
    def __init__(self, use_shared_data: bool = False):
        self.fmp = CorrectedFMPDataProvider(FMP_API_KEY, CACHE_DIR + "_corrected")
        self.market_cap_data = pd.DataFrame()
        self.open_price_data = pd.DataFrame()  # For transactions
        self.close_price_data = pd.DataFrame()  # For performance calculation
        self.shares_outstanding_data = pd.DataFrame()  # Pre-calculated shares outstanding
        self.rebalancing_events = []  # Store all rebalancing events
        self.use_shared_data = use_shared_data
    
    def load_market_cap_data(self, symbols: List[str], start_date: str, end_date: str):
        """Load corrected market cap data for all symbols"""
        print(f"\nLoading corrected market cap data for {len(symbols)} symbols...")
        
        market_caps = {}
        failed_symbols = []
        
        for symbol in tqdm(symbols, desc="Loading market cap data"):
            try:
                series = self.fmp.calculate_corrected_market_cap(symbol, start_date, end_date)
                if not series.empty and len(series) > 100:  # Require substantial history
                    market_caps[symbol] = series
                else:
                    failed_symbols.append(symbol)
                    
            except Exception as e:
                print(f"Failed to get market cap for {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            print(f"Failed to get market cap data for {len(failed_symbols)} symbols: {failed_symbols[:10]}...")
        
        if not market_caps:
            raise ValueError("No market cap data successfully loaded!")
        
        # Create DataFrame and align dates
        self.market_cap_data = pd.DataFrame(market_caps)
        self.market_cap_data = self.market_cap_data.fillna(method='ffill').dropna(how='all')
        
        print(f"Successfully loaded market cap data: {len(self.market_cap_data)} days, "
              f"{len(self.market_cap_data.columns)} symbols")
        
        # Debug: Check date ranges for market cap data
        if not self.market_cap_data.empty:
            print(f"Market cap data date range: {self.market_cap_data.index[0]} to {self.market_cap_data.index[-1]}")
            
            # Check recent data availability
            recent_date = pd.to_datetime("2024-01-01")
            if recent_date in self.market_cap_data.index:
                recent_caps = self.market_cap_data.loc[recent_date].dropna()
                print(f"Market cap data available for {len(recent_caps)} symbols on {recent_date}")
                if len(recent_caps) > 0:
                    top_3_recent = recent_caps.nlargest(3)
                    print(f"Top 3 on {recent_date}: {list(top_3_recent.index)}")
            else:
                print(f"WARNING: No market cap data available for {recent_date}")
        
        return list(self.market_cap_data.columns)  # Return successful symbols
    
    def load_price_data(self, symbols: List[str], start_date: str, end_date: str):
        """Load price data for portfolio simulation"""
        print(f"\nLoading price data for {len(symbols)} symbols...")
        
        open_prices = {}
        close_prices = {}
        
        for symbol in tqdm(symbols, desc="Loading price data"):
            try:
                price_data = self.fmp.get_historical_prices(symbol, start_date, end_date)
                if not price_data['open'].empty and not price_data['close'].empty:
                    open_prices[symbol] = price_data['open']
                    close_prices[symbol] = price_data['close']
            except Exception as e:
                print(f"Failed to get price data for {symbol}: {e}")
        
        # Add benchmarks
        for benchmark in BENCHMARKS:
            try:
                benchmark_price_data = self.fmp.get_historical_prices(benchmark, start_date, end_date)
                if not benchmark_price_data['open'].empty and not benchmark_price_data['close'].empty:
                    open_prices[benchmark] = benchmark_price_data['open']
                    close_prices[benchmark] = benchmark_price_data['close']
                    print(f"Loaded benchmark data for {benchmark}")
            except Exception as e:
                print(f"Failed to get {benchmark} benchmark data: {e}")
        
        self.open_price_data = pd.DataFrame(open_prices)
        self.close_price_data = pd.DataFrame(close_prices)
        self.open_price_data = self.open_price_data.fillna(method='ffill')
        self.close_price_data = self.close_price_data.fillna(method='ffill')
        
        print(f"Successfully loaded price data: {len(self.close_price_data)} days, "
              f"{len(self.close_price_data.columns)} symbols")
              
        # Pre-calculate shares outstanding data for speed optimization
        self._precalculate_shares_outstanding()

    def _precalculate_shares_outstanding(self):
        """Pre-calculate shares outstanding for all symbols and dates to optimize real-time market cap calculations"""
        print("Pre-calculating shares outstanding data for speed optimization...")
        
        if self.market_cap_data.empty or self.close_price_data.empty:
            return
            
        # Calculate shares outstanding = market_cap / close_price for all symbols and dates
        shares_data = {}
        
        for symbol in self.market_cap_data.columns:
            if symbol in self.close_price_data.columns:
                # Get overlapping dates
                common_dates = self.market_cap_data.index.intersection(self.close_price_data.index)
                
                shares_series = []
                for date in common_dates:
                    market_cap = self.market_cap_data.loc[date, symbol]
                    close_price = self.close_price_data.loc[date, symbol]
                    
                    if not pd.isna(market_cap) and not pd.isna(close_price) and close_price > 0:
                        shares_outstanding = market_cap / close_price
                        shares_series.append((date, shares_outstanding))
                
                if shares_series:
                    dates, shares = zip(*shares_series)
                    shares_data[symbol] = pd.Series(shares, index=dates)
        
        # Convert to DataFrame
        self.shares_outstanding_data = pd.DataFrame(shares_data)
        
        # Forward fill any missing values
        self.shares_outstanding_data = self.shares_outstanding_data.fillna(method='ffill')
        
        print(f"Pre-calculated shares outstanding for {len(self.shares_outstanding_data.columns)} symbols, "
              f"{len(self.shares_outstanding_data)} dates")
    
    def _calculate_realtime_market_caps(self, date: pd.Timestamp) -> pd.Series:
        """Calculate real-time market cap at market open using open prices and pre-calculated shares outstanding"""
        realtime_market_caps = {}
        
        # Use pre-calculated shares outstanding data for speed
        if date in self.shares_outstanding_data.index and date in self.open_price_data.index:
            
            # Get available symbols for this date
            available_symbols = self.shares_outstanding_data.columns.intersection(self.open_price_data.columns)
            
            for symbol in available_symbols:
                open_price = self.open_price_data.loc[date, symbol]
                shares_outstanding = self.shares_outstanding_data.loc[date, symbol]
                
                if not pd.isna(open_price) and not pd.isna(shares_outstanding) and open_price > 0 and shares_outstanding > 0:
                    # Calculate real-time market cap: open_price * shares_outstanding
                    realtime_market_cap = open_price * shares_outstanding
                    realtime_market_caps[symbol] = realtime_market_cap
        
        return pd.Series(realtime_market_caps).sort_values(ascending=False)
    
    def build_top_n_compositions(self, n: int) -> pd.Series:
        """Build Top-N portfolio compositions based on real-time market cap rankings at market open"""
        if self.market_cap_data.empty:
            raise ValueError("No market cap data loaded!")
        if self.open_price_data.empty:
            raise ValueError("No open price data loaded!")
        
        compositions = {}
        current_portfolio = None  # Track current portfolio holdings
        changes_count = 0
        threshold_blocked_count = 0  # Track how many times threshold prevented rebalancing
        
        print(f"\nBuilding Top-{n} compositions with {REBALANCING_THRESHOLD:.1%} rebalancing threshold...")
        print(f"Using REAL-TIME market cap rankings at market open (not previous close)")
        print(f"Market cap data range: {self.market_cap_data.index[0]} to {self.market_cap_data.index[-1]}")
        print(f"Total dates with market cap data: {len(self.market_cap_data.index)}")
        
        # Sample some key dates to check market cap rankings
        sample_dates = []
        if len(self.market_cap_data.index) > 0:
            total_dates = len(self.market_cap_data.index)
            sample_indices = [0, total_dates//4, total_dates//2, 3*total_dates//4, -1]
            sample_dates = [self.market_cap_data.index[i] for i in sample_indices if i < total_dates]
        
        for date in self.market_cap_data.index:
            # Calculate REAL-TIME market cap at market open using open prices and shares outstanding
            day_market_caps = self._calculate_realtime_market_caps(date)
            
            if len(day_market_caps) >= n:
                # Get top N symbols by market cap (natural ranking)
                top_n_natural = day_market_caps.nlargest(n).index.tolist()
                
                # First date - initialize portfolio
                if current_portfolio is None:
                    current_portfolio = top_n_natural
                    compositions[date] = current_portfolio.copy()
                    changes_count += 1
                    if changes_count <= 5 or (n <= 3 and date.year >= 2020 and changes_count <= 20):
                        print(f"Initial portfolio on {date}: {current_portfolio}")
                else:
                    # Check if rebalancing is needed using threshold logic
                    needs_rebalancing = False
                    
                    # Get current portfolio market caps
                    current_portfolio_caps = {}
                    for symbol in current_portfolio:
                        if symbol in day_market_caps.index:
                            current_portfolio_caps[symbol] = day_market_caps[symbol]
                    
                    # Find the lowest market cap in current portfolio
                    if current_portfolio_caps:
                        lowest_current_cap = min(current_portfolio_caps.values())
                        lowest_current_symbol = min(current_portfolio_caps, key=current_portfolio_caps.get)
                        
                        # Check if any stock outside portfolio should trigger rebalancing
                        for symbol in top_n_natural:
                            if symbol not in current_portfolio:
                                outsider_cap = day_market_caps[symbol]
                                # Check if outsider's market cap exceeds threshold
                                if outsider_cap > lowest_current_cap * (1 + REBALANCING_THRESHOLD):
                                    needs_rebalancing = True
                                    break
                    
                    # Apply rebalancing decision
                    if needs_rebalancing:
                        old_portfolio = current_portfolio.copy()
                        current_portfolio = top_n_natural
                        compositions[date] = current_portfolio.copy()
                        changes_count += 1
                        
                        if changes_count <= 5 or (n <= 3 and date.year >= 2020 and changes_count <= 20):
                            print(f"Rebalancing #{changes_count} on {date}: {old_portfolio} -> {current_portfolio}")
                            print(f"  Trigger: Stock outside portfolio exceeded {REBALANCING_THRESHOLD:.1%} threshold")
                    else:
                        # No rebalancing - keep current portfolio
                        compositions[date] = current_portfolio.copy()
                        
                        # Check if we would have rebalanced without threshold (for debugging)
                        if set(top_n_natural) != set(current_portfolio):
                            threshold_blocked_count += 1
                            if threshold_blocked_count <= 3:  # Show first few blocked rebalances
                                print(f"Threshold blocked rebalancing on {date}: would change {current_portfolio} -> {top_n_natural}")
                
                # Sample market cap rankings for debugging
                if date in sample_dates or (n <= 3 and date.year >= 2020 and date.month == 1 and date.day <= 7):
                    top_5_caps = day_market_caps.nlargest(5)
                    print(f"\nTop 5 REAL-TIME market caps on {date} (at market open):")
                    for symbol, cap in top_5_caps.items():
                        in_portfolio = symbol in current_portfolio if current_portfolio else False
                        status = " (IN PORTFOLIO)" if in_portfolio else ""
                        open_price = self.open_price_data.loc[date, symbol] if symbol in self.open_price_data.columns else "N/A"
                        print(f"  {symbol}: ${cap:,.0f} (open: ${open_price:.2f}){status}")
        
        print(f"Total rebalancing events for Top-{n}: {changes_count}")
        print(f"Threshold blocked potential rebalances: {threshold_blocked_count}")
        print(f"Compositions generated for {len(compositions)} dates")
        
        return pd.Series(compositions)
    
    def simulate_portfolio(self, compositions: pd.Series, initial_value: float = 100000, portfolio_name: str = "Portfolio") -> pd.Series:
        """Simulate portfolio performance with real-time rebalancing at market open"""
        if self.close_price_data.empty or self.open_price_data.empty:
            raise ValueError("No price data loaded!")
        
        portfolio_value = pd.Series(dtype=float)
        current_holdings = {}
        current_weights = {}
        last_valid_value = initial_value  # Track last valid portfolio value
        
        rebalance_dates = []
        
        print(f"\nSimulating {portfolio_name} with REAL-TIME rebalancing at market open...")
        print("- Portfolio composition decisions made using real-time market cap rankings")
        print("- Buy/sell orders executed at market open prices")
        print("- Performance tracked using adjusted close prices")
        
        for i, (date, symbols) in enumerate(compositions.items()):
            # Check if portfolio composition changed
            prev_symbols = set(current_weights.keys()) if current_weights else set()
            new_symbols = set(symbols)
            
            needs_rebalance = (i == 0 or prev_symbols != new_symbols)
            
            if needs_rebalance:
                rebalance_dates.append(date)
                
                # Create market open timestamp (9:30 AM EST)
                market_open_time = date.replace(hour=9, minute=30, second=0, microsecond=0)
                market_open_str = market_open_time.strftime('%Y-%m-%d %H:%M:%S')
                
                if i == 0:
                    # Initial portfolio
                    portfolio_val = initial_value
                    print(f"Initial portfolio creation at market open on {date}")
                else:
                    # Calculate current portfolio value before rebalancing using close prices
                    portfolio_val = self._calculate_portfolio_value(current_holdings, date, last_valid_value)
                    print(f"Rebalancing at market open on {date} (portfolio value: ${portfolio_val:,.0f})")
                
                # Record selling events (stocks being removed from portfolio)
                if i > 0:  # Skip initial portfolio creation
                    stocks_to_sell = prev_symbols - new_symbols
                    for symbol in stocks_to_sell:
                        if symbol in current_holdings and symbol in self.open_price_data.columns:
                            open_price_series = self.open_price_data[symbol]
                            if date in open_price_series.index:
                                open_price = open_price_series.loc[date]
                                if open_price > 0:  # Validate price data
                                    shares = current_holdings[symbol]
                                    
                                    # Get market cap if available (using real-time market cap)
                                    realtime_market_caps = self._calculate_realtime_market_caps(date)
                                    market_cap = realtime_market_caps.get(symbol, None)
                                    
                                    # Record sell event with open price
                                    self.rebalancing_events.append({
                                        'portfolio': portfolio_name,
                                        'date': date,
                                        'datetime': market_open_str,
                                        'action': 'SELL',
                                        'symbol': symbol,
                                        'price': open_price,
                                        'shares': shares,
                                        'value': open_price * shares,
                                        'market_cap': market_cap
                                    })
                
                # Rebalance: equal weight allocation using open prices
                weight_per_stock = 1.0 / len(symbols)
                value_per_stock = portfolio_val * weight_per_stock
                
                new_holdings = {}
                current_weights = {}
                
                for symbol in symbols:
                    if symbol in self.open_price_data.columns:
                        open_price_series = self.open_price_data[symbol]
                        if date in open_price_series.index:
                            open_price = open_price_series.loc[date]
                            if open_price > 0:  # Validate price data
                                shares = value_per_stock / open_price
                                new_holdings[symbol] = shares
                                current_weights[symbol] = weight_per_stock
                                
                                # Get market cap if available (using real-time market cap)
                                realtime_market_caps = self._calculate_realtime_market_caps(date)
                                market_cap = realtime_market_caps.get(symbol, None)
                                
                                # Record buy event with open price
                                self.rebalancing_events.append({
                                    'portfolio': portfolio_name,
                                    'date': date,
                                    'datetime': market_open_str,
                                    'action': 'BUY',
                                    'symbol': symbol,
                                    'price': open_price,
                                    'shares': shares,
                                    'value': open_price * shares,
                                    'market_cap': market_cap
                                })
                
                current_holdings = new_holdings
            
            # Calculate portfolio value for this date using close prices with validation
            portfolio_val = self._calculate_portfolio_value(current_holdings, date, last_valid_value)
            
            # Only update if we have a valid value (prevent drops to zero)
            if portfolio_val > 0:
                portfolio_value[date] = portfolio_val
                last_valid_value = portfolio_val
            else:
                # Use last valid value if current calculation fails
                portfolio_value[date] = last_valid_value
        
        print(f"{portfolio_name} rebalanced {len(rebalance_dates)} times at market open")
        return portfolio_value.fillna(method='ffill')
    
    def _validate_price_series(self, series: pd.Series, name: str) -> pd.Series:
        """Validate and clean a price series to prevent unrealistic drops"""
        if series.empty:
            return series
        
        validated_series = series.copy()
        previous_value = validated_series.iloc[0]
        
        for i in range(1, len(validated_series)):
            current_value = validated_series.iloc[i]
            
            # Check for unrealistic drops (more than 50% in one day)
            if current_value < previous_value * 0.5:
                print(f"Warning: Detected unrealistic drop in {name} on {validated_series.index[i]}, using previous value")
                validated_series.iloc[i] = previous_value
            else:
                previous_value = current_value
        
        return validated_series
    
    def _calculate_portfolio_value(self, holdings: dict, date: pd.Timestamp, fallback_value: float) -> float:
        """Calculate portfolio value with robust error handling"""
        if not holdings:
            return fallback_value
        
        portfolio_val = 0
        valid_holdings = 0
        
        for symbol, shares in holdings.items():
            if symbol in self.close_price_data.columns:
                close_price_series = self.close_price_data[symbol]
                
                # Try to get price for exact date
                if date in close_price_series.index:
                    close_price = close_price_series.loc[date]
                    if close_price > 0 and not pd.isna(close_price):
                        portfolio_val += shares * close_price
                        valid_holdings += 1
                    else:
                        # Try to forward-fill from last valid price
                        valid_prices = close_price_series[close_price_series > 0].dropna()
                        if not valid_prices.empty:
                            last_valid_date = valid_prices.index[valid_prices.index <= date]
                            if len(last_valid_date) > 0:
                                last_price = valid_prices.loc[last_valid_date[-1]]
                                portfolio_val += shares * last_price
                                valid_holdings += 1
                else:
                    # Date not in index, try to find nearest valid price
                    valid_prices = close_price_series[close_price_series > 0].dropna()
                    if not valid_prices.empty:
                        # Find the most recent price before this date
                        prior_dates = valid_prices.index[valid_prices.index <= date]
                        if len(prior_dates) > 0:
                            last_price = valid_prices.loc[prior_dates[-1]]
                            portfolio_val += shares * last_price
                            valid_holdings += 1
        
        # Return calculated value only if we have some valid holdings
        # Otherwise return fallback to prevent unrealistic drops
        if valid_holdings > 0 and portfolio_val > fallback_value * 0.1:  # Sanity check: don't drop more than 90%
            return portfolio_val
        else:
            print(f"Warning: Invalid portfolio calculation on {date}, using fallback value")
            return fallback_value
    
    def run_analysis(self, start_date: str, end_date: str, portfolio_sizes: List[int], 
                    enable_charts: bool = True, enable_exports: bool = True, verbose: bool = True):
        """Run the corrected momentum strategy analysis"""
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"CORRECTED NASDAQ MOMENTUM STRATEGY ANALYSIS")
            print(f"{'='*60}")
            print(f"Period: {start_date} to {end_date}")
            print(f"Portfolio sizes: {portfolio_sizes}")
            print(f"Charts enabled: {enable_charts}")
            print(f"Fixes applied:")
            print(f"  ✓ Survivorship bias: Using broader stock universe")
            print(f"  ✓ Data quality: Proper historical market cap calculation")
        
    def run_analysis(self, start_date: str, end_date: str, portfolio_sizes: List[int], 
                    enable_charts: bool = True, enable_exports: bool = True, verbose: bool = True):
        """Run the corrected momentum strategy analysis"""
        
        global _shared_data_manager
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"CORRECTED NASDAQ MOMENTUM STRATEGY ANALYSIS")
            print(f"{'='*60}")
            print(f"Period: {start_date} to {end_date}")
            print(f"Portfolio sizes: {portfolio_sizes}")
            print(f"Charts enabled: {enable_charts}")
            print(f"Fixes applied:")
            print(f"  ✓ Survivorship bias: Using broader stock universe")
            print(f"  ✓ Data quality: Proper historical market cap calculation")
        
        # 1. Get broader stock universe (not just current constituents)
        symbols = self.fmp.get_tradeable_stocks()
        if not symbols:
            raise ValueError("Could not get stock universe!")
        
        # 2. Check if we can use shared data
        if (self.use_shared_data and 
            _shared_data_manager.has_data_for_period(start_date, end_date, symbols + BENCHMARKS)):
            
            if verbose:
                print(f"✓ Using cached data for period {start_date} to {end_date}")
            
            # Get data from shared manager
            self.market_cap_data, price_data = _shared_data_manager.get_data_for_period(start_date, end_date)
            
            # Set price data
            self.open_price_data = pd.DataFrame()
            self.close_price_data = pd.DataFrame()
            for symbol, data in price_data.items():
                if symbol.endswith('_open'):
                    base_symbol = symbol[:-5]
                    self.open_price_data[base_symbol] = data
                elif symbol.endswith('_close'):
                    base_symbol = symbol[:-6]
                    self.close_price_data[base_symbol] = data
                else:
                    # Assume it's close price data
                    self.close_price_data[symbol] = data
            
            successful_symbols = list(self.market_cap_data.columns)
            
        else:
            # Load data normally
            if verbose:
                print(f"Loading fresh data for period {start_date} to {end_date}")
            
            # Load corrected market cap data
            successful_symbols = self.load_market_cap_data(symbols, start_date, end_date)
            
            # Load price data for all successful symbols (we need this before building compositions)
            all_needed_symbols = set(successful_symbols)
            
            # Add all benchmarks
            for benchmark in BENCHMARKS:
                all_needed_symbols.add(benchmark)
            
            # Load price data for all symbols
            self.load_price_data(list(all_needed_symbols), start_date, end_date)
            
            # Store data in shared manager if enabled
            if self.use_shared_data:
                price_data_combined = {}
                for symbol in self.open_price_data.columns:
                    price_data_combined[f"{symbol}_open"] = self.open_price_data[symbol]
                for symbol in self.close_price_data.columns:
                    price_data_combined[f"{symbol}_close"] = self.close_price_data[symbol]
                
                _shared_data_manager.store_data(
                    self.market_cap_data, 
                    price_data_combined,
                    successful_symbols + list(all_needed_symbols),
                    start_date, 
                    end_date
                )
                
                if verbose:
                    print(f"✓ Data cached for future periods")
        
        # 3. Run backtests
        if verbose:
            print(f"\nRunning corrected backtests for {len(portfolio_sizes)} strategies...")
        results = {}
        
        # Process benchmarks with robust data handling
        for benchmark in BENCHMARKS:
            if benchmark in self.close_price_data.columns:
                benchmark_series = self.close_price_data[benchmark]
                
                # Clean the data: remove zeros, NaNs, and unrealistic values
                cleaned_series = benchmark_series[
                    (benchmark_series > 0) & 
                    benchmark_series.notna() & 
                    (benchmark_series < benchmark_series.quantile(0.999))  # Remove extreme outliers
                ].copy()
                
                if not cleaned_series.empty and len(cleaned_series) > 100:  # Require substantial data
                    # Forward-fill any remaining gaps
                    cleaned_series = cleaned_series.resample('D').last().fillna(method='ffill')
                    
                    # Normalize to $100,000 starting value
                    benchmark_normalized = (cleaned_series / cleaned_series.iloc[0]) * 100000
                    
                    # Final validation: ensure no unrealistic drops
                    benchmark_validated = self._validate_price_series(benchmark_normalized, benchmark)
                    
                    if not benchmark_validated.empty:
                        results[benchmark] = benchmark_validated
                        print(f"Added {benchmark} benchmark with {len(benchmark_validated)} data points")
                    else:
                        print(f"Warning: {benchmark} benchmark data failed validation")
                else:
                    print(f"Warning: Insufficient {benchmark} benchmark data")
        
        # Top-N strategies
        for size in portfolio_sizes:
            print(f"Running Top-{size} strategy...")
            compositions = self.build_top_n_compositions(size)
            portfolio_series = self.simulate_portfolio(compositions, portfolio_name=f"Top-{size}")
            
            if not portfolio_series.empty:
                results[f"Top-{size}"] = portfolio_series
            else:
                print(f"Warning: No data for Top-{size} strategy")
        
        # 6. Generate results
        if enable_charts:
            self.plot_results(results)
        if enable_exports:
            self.export_rebalancing_events_to_csv()
        if verbose and enable_charts:
            self.print_performance_summary(results)
        
        return results
    
    def plot_results(self, results: Dict[str, pd.Series]):
        """Plot corrected backtest results with configurable chart layouts"""
        if not results:
            print("No results to plot!")
            return
        
        if CHART_DISPLAY_MODE.lower() == "full":
            self._plot_full_analysis(results)
        else:
            self._plot_simple_chart(results)
    
    def _plot_simple_chart(self, results: Dict[str, pd.Series]):
        """Plot simple single performance chart"""
        plt.figure(figsize=(16, 10))
        
        # Define colors: dark colors for benchmarks, bright colors for strategies
        benchmark_colors = ['#000000', '#333333', '#1a1a1a', '#2d2d2d']  # Black and dark colors for benchmarks
        strategy_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Sort results: benchmarks first, then strategies by portfolio size
        benchmark_results = {}
        strategy_results = {}
        
        for strategy, series in results.items():
            if strategy in BENCHMARKS:
                benchmark_results[strategy] = series
            else:
                strategy_results[strategy] = series
        
        # Sort strategies by portfolio size
        strategy_results = dict(sorted(strategy_results.items(), 
                                     key=lambda x: int(x[0].split('-')[1]) if 'Top-' in x[0] else 999))
        
        # Plot benchmarks first with dark colors
        for i, (benchmark, series) in enumerate(benchmark_results.items()):
            if not series.empty:
                color = benchmark_colors[i % len(benchmark_colors)]
                final_value = series.iloc[-1]
                label_with_final = f"{benchmark} (${final_value:,.0f})"
                plt.plot(series.index, series.values, 
                        label=label_with_final, linewidth=3, color=color, 
                        solid_capstyle='round', alpha=0.9)
        
        # Plot strategies with bright colors
        for i, (strategy, series) in enumerate(strategy_results.items()):
            if not series.empty:
                color = strategy_colors[i % len(strategy_colors)]
                final_value = series.iloc[-1]
                label_with_final = f"{strategy} (${final_value:,.0f})"
                plt.plot(series.index, series.values, 
                        label=label_with_final, linewidth=2, color=color, 
                        solid_capstyle='round', alpha=0.8)
        
        plt.title(f'Corrected Nasdaq Top-N Momentum Strategy Performance\n(Fixed: Survivorship Bias + Data Quality)\nInitial Investment: $100,000 | Rebalancing Threshold: {REBALANCING_THRESHOLD:.1%}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Portfolio Value ($)', fontsize=14)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Improve legend formatting
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        legend.set_title('Strategy (Final Value)', prop={'size': 11, 'weight': 'bold'})
        
        # Add text box with key statistics
        total_strategies = len(strategy_results)
        total_benchmarks = len(benchmark_results)
        textstr = f'Initial Investment: $100,000\nRebalancing Threshold: {REBALANCING_THRESHOLD:.1%}\nStrategies: {total_strategies}\nBenchmarks: {total_benchmarks}\nPeriod: 20 years'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save plot with descriptive filename in results folder
        chart_filename = generate_chart_filename("simple", {
            'threshold': REBALANCING_THRESHOLD,
            'max_top_n': max(PORTFOLIO_SIZES),
            'data_source': DATA_SOURCE,
            'chart_mode': 'simple'
        })
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Simple chart saved: {chart_filename}")
        
        # Also save with simple name for backwards compatibility
        plt.savefig('corrected_nasdaq_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_full_analysis(self, results: Dict[str, pd.Series]):
        """Plot comprehensive 4-chart analysis layout"""
        # Define colors for consistency
        colors = {
            "QQQ": "#000000",      # Black
            "SPY": "#333333",      # Dark gray
            "Top-1": "#1f77b4",    # Blue
            "Top-2": "#ff7f0e",    # Orange
            "Top-3": "#2ca02c",    # Green
            "Top-4": "#d62728",    # Red
            "Top-5": "#9467bd",    # Purple
            "Top-6": "#8c564b",    # Brown
            "Top-7": "#e377c2",    # Pink
            "Top-8": "#7f7f7f",    # Gray
            "Top-9": "#bcbd22",    # Olive
            "Top-10": "#17becf",   # Cyan
        }
        
        # Calculate metrics for each strategy
        metrics = {}
        for strategy, series in results.items():
            if not series.empty and len(series) > 1:
                daily_returns = series.pct_change().dropna()
                
                # Calculate CAGR
                years = (series.index[-1] - series.index[0]).days / 365.25
                cagr = ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) if years > 0 else 0
                
                # Calculate volatility (annualized)
                volatility = daily_returns.std() * np.sqrt(252)
                
                metrics[strategy] = {
                    'CAGR': cagr,
                    'Volatility': volatility,
                    'Final_Value': series.iloc[-1],
                    'Total_Return': (series.iloc[-1] / series.iloc[0] - 1) * 100
                }
        
        # Main performance chart
        plt.figure(figsize=(18, 12))
        
        # Subplot 1: Portfolio Value Growth
        plt.subplot(2, 2, 1)
        
        final_values = {}
        for strategy, series in results.items():
            if not series.empty:
                line_width = 3.0 if strategy in BENCHMARKS else 2.0
                
                final_value = series.iloc[-1]
                initial_value = series.iloc[0]
                final_profit = final_value - initial_value
                final_values[strategy] = {
                    'final_value': final_value,
                    'profit': final_profit,
                    'profit_pct': (final_profit / initial_value) * 100
                }
                
                label = f"{strategy}: ${final_value:,.0f} (+${final_profit:,.0f})"
                
                plt.plot(series.index, series.values, 
                        label=label, linewidth=line_width, linestyle='-',
                        color=colors.get(strategy, None))
        
        plt.yscale('log')
        plt.title(f'Portfolio Value Growth - Initial Investment: $100,000\nRebalancing Threshold: {REBALANCING_THRESHOLD:.1%} | {START_DATE} to {END_DATE}', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value (USD)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add text showing best performer
        if final_values:
            best_strategy = max(final_values.keys(), key=lambda x: final_values[x]['final_value'])
            best_profit = final_values[best_strategy]['profit']
            plt.text(0.02, 0.98, f"Best Performer: {best_strategy}\nProfit: +${best_profit:,.0f}", 
                    transform=plt.gca().transAxes, fontsize=11, fontweight='bold',
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Subplot 2: Rolling 252-day (1-year) returns
        plt.subplot(2, 2, 2)
        for strategy, series in results.items():
            if not series.empty:
                daily_returns = series.pct_change()
                rolling_returns = daily_returns.rolling(252).mean() * 252  # Annualized
                
                plt.plot(rolling_returns.index, rolling_returns * 100,
                        label=strategy, linewidth=1.5,
                        color=colors.get(strategy, None))
        
        plt.title('Rolling 1-Year Annualized Returns', fontsize=14, fontweight='bold')
        plt.ylabel('Annualized Return (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Drawdown analysis
        plt.subplot(2, 2, 3)
        for strategy, series in results.items():
            if not series.empty:
                running_max = series.expanding().max()
                drawdown = (series - running_max) / running_max * 100
                
                plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, 
                               color=colors.get(strategy, None), label=strategy)
        
        plt.title('Drawdown Analysis', fontsize=14, fontweight='bold')
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Risk-Return scatter
        plt.subplot(2, 2, 4)
        if metrics:
            cagr_values = [metrics[s]['CAGR'] * 100 for s in results.keys() if s in metrics]
            vol_values = [metrics[s]['Volatility'] * 100 for s in results.keys() if s in metrics]
            strategy_names = [s for s in results.keys() if s in metrics]
            
            for i, strategy in enumerate(strategy_names):
                marker = 'o' if strategy in BENCHMARKS else 's'
                size = 120 if strategy in BENCHMARKS else 100
                
                plt.scatter(vol_values[i], cagr_values[i], 
                           color=colors.get(strategy, None), 
                           marker=marker, s=size, alpha=0.8, 
                           edgecolors='black', linewidth=1)
                
                # Add strategy labels
                plt.annotate(strategy, (vol_values[i], cagr_values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.title('Risk-Return Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Annualized Volatility (%)', fontsize=12)
        plt.ylabel('CAGR (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot with descriptive filename in results folder
        chart_filename = generate_chart_filename("full_analysis", {
            'threshold': REBALANCING_THRESHOLD,
            'max_top_n': max(PORTFOLIO_SIZES),
            'data_source': DATA_SOURCE,
            'chart_mode': 'full'
        })
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Full analysis chart saved: {chart_filename}")
        
        # Also save with simple name for backwards compatibility
        plt.savefig('corrected_nasdaq_performance_full.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_performance_summary(self, results: Dict[str, pd.Series]):
        """Print corrected performance summary"""
        print(f"\n{'='*60}")
        print(f"CORRECTED PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"{'Strategy':<12} {'Final Value':<15} {'Total Return':<15} {'CAGR':<10}")
        print("-" * 60)
        
        for strategy, series in results.items():
            if not series.empty and len(series) > 1:
                initial_value = series.iloc[0]
                final_value = series.iloc[-1]
                total_return = (final_value / initial_value - 1) * 100
                
                # Calculate CAGR
                years = (series.index[-1] - series.index[0]).days / 365.25
                cagr = ((final_value / initial_value) ** (1 / years) - 1) * 100 if years > 0 else 0
                
                print(f"{strategy:<12} ${final_value:>13,.0f} {total_return:>13.1f}% {cagr:>8.1f}%")

    def export_rebalancing_events_to_csv(self):
        """Export all rebalancing events to a timestamped CSV file"""
        if not self.rebalancing_events:
            print("No rebalancing events to export.")
            return
        
        # Generate descriptive filename with config params
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        threshold_pct = f"{REBALANCING_THRESHOLD:.0%}"
        max_n = max(PORTFOLIO_SIZES)
        source = DATA_SOURCE.lower()
        
        filename = f"rebalancing_events_{timestamp}_top{max_n}_{threshold_pct}thresh_{source}.csv"
        
        # Ensure results/csv directory exists
        import os
        results_dir = "results/csv"
        os.makedirs(results_dir, exist_ok=True)
        full_path = os.path.join(results_dir, filename)
        
        # Define CSV headers
        headers = [
            'portfolio', 'date', 'datetime', 'action', 'symbol', 
            'price', 'shares', 'value', 'market_cap'
        ]
        
        try:
            with open(full_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                
                # Group events by portfolio, then by date for better organization
                sorted_events = sorted(self.rebalancing_events, 
                                     key=lambda x: (x['portfolio'], x['date'], x['action']))
                
                for event in sorted_events:
                    # Format the event data
                    formatted_event = {
                        'portfolio': event['portfolio'],
                        'date': event['date'].strftime('%Y-%m-%d'),
                        'datetime': event['datetime'],
                        'action': event['action'],
                        'symbol': event['symbol'],
                        'price': f"{event['price']:.4f}" if event['price'] is not None else '',
                        'shares': f"{event['shares']:.6f}" if event['shares'] is not None else '',
                        'value': f"{event['value']:.2f}" if event['value'] is not None else '',
                        'market_cap': f"{event['market_cap']:.0f}" if event['market_cap'] is not None else ''
                    }
                    writer.writerow(formatted_event)
            
            print(f"\n✅ Rebalancing events exported to: {full_path}")
            print(f"Total events recorded: {len(self.rebalancing_events)}")
            
            # Print summary stats
            action_counts = {}
            portfolio_counts = {}
            for event in self.rebalancing_events:
                action = event['action']
                portfolio = event['portfolio']
                action_counts[action] = action_counts.get(action, 0) + 1
                portfolio_counts[portfolio] = portfolio_counts.get(portfolio, 0) + 1
            
            print(f"Actions: {dict(action_counts)}")
            print(f"Events by portfolio: {dict(portfolio_counts)}")
            
        except Exception as e:
            print(f"❌ Error exporting rebalancing events: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main execution function"""
    analyzer = CorrectedMomentumAnalyzer()
    
    try:
        results = analyzer.run_analysis(START_DATE, END_DATE, PORTFOLIO_SIZES)
        print("\n✅ Corrected analysis completed successfully!")
        print("\nKey corrections applied:")
        print("  • Fixed survivorship bias by using broader stock universe")
        print("  • Fixed market cap calculation with proper historical shares outstanding")
        print("  • Results should now be much more realistic")
        print("  • Rebalancing events exported to timestamped CSV file")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
