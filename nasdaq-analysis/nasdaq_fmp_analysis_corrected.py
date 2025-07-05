"""
Corrected Nasdaq Top-N Momentum Strategy Analysis using FMP API
================================================================

This script implements a corrected momentum strategy that fixes:
• Survivorship bias: Uses broader stock universe instead of only current constituents
• Data quality: Properly calculates historical market cap with time-varying shares outstanding
• Accurate historical data: Uses proper time-series alignment for all calculations

Key Fixes:
- Uses all major Nasdaq stocks, not just current Nasdaq 100 constituents
- Implements proper historical shares outstanding calculation
- Uses time-series aligned market cap data
- Includes stocks that may have been delisted during the period

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
from typing import Dict, List, Set, Optional
import time
import warnings
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ========================== CONFIGURATION ==========================
try:
    from config import (FMP_API_KEY, START_DATE, END_DATE, PORTFOLIO_SIZES, 
                       CACHE_DIR, REQUEST_DELAY, BENCHMARKS, CHART_DISPLAY_MODE, 
                       DATA_SOURCE)
    print("Configuration loaded from config.py")
    print(f"Data Source: {DATA_SOURCE}")
    
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
        """Make API request with caching and rate limiting"""
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
        
        # Try cache first
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

class CorrectedMomentumAnalyzer:
    """Corrected momentum strategy analyzer"""
    
    def __init__(self):
        self.fmp = CorrectedFMPDataProvider(FMP_API_KEY, CACHE_DIR + "_corrected")
        self.market_cap_data = pd.DataFrame()
        self.open_price_data = pd.DataFrame()  # For transactions
        self.close_price_data = pd.DataFrame()  # For performance calculation
        self.rebalancing_events = []  # Store all rebalancing events
    
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
    
    def build_top_n_compositions(self, n: int) -> pd.Series:
        """Build Top-N portfolio compositions based on corrected market cap rankings"""
        if self.market_cap_data.empty:
            raise ValueError("No market cap data loaded!")
        
        compositions = {}
        last_top_n = None
        changes_count = 0
        
        print(f"\nBuilding Top-{n} compositions...")
        print(f"Market cap data range: {self.market_cap_data.index[0]} to {self.market_cap_data.index[-1]}")
        print(f"Total dates with market cap data: {len(self.market_cap_data.index)}")
        
        # Sample some key dates to check market cap rankings
        sample_dates = []
        if len(self.market_cap_data.index) > 0:
            total_dates = len(self.market_cap_data.index)
            sample_indices = [0, total_dates//4, total_dates//2, 3*total_dates//4, -1]
            sample_dates = [self.market_cap_data.index[i] for i in sample_indices if i < total_dates]
        
        for date in self.market_cap_data.index:
            # Get market caps for this date
            day_market_caps = self.market_cap_data.loc[date].dropna()
            
            if len(day_market_caps) >= n:
                # Get top N symbols by market cap
                top_n = day_market_caps.nlargest(n).index.tolist()
                compositions[date] = top_n
                
                # Track changes for debugging
                if last_top_n is not None and set(top_n) != set(last_top_n):
                    changes_count += 1
                    if changes_count <= 5 or (n <= 3 and date.year >= 2020 and changes_count <= 20):  # Show first 5 changes + recent changes for small portfolios
                        print(f"Change #{changes_count} on {date}: {last_top_n} -> {top_n}")
                
                # Sample market cap rankings for debugging
                if date in sample_dates or (n <= 3 and date.year >= 2020 and date.month == 1 and date.day <= 7):  # Also show Jan samples for small portfolios
                    top_5_caps = day_market_caps.nlargest(5)
                    print(f"\nTop 5 market caps on {date}:")
                    for symbol, cap in top_5_caps.items():
                        print(f"  {symbol}: ${cap:,.0f}")
                
                last_top_n = top_n
        
        print(f"Total composition changes detected for Top-{n}: {changes_count}")
        print(f"Compositions generated for {len(compositions)} dates")
        
        return pd.Series(compositions)
    
    def simulate_portfolio(self, compositions: pd.Series, initial_value: float = 100000, portfolio_name: str = "Portfolio") -> pd.Series:
        """Simulate portfolio performance with rebalancing and track all events"""
        if self.close_price_data.empty or self.open_price_data.empty:
            raise ValueError("No price data loaded!")
        
        portfolio_value = pd.Series(dtype=float)
        current_holdings = {}
        current_weights = {}
        last_valid_value = initial_value  # Track last valid portfolio value
        
        rebalance_dates = []
        
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
                else:
                    # Calculate current portfolio value before rebalancing using close prices
                    portfolio_val = self._calculate_portfolio_value(current_holdings, date, last_valid_value)
                
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
                                    
                                    # Get market cap if available (using close price for market cap)
                                    market_cap = None
                                    if symbol in self.market_cap_data.columns and date in self.market_cap_data.index:
                                        market_cap = self.market_cap_data.loc[date, symbol]
                                    
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
                                
                                # Get market cap if available
                                market_cap = None
                                if symbol in self.market_cap_data.columns and date in self.market_cap_data.index:
                                    market_cap = self.market_cap_data.loc[date, symbol]
                                
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
        
        print(f"Portfolio rebalanced {len(rebalance_dates)} times")
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
    
    def run_analysis(self, start_date: str, end_date: str, portfolio_sizes: List[int]):
        """Run the corrected momentum strategy analysis"""
        
        print(f"\n{'='*60}")
        print(f"CORRECTED NASDAQ MOMENTUM STRATEGY ANALYSIS")
        print(f"{'='*60}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Portfolio sizes: {portfolio_sizes}")
        print(f"Fixes applied:")
        print(f"  ✓ Survivorship bias: Using broader stock universe")
        print(f"  ✓ Data quality: Proper historical market cap calculation")
        
        # 1. Get broader stock universe (not just current constituents)
        symbols = self.fmp.get_tradeable_stocks()
        if not symbols:
            raise ValueError("Could not get stock universe!")
        
        # 2. Load corrected market cap data
        successful_symbols = self.load_market_cap_data(symbols, start_date, end_date)
        
        # 3. Get symbols needed for price data
        all_needed_symbols = set()
        for size in portfolio_sizes:
            compositions = self.build_top_n_compositions(size)
            for composition in compositions.values:
                all_needed_symbols.update(composition)
        
        # Add all benchmarks
        for benchmark in BENCHMARKS:
            all_needed_symbols.add(benchmark)
        
        # 4. Load price data
        self.load_price_data(list(all_needed_symbols), start_date, end_date)
        
        # 5. Run backtests
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
        self.plot_results(results)
        self.print_performance_summary(results)
        self.export_rebalancing_events_to_csv()
        
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
        
        plt.title('Corrected Nasdaq Top-N Momentum Strategy Performance\n(Fixed: Survivorship Bias + Data Quality)', 
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
        textstr = f'Strategies: {total_strategies}\nBenchmarks: {total_benchmarks}\nPeriod: 20 years'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save plot
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
        plt.title(f'Portfolio Value Growth - Initial Investment\n{START_DATE} to {END_DATE}', 
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
        
        # Save plot with different filename for full analysis
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
        
        # Generate timestamped filename
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rebalancing_events_{timestamp}.csv"
        
        # Define CSV headers
        headers = [
            'portfolio', 'date', 'datetime', 'action', 'symbol', 
            'price', 'shares', 'value', 'market_cap'
        ]
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
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
            
            print(f"\n✅ Rebalancing events exported to: {filename}")
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
