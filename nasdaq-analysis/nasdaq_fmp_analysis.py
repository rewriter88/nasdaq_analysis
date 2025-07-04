"""
Enhanced Nasdaq Top-N Momentum Strategy Analysis using FMP API
==============================================================

This script implements a sophisticated momentum strategy that:
• Fetches actual market cap data from Financial Modeling Prep (FMP) API
• Builds dynamic Top-N portfolios based on market capitalization
• Rebalances when portfolio composition changes
• Compares multiple portfolio sizes (2, 3, 4, 5, 6, 8, 10 stocks)
• Benchmarks against QQQ with comprehensive performance metrics

Key Features:
- Uses FMP API for accurate historical market cap data (supports 5+ years with API key)
- Intelligent caching system to minimize API calls
- Multiple portfolio sizes tested simultaneously  
- Professional-grade performance analytics
- Beautiful visualizations with log-scale equity curves

Dependencies:
    pip install yfinance pandas numpy matplotlib requests tqdm

Usage:
    1. Set your FMP API key in the FMP_KEY variable below
    2. Adjust START/END dates and SIZES as needed
    3. Run the script
"""

import os
import json
import datetime as dt
from collections import defaultdict
from typing import Dict, List, Set
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm

# ========================== CONFIGURATION ==========================
# Try to load configuration from config.py, fallback to defaults
try:
    from config import FMP_API_KEY, START_DATE, END_DATE, PORTFOLIO_SIZES, CACHE_DIR, REQUEST_DELAY, BENCHMARKS
    print("Configuration loaded from config.py")
except ImportError:
    print("config.py not found, using default configuration")
    print("Copy config_template.py to config.py and set your API key!")
    
    START_DATE = "2005-01-01"  # Start date for analysis - 20 years back
    BENCHMARKS = ["QQQ", "SPY"]
    END_DATE = "2025-01-01"    # End date for analysis
    PORTFOLIO_SIZES = [1, 2, 3, 4, 5, 6, 8, 10]  # Portfolio sizes to test
    
    # FMP API Configuration
    # Get your free API key from: https://financialmodelingprep.com/developer/docs
    # Free tier: 5 years of data, 250 requests/day
    # Premium: 20+ years of data, unlimited requests
    FMP_API_KEY = "YOUR_FMP_API_KEY_HERE"  # Replace with your actual API key
    
    CACHE_DIR = "fmp_cache"  # Directory for caching API responses
    REQUEST_DELAY = 0.1      # Delay between API requests (seconds)
# ===================================================================

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize session for efficient HTTP requests
session = requests.Session()
session.headers.update({
    'User-Agent': 'Nasdaq-Analysis-Tool/1.0'
})

class FMPDataProvider:
    """Enhanced FMP API client with intelligent caching and rate limiting"""
    
    def __init__(self, api_key: str, cache_dir: str = "fmp_cache"):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.session = requests.Session()
        self.last_request_time = 0
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make rate-limited API request with caching"""
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - time_since_last)
        
        # Prepare request
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        # Create a simple cache filename from endpoint and symbol
        cache_key = endpoint.replace('/', '_').replace('-', '_')
        if 'historical-market-capitalization' in endpoint:
            symbol = endpoint.split('/')[-1]
            cache_key = f"{symbol}_market_cap"
        elif 'historical-price-full' in endpoint:
            symbol = endpoint.split('/')[-1]
            # Include date range in cache key for price data
            from_date = params.get('from', '') if params else ''
            to_date = params.get('to', '') if params else ''
            cache_key = f"{symbol}_prices_adj_{from_date}_{to_date}"
        elif 'enterprise-values' in endpoint:
            symbol = endpoint.split('/')[-1]
            cache_key = f"{symbol}_enterprise_values"
        elif 'key-metrics' in endpoint:
            symbol = endpoint.split('/')[-1]
            cache_key = f"{symbol}_key_metrics"
        elif 'shares_float' in endpoint:
            symbol = params.get('symbol', 'unknown') if params else 'unknown'
            cache_key = f"{symbol}_shares_float"
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        # Try cache first
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
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
    
    def get_nasdaq_constituents(self) -> List[str]:
        """Get current Nasdaq 100 constituents"""
        data = self._make_request("nasdaq_constituent")
        if not data:
            print("Warning: Could not fetch Nasdaq constituents, using fallback list")
            # Fallback to major Nasdaq stocks
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE", "CRM"]
        
        symbols = [item["symbol"] for item in data if isinstance(item, dict) and "symbol" in item]
        print(f"Found {len(symbols)} Nasdaq constituents")
        return sorted(symbols)
    
    def get_shares_outstanding(self, symbol: str) -> Dict:
        """Get historical shares outstanding data for a symbol"""
        endpoint = f"shares_float"
        params = {'symbol': symbol}
        
        data = self._make_request(endpoint, params)
        if not data:
            print(f"No shares outstanding data returned for {symbol}")
            return {}
        
        try:
            if isinstance(data, list) and data:
                # Get the most recent shares outstanding data
                shares_data = data[0] if data else {}
                return shares_data
            return {}
        except Exception as e:
            print(f"Error processing shares outstanding for {symbol}: {e}")
            return {}
    
    def get_enterprise_value_data(self, symbol: str) -> pd.DataFrame:
        """Get enterprise value data which includes shares outstanding over time"""
        endpoint = f"enterprise-values/{symbol}"
        params = {'limit': 40}  # Get quarterly data for ~10 years
        
        data = self._make_request(endpoint, params)
        if not data:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(data)
            if df.empty or 'date' not in df.columns:
                return pd.DataFrame()
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            return df
        except Exception as e:
            print(f"Error processing enterprise value data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_historical_market_cap(self, symbol: str, start_date: str, end_date: str) -> pd.Series:
        """Calculate historical market cap using price * shares outstanding"""
        
        # Get historical price data
        prices = self.get_historical_prices(symbol, start_date, end_date)
        if prices.empty:
            print(f"No price data for {symbol}")
            return pd.Series(dtype=float)
        
        # Try to get enterprise value data for shares outstanding
        ev_data = self.get_enterprise_value_data(symbol)
        
        shares_outstanding = None
        if not ev_data.empty and 'numberOfShares' in ev_data.columns:
            # Use the most recent shares outstanding data
            shares_outstanding = ev_data['numberOfShares'].dropna().iloc[-1] if len(ev_data['numberOfShares'].dropna()) > 0 else None
        
        # Fallback: try shares float endpoint
        if shares_outstanding is None:
            shares_data = self.get_shares_outstanding(symbol)
            if shares_data and 'floatShares' in shares_data:
                shares_outstanding = shares_data['floatShares']
            elif shares_data and 'outstandingShares' in shares_data:
                shares_outstanding = shares_data['outstandingShares']
        
        # If we still don't have shares outstanding, try a different approach
        if shares_outstanding is None:
            # Try to get key metrics which might have shares data
            key_metrics = self._make_request(f"key-metrics/{symbol}", {'limit': 1})
            if key_metrics and isinstance(key_metrics, list) and key_metrics:
                metrics = key_metrics[0]
                if 'numberOfShares' in metrics:
                    shares_outstanding = metrics['numberOfShares']
        
        if shares_outstanding is None or shares_outstanding <= 0:
            print(f"Could not get valid shares outstanding for {symbol}")
            return pd.Series(dtype=float)
        
        # Calculate market cap = price * shares outstanding
        market_cap = prices * shares_outstanding
        
        print(f"Calculated market cap for {symbol}: {len(market_cap)} days, "
              f"shares outstanding: {shares_outstanding:,.0f}")
        
        return market_cap.dropna()
    
    def get_historical_prices(self, symbol: str, start_date: str, end_date: str) -> pd.Series:
        """Get historical price data for a symbol from FMP"""
        endpoint = f"historical-price-full/{symbol}"
        
        # Request full OHLC data to get adjClose
        params = {
            'from': start_date,
            'to': end_date,
            'limit': 10000  # High limit to get all data
        }
        
        data = self._make_request(endpoint, params)
        if not data:
            print(f"No price data returned for {symbol}")
            return pd.Series(dtype=float)
        
        try:
            # FMP returns data in 'historical' key
            if isinstance(data, dict) and 'historical' in data:
                historical_data = data['historical']
            elif isinstance(data, list):
                historical_data = data
            else:
                print(f"Unexpected price data format for {symbol}")
                return pd.Series(dtype=float)
            
            if not historical_data:
                print(f"Empty historical price data for {symbol}")
                return pd.Series(dtype=float)
            
            df = pd.DataFrame(historical_data)
            if df.empty or 'date' not in df.columns:
                print(f"Invalid price data format for {symbol}")
                return pd.Series(dtype=float)
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            # Prefer adjusted close (adjClose) over close for proper split/dividend adjustment
            if 'adjClose' in df.columns:
                price_column = 'adjClose'
                print(f"Using adjusted close prices for {symbol}")
            elif 'close' in df.columns:
                price_column = 'close'
                print(f"Warning: Using unadjusted close prices for {symbol} (adjClose not available)")
            else:
                print(f"No price column found for {symbol}")
                return pd.Series(dtype=float)
            
            # Convert to numeric and handle any non-numeric values
            df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
            
            # Filter to requested date range
            df = df.loc[start_date:end_date]
            
            print(f"Price data for {symbol}: {len(df)} days from {df.index.min()} to {df.index.max()}")
            
            return df[price_column].dropna()
            
        except Exception as e:
            print(f"Error processing price data for {symbol}: {e}")
            return pd.Series(dtype=float)


class NasdaqMomentumAnalyzer:
    """Main analyzer class for Nasdaq momentum strategies"""
    
    def __init__(self, fmp_provider: FMPDataProvider):
        self.fmp = fmp_provider
        self.market_cap_data = pd.DataFrame()
        self.price_data = pd.DataFrame()
        self.portfolio_compositions = {}
        
    def load_market_cap_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical market cap data for all symbols by calculating price * shares"""
        print(f"Calculating market cap data for {len(symbols)} symbols using price * shares outstanding...")
        
        market_caps = {}
        failed_symbols = []
        
        for symbol in tqdm(symbols, desc="Calculating market caps"):
            try:
                series = self.fmp.calculate_historical_market_cap(symbol, start_date, end_date)
                if not series.empty:
                    market_caps[symbol] = series
                else:
                    failed_symbols.append(symbol)
            except Exception as e:
                print(f"Failed to calculate market cap for {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            print(f"Warning: Failed to calculate market cap for {len(failed_symbols)} symbols: {failed_symbols[:10]}...")
        
        if not market_caps:
            raise ValueError("No market cap data was successfully calculated!")
        
        # Combine all market cap series
        self.market_cap_data = pd.DataFrame(market_caps)
        
        # Forward fill missing values and drop completely empty rows
        self.market_cap_data = self.market_cap_data.ffill().dropna(how='all')
        
        # Show the date range we actually have
        print(f"Successfully calculated market cap data: {len(self.market_cap_data)} days, "
              f"{len(self.market_cap_data.columns)} symbols")
        print(f"Date range: {self.market_cap_data.index.min()} to {self.market_cap_data.index.max()}")
        
        return self.market_cap_data
    
    def build_top_n_compositions(self, n: int) -> pd.Series:
        """Build Top-N portfolio compositions based on market cap rankings"""
        if self.market_cap_data.empty:
            raise ValueError("Market cap data not loaded!")
        
        # Rank stocks by market cap (1 = largest)
        rankings = self.market_cap_data.rank(axis=1, method='first', ascending=False)
        
        # Get top N stocks for each date
        top_n_masks = rankings <= n
        
        # Convert to sets of symbols for each date
        compositions = top_n_masks.apply(
            lambda row: frozenset(row[row].index), axis=1
        )
        
        # Only keep dates where composition actually changed
        composition_changes = compositions[compositions != compositions.shift(1)]
        
        print(f"Top {n} strategy: {len(composition_changes)} composition changes over {len(self.market_cap_data)} days")
        
        return composition_changes
    
    def load_price_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load price data from FMP API with comprehensive caching"""
        print(f"Loading price data for {len(symbols)} symbols from FMP...")
        
        all_price_data = {}
        failed_symbols = []
        
        for symbol in tqdm(symbols, desc="Fetching FMP prices"):
            try:
                series = self.fmp.get_historical_prices(symbol, start_date, end_date)
                if not series.empty:
                    all_price_data[symbol] = series
                else:
                    failed_symbols.append(symbol)
            except Exception as e:
                print(f"Failed to get price data for {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            print(f"Warning: Failed to get price data for {len(failed_symbols)} symbols: {failed_symbols[:10]}...")
        
        if not all_price_data:
            print("No price data was successfully loaded from FMP!")
            return pd.DataFrame()
        
        # Combine all price data
        self.price_data = pd.DataFrame(all_price_data)
        self.price_data = self.price_data.ffill().dropna(how='all')
        
        print(f"Successfully loaded price data from FMP: {len(self.price_data)} days, {len(self.price_data.columns)} symbols")
        return self.price_data
    
    def backtest_strategy(self, portfolio_size: int) -> pd.Series:
        """Backtest a Top-N momentum strategy"""
        # Get portfolio compositions
        compositions = self.build_top_n_compositions(portfolio_size)
        
        # Get unique symbols that appear in any composition
        all_symbols = set()
        for composition in compositions.values:
            all_symbols.update(composition)
        
        # Ensure we have price data for all needed symbols
        missing_symbols = all_symbols - set(self.price_data.columns)
        if missing_symbols:
            print(f"Warning: Missing price data for {len(missing_symbols)} symbols in Top {portfolio_size} strategy")
        
        # Calculate daily returns
        returns = self.price_data.pct_change().fillna(0)
        
        # Initialize portfolio tracking
        portfolio_value = 1.0
        portfolio_values = []
        current_weights = {}
        
        for date in returns.index:
            # Check if we need to rebalance
            if date in compositions.index:
                # Rebalance: equal weight the new composition
                new_composition = compositions.loc[date]
                available_symbols = [s for s in new_composition if s in returns.columns]
                
                if available_symbols:
                    weight_per_stock = 1.0 / len(available_symbols)
                    current_weights = {symbol: weight_per_stock for symbol in available_symbols}
                else:
                    current_weights = {}
            
            # Calculate portfolio return for this day
            portfolio_return = 0.0
            for symbol, weight in current_weights.items():
                if symbol in returns.columns:
                    portfolio_return += weight * returns.loc[date, symbol]
            
            # Update portfolio value
            portfolio_value *= (1 + portfolio_return)
            portfolio_values.append(portfolio_value)
            
            # Update weights based on individual stock performance (let them drift)
            if current_weights:
                total_weight = 0.0
                for symbol in list(current_weights.keys()):
                    if symbol in returns.columns:
                        current_weights[symbol] *= (1 + returns.loc[date, symbol])
                        total_weight += current_weights[symbol]
                
                # Renormalize weights
                if total_weight > 0:
                    for symbol in current_weights:
                        current_weights[symbol] /= total_weight
        
        return pd.Series(portfolio_values, index=returns.index, name=f"Top {portfolio_size}")
    
    def run_full_analysis(self, start_date: str, end_date: str, portfolio_sizes: List[int]):
        """Run complete analysis for all portfolio sizes"""
        print(f"\n{'='*60}")
        print(f"NASDAQ MOMENTUM STRATEGY ANALYSIS")
        print(f"{'='*60}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Portfolio sizes: {portfolio_sizes}")
        
        # 1. Get Nasdaq constituents
        symbols = self.fmp.get_nasdaq_constituents()
        if not symbols:
            raise ValueError("Could not get Nasdaq constituents!")
        
        # 2. Load market cap data
        self.load_market_cap_data(symbols, start_date, end_date)
        
        # 3. Get all symbols needed for price data
        all_needed_symbols = set()
        for size in portfolio_sizes:
            compositions = self.build_top_n_compositions(size)
            for composition in compositions.values:
                all_needed_symbols.update(composition)
        
        all_needed_symbols.add("QQQ")  # Add benchmark
        
        # 4. Load price data
        self.load_price_data(list(all_needed_symbols), start_date, end_date)
        
        # 5. Run backtests
        print(f"\nRunning backtests for {len(portfolio_sizes)} strategies...")
        results = {}
        
        # Add QQQ benchmark using FMP
        print("Loading QQQ benchmark data from FMP...")
        qqq_prices = self.fmp.get_historical_prices("QQQ", start_date, end_date)
        
        if qqq_prices is not None and not qqq_prices.empty:
            qqq_returns = qqq_prices.pct_change().fillna(0)
            qqq_cumulative = (1 + qqq_returns).cumprod()
            results["QQQ"] = qqq_cumulative
        else:
            print("Warning: Could not load QQQ benchmark data from FMP")
        
        # Backtest each portfolio size
        for size in tqdm(portfolio_sizes, desc="Backtesting strategies"):
            try:
                strategy_performance = self.backtest_strategy(size)
                results[f"Top {size}"] = strategy_performance
            except Exception as e:
                print(f"Error backtesting Top {size} strategy: {e}")
        
        if not results:
            raise ValueError("No successful backtests completed!")
        
        # 6. Combine results and analyze
        performance_df = pd.DataFrame(results)
        self.analyze_and_plot_results(performance_df, start_date, end_date)
        
        return performance_df
    
    def analyze_and_plot_results(self, performance_df: pd.DataFrame, start_date: str, end_date: str):
        """Analyze results and create visualizations"""
        
        # Calculate performance metrics
        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        metrics = {}
        for strategy in performance_df.columns:
            total_return = performance_df[strategy].iloc[-1] / performance_df[strategy].iloc[0] - 1
            
            # Calculate annualized return (CAGR)
            years = (performance_df.index[-1] - performance_df.index[0]).days / 365.25
            cagr = (performance_df[strategy].iloc[-1] / performance_df[strategy].iloc[0]) ** (1/years) - 1
            
            # Calculate volatility
            daily_returns = performance_df[strategy].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252)
            
            # Calculate Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = cagr / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            running_max = performance_df[strategy].expanding().max()
            drawdown = (performance_df[strategy] - running_max) / running_max
            max_drawdown = drawdown.min()
            
            metrics[strategy] = {
                'Total Return': total_return,
                'CAGR': cagr,
                'Volatility': volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Final Value': performance_df[strategy].iloc[-1] * 100000  # $100k initial investment
            }
            
            final_profit = (performance_df[strategy].iloc[-1] * 100000) - 100000
            
            print(f"\n{strategy}:")
            print(f"  Total Return: {total_return:.1%}")
            print(f"  CAGR: {cagr:.2%}")
            print(f"  Volatility: {volatility:.2%}")
            print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {max_drawdown:.1%}")
            print(f"  Final Value: ${performance_df[strategy].iloc[-1] * 100000:,.0f}")
            print(f"  Total Profit: +${final_profit:,.0f}")
        
        # Create visualizations
        self.create_visualizations(performance_df, metrics, start_date, end_date)
        
        return metrics
    
    def create_visualizations(self, performance_df: pd.DataFrame, metrics: Dict, start_date: str, end_date: str):
        """Create comprehensive visualizations showing $100,000 investment growth"""
        
        # Convert to $100,000 investment
        INITIAL_INVESTMENT = 100000
        profit_df = performance_df * INITIAL_INVESTMENT
        
        # Define distinctive colors for each strategy
        colors = {
            "QQQ": "#000000",      # Black - benchmark
            "Top 1": "#FF0000",    # Red - single stock
            "Top 2": "#0066FF",    # Blue 
            "Top 3": "#00CC00",    # Green
            "Top 4": "#FF6600",    # Orange
            "Top 5": "#9933CC",    # Purple
            "Top 6": "#FF1493",    # Deep Pink
            "Top 8": "#00CED1",    # Dark Turquoise
            "Top 9": "#8B4513",    # Saddle Brown
            "Top 10": "#FFD700",   # Gold
        }
        
        # Main performance chart
        plt.figure(figsize=(18, 12))
        
        # Subplot 1: Portfolio Value Growth ($100,000 initial investment)
        plt.subplot(2, 2, 1)
        
        final_values = {}
        for strategy in profit_df.columns:
            line_width = 3.0 if strategy == 'QQQ' else 2.0
            
            final_value = profit_df[strategy].iloc[-1]
            final_profit = final_value - INITIAL_INVESTMENT
            final_values[strategy] = {
                'final_value': final_value,
                'profit': final_profit,
                'profit_pct': (final_profit / INITIAL_INVESTMENT) * 100
            }
            
            label = f"{strategy}: ${final_value:,.0f} (+${final_profit:,.0f})"
            
            plt.plot(profit_df.index, profit_df[strategy], 
                    label=label, linewidth=line_width, linestyle='-',
                    color=colors.get(strategy, None))
        
        plt.yscale('log')
        plt.title(f'Portfolio Value Growth - $100,000 Initial Investment\n{start_date} to {end_date}', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value (USD)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add text showing best performer
        best_strategy = max(final_values.keys(), key=lambda x: final_values[x]['final_value'])
        best_profit = final_values[best_strategy]['profit']
        plt.text(0.02, 0.98, f"Best Performer: {best_strategy}\nProfit: +${best_profit:,.0f}", 
                transform=plt.gca().transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Subplot 2: Rolling 252-day (1-year) returns
        plt.subplot(2, 2, 2)
        for strategy in performance_df.columns:
            daily_returns = performance_df[strategy].pct_change()
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
        for strategy in performance_df.columns:
            running_max = performance_df[strategy].expanding().max()
            drawdown = (performance_df[strategy] - running_max) / running_max * 100
            
            plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, 
                           color=colors.get(strategy, None), label=strategy)
        
        plt.title('Drawdown Analysis', fontsize=14, fontweight='bold')
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Risk-Return scatter with profit annotations
        plt.subplot(2, 2, 4)
        cagr_values = [metrics[s]['CAGR'] * 100 for s in performance_df.columns]
        vol_values = [metrics[s]['Volatility'] * 100 for s in performance_df.columns]
        
        for i, strategy in enumerate(performance_df.columns):
            marker = 'o' if strategy == 'QQQ' else 's'
            size = 120 if strategy == 'QQQ' else 100
            
            plt.scatter(vol_values[i], cagr_values[i], 
                       color=colors.get(strategy, None), 
                       marker=marker, s=size, label=strategy, alpha=0.8)
            
            # Add profit annotation
            profit = final_values[strategy]['profit']
            plt.annotate(f'+${profit/1000:.0f}K', 
                        (vol_values[i], cagr_values[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')
        
        plt.title('Risk-Return Profile with Final Profits', fontsize=14, fontweight='bold')
        plt.xlabel('Volatility (% Annualized)', fontsize=12)
        plt.ylabel('CAGR (% Annualized)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Nasdaq Top-N Momentum Strategy Analysis - 20 Year Backtest\n{start_date} to {end_date}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        filename = f'nasdaq_momentum_analysis_{start_date}_{end_date}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nChart saved as: {filename}")
        
        # Print final profit summary
        print(f"\n{'='*80}")
        print(f"FINAL PROFIT SUMMARY - $100,000 INITIAL INVESTMENT")
        print(f"{'='*80}")
        
        sorted_strategies = sorted(final_values.items(), key=lambda x: x[1]['final_value'], reverse=True)
        
        for strategy, values in sorted_strategies:
            print(f"{strategy:>8}: ${values['final_value']:>10,.0f} | "
                  f"Profit: +${values['profit']:>10,.0f} | "
                  f"Return: {values['profit_pct']:>6.1f}%")
        
        plt.show()


def main():
    """Main execution function"""
    
    # Validate API key
    if FMP_API_KEY == "YOUR_FMP_API_KEY_HERE" or not FMP_API_KEY:
        print("="*60)
        print("ERROR: FMP API KEY NOT SET!")
        print("="*60)
        print("Please set your Financial Modeling Prep API key in the FMP_API_KEY variable.")
        print("Get your free API key at: https://financialmodelingprep.com/developer/docs")
        print("Free tier includes 5 years of historical data and 250 requests/day.")
        print("="*60)
        return
    
    try:
        # Initialize components
        fmp_provider = FMPDataProvider(FMP_API_KEY, CACHE_DIR)
        analyzer = NasdaqMomentumAnalyzer(fmp_provider)
        
        # Run analysis
        results = analyzer.run_full_analysis(START_DATE, END_DATE, PORTFOLIO_SIZES)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Results saved and visualizations generated.")
        print(f"Cache directory: {CACHE_DIR}")
        
    except Exception as e:
        print(f"\nERROR: Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
