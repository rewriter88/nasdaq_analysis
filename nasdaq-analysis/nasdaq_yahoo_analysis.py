"""
Enhanced Nasdaq Top-N Momentum Strategy Analysis using Yahoo Finance
===================================================================

This script implements a sophisticated momentum strategy that:
• Fetches actual market cap data from Yahoo Finance
• Builds dynamic Top-N portfolios based on market capitalization
• Rebalances when portfolio composition changes
• Compares multiple portfolio sizes (1, 2, 3, 4, 5, 6, 8, 9, 10 stocks)
• Benchmarks against QQQ with comprehensive performance metrics

Key Features:
- Uses Yahoo Finance for historical market cap and price data
- Intelligent caching system to minimize API calls
- Multiple portfolio sizes tested simultaneously  
- Professional-grade performance analytics
- Beautiful visualizations with log-scale equity curves

Dependencies:
    pip install yfinance pandas numpy matplotlib tqdm

Usage:
    1. Adjust START/END dates and SIZES in config.py
    2. Run the script
"""

import os
import json
import datetime as dt
from collections import defaultdict
from typing import Dict, List, Set
import time
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm import tqdm

# Load configuration
try:
    from config import START_DATE, END_DATE, PORTFOLIO_SIZES, CACHE_DIR, REQUEST_DELAY
    print("Configuration loaded from config.py")
except ImportError:
    print("config.py not found, using default configuration")
    
    START_DATE = "2005-01-01"  # Start date for analysis
    END_DATE = "2025-01-01"    # End date for analysis
    PORTFOLIO_SIZES = [1, 2, 3, 4, 5, 6, 8, 9, 10]  # Portfolio sizes to test
    
    CACHE_DIR = "yahoo_cache"  # Directory for caching Yahoo responses
    REQUEST_DELAY = 0.1        # Delay between requests (seconds)

# Create cache directory
YAHOO_CACHE_DIR = "yahoo_cache"
os.makedirs(YAHOO_CACHE_DIR, exist_ok=True)

class YahooDataProvider:
    """Yahoo Finance data provider with intelligent caching"""
    
    def __init__(self, cache_dir: str = "yahoo_cache"):
        self.cache_dir = cache_dir
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Simple rate limiting"""
        time_since_last = time.time() - self.last_request_time
        if time_since_last < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - time_since_last)
        self.last_request_time = time.time()
    
    def get_nasdaq_constituents(self) -> List[str]:
        """Get Nasdaq 100 constituents"""
        cache_file = os.path.join(self.cache_dir, "nasdaq_constituents.pkl")
        
        # Try cache first
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                symbols = pickle.load(f)
                print(f"Loaded {len(symbols)} Nasdaq constituents from cache")
                return symbols
        
        # Fallback to major Nasdaq stocks since Yahoo doesn't have a direct constituent API
        symbols = [
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE",
            "CRM", "PYPL", "INTC", "CSCO", "QCOM", "COST", "PEP", "AVGO", "TXN", "CMCSA",
            "HON", "AMGN", "SBUX", "GILD", "MDLZ", "BKNG", "ISRG", "REGN", "MU", "ADP",
            "LRCX", "AMAT", "KLAC", "MRVL", "ADI", "SNPS", "CDNS", "INTU", "ORLY", "FAST",
            "CTAS", "PAYX", "VRTX", "MCHP", "MNST", "CSX", "PCAR", "CHTR", "BIIB", "EA",
            "AMD", "LULU", "DXCM", "IDXX", "MAR", "MELI", "CTSH", "PANW", "ANSS", "ROST",
            "KDP", "EXC", "XEL", "AEP", "CPRT", "ODFL", "VRSK", "ROP", "CSGP", "FTNT",
            "TEAM", "WDAY", "TTWO", "CRWD", "ZS", "DDOG", "ABNB", "DASH", "KHC", "GEHC",
            "CEG", "NXPI", "TMUS", "ON", "CDW", "GFS", "FANG", "APP", "ARM", "ASML",
            "AZN", "LIN", "AXON", "BKR", "CCEP", "WBD", "TTD", "PLTR", "MDB", "MSTR", "PDD"
        ]
        
        # Cache the symbols
        with open(cache_file, 'wb') as f:
            pickle.dump(symbols, f)
        
        print(f"Using {len(symbols)} major Nasdaq stocks")
        return symbols
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get stock info including shares outstanding"""
        cache_file = os.path.join(self.cache_dir, f"{symbol}_info.pkl")
        
        # Try cache first
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        try:
            self._rate_limit()
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Cache the info
            with open(cache_file, 'wb') as f:
                pickle.dump(info, f)
                
            return info
        except Exception as e:
            print(f"Error getting info for {symbol}: {e}")
            return {}
    
    def get_historical_prices(self, symbol: str, start_date: str, end_date: str) -> pd.Series:
        """Get historical price data for a symbol from Yahoo Finance"""
        cache_file = os.path.join(self.cache_dir, f"{symbol}_prices_{start_date}_{end_date}.pkl")
        
        # Try cache first
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        try:
            self._rate_limit()
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if data.empty:
                print(f"No price data returned for {symbol}")
                return pd.Series(dtype=float)
            
            # Use Close price (already adjusted)
            prices = data['Close'].dropna()
            
            # Cache the prices
            with open(cache_file, 'wb') as f:
                pickle.dump(prices, f)
            
            print(f"Price data for {symbol}: {len(prices)} days from {prices.index.min().date()} to {prices.index.max().date()}")
            return prices
            
        except Exception as e:
            print(f"Error getting price data for {symbol}: {e}")
            return pd.Series(dtype=float)
    
    def calculate_historical_market_cap(self, symbol: str, start_date: str, end_date: str) -> pd.Series:
        """Calculate historical market cap using price * shares outstanding"""
        cache_file = os.path.join(self.cache_dir, f"{symbol}_market_cap_{start_date}_{end_date}.pkl")
        
        # Try cache first
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Get historical price data
        prices = self.get_historical_prices(symbol, start_date, end_date)
        if prices.empty:
            print(f"No price data for {symbol}")
            return pd.Series(dtype=float)
        
        # Get shares outstanding from stock info
        info = self.get_stock_info(symbol)
        shares_outstanding = info.get('sharesOutstanding', None)
        
        if shares_outstanding is None or shares_outstanding <= 0:
            # Try alternative fields
            shares_outstanding = info.get('impliedSharesOutstanding', None)
            if shares_outstanding is None:
                shares_outstanding = info.get('floatShares', None)
        
        if shares_outstanding is None or shares_outstanding <= 0:
            print(f"Could not get valid shares outstanding for {symbol}")
            return pd.Series(dtype=float)
        
        # Calculate market cap = price * shares outstanding
        market_cap = prices * shares_outstanding
        
        # Cache the market cap
        with open(cache_file, 'wb') as f:
            pickle.dump(market_cap, f)
        
        print(f"Calculated market cap for {symbol}: {len(market_cap)} days, "
              f"shares outstanding: {shares_outstanding:,.0f}")
        
        return market_cap.dropna()


class NasdaqMomentumAnalyzer:
    """Main analyzer class for Nasdaq momentum strategies"""
    
    def __init__(self, yahoo_provider: YahooDataProvider):
        self.yahoo = yahoo_provider
        self.market_cap_data = pd.DataFrame()
        self.price_data = pd.DataFrame()
        self.portfolio_compositions = {}
        
    def load_market_cap_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical market cap data for all symbols by calculating price * shares"""
        print(f"Calculating market cap data for {len(symbols)} symbols using Yahoo Finance...")
        
        market_caps = {}
        failed_symbols = []
        
        for symbol in tqdm(symbols, desc="Calculating market caps"):
            try:
                series = self.yahoo.calculate_historical_market_cap(symbol, start_date, end_date)
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
        print(f"Date range: {self.market_cap_data.index.min().date()} to {self.market_cap_data.index.max().date()}")
        
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
        """Load price data from Yahoo Finance with comprehensive caching"""
        print(f"Loading price data for {len(symbols)} symbols from Yahoo Finance...")
        
        all_price_data = {}
        failed_symbols = []
        
        for symbol in tqdm(symbols, desc="Fetching Yahoo prices"):
            try:
                series = self.yahoo.get_historical_prices(symbol, start_date, end_date)
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
            print("No price data was successfully loaded from Yahoo Finance!")
            return pd.DataFrame()
        
        # Combine all price data
        self.price_data = pd.DataFrame(all_price_data)
        self.price_data = self.price_data.ffill().dropna(how='all')
        
        print(f"Successfully loaded price data from Yahoo Finance: {len(self.price_data)} days, {len(self.price_data.columns)} symbols")
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
        print(f"NASDAQ MOMENTUM STRATEGY ANALYSIS (YAHOO FINANCE)")
        print(f"{'='*60}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Portfolio sizes: {portfolio_sizes}")
        
        # 1. Get Nasdaq constituents
        symbols = self.yahoo.get_nasdaq_constituents()
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
        
        # Add QQQ benchmark using Yahoo Finance
        print("Loading QQQ benchmark data from Yahoo Finance...")
        qqq_prices = self.yahoo.get_historical_prices("QQQ", start_date, end_date)
        
        if qqq_prices is not None and not qqq_prices.empty:
            qqq_returns = qqq_prices.pct_change().fillna(0)
            qqq_cumulative = (1 + qqq_returns).cumprod()
            results["QQQ"] = qqq_cumulative
        else:
            print("Warning: Could not load QQQ benchmark data from Yahoo Finance")
        
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
        print("PERFORMANCE SUMMARY (YAHOO FINANCE)")
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
        plt.title(f'Portfolio Value Growth - $100,000 Initial Investment (Yahoo Finance)\n{start_date} to {end_date}', 
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
        
        plt.suptitle(f'Nasdaq Top-N Momentum Strategy Analysis - Yahoo Finance\n{start_date} to {end_date}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the plot
        filename = f'nasdaq_momentum_analysis_yahoo_{start_date}_{end_date}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nChart saved as: {filename}")
        
        # Print final profit summary
        print(f"\n{'='*80}")
        print(f"FINAL PROFIT SUMMARY - $100,000 INITIAL INVESTMENT (YAHOO FINANCE)")
        print(f"{'='*80}")
        
        sorted_strategies = sorted(final_values.items(), key=lambda x: x[1]['final_value'], reverse=True)
        
        for strategy, values in sorted_strategies:
            print(f"{strategy:>8}: ${values['final_value']:>10,.0f} | "
                  f"Profit: +${values['profit']:>10,.0f} | "
                  f"Return: {values['profit_pct']:>6.1f}%")
        
        plt.show()


def main():
    """Main execution function"""
    
    try:
        # Initialize components
        yahoo_provider = YahooDataProvider(YAHOO_CACHE_DIR)
        analyzer = NasdaqMomentumAnalyzer(yahoo_provider)
        
        # Run analysis
        results = analyzer.run_full_analysis(START_DATE, END_DATE, PORTFOLIO_SIZES)
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Results saved and visualizations generated.")
        print(f"Cache directory: {YAHOO_CACHE_DIR}")
        
    except Exception as e:
        print(f"\nERROR: Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
