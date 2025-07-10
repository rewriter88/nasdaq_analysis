#!/usr/bin/env python3
"""
Enhanced NASDAQ Top-N Momentum Strategy Analysis with Open Price Execution

This version uses open prices for buy/sell execution to provide more realistic trading scenarios,
accounting for the fact that momentum signals are typically generated after market close
and executed at the next day's open.

Author: Enhanced by AI Assistant
Date: July 2025
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import FMP_API_KEY, TOP_N_STOCKS, REBALANCE_FREQUENCY_DAYS, CHART_DISPLAY_MODE

class NasdaqMomentumAnalyzer:
    """Enhanced NASDAQ momentum strategy analyzer with open price execution"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.cache_dir = "fmp_cache_corrected"
        self.session = requests.Session()
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum time between requests (100ms)
        
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _get_cache_filename(self, symbol: str, endpoint: str, **kwargs) -> str:
        """Generate cache filename based on symbol and parameters"""
        param_str = "_".join(f"{k}-{v}" for k, v in sorted(kwargs.items()) if v is not None)
        if param_str:
            return f"{symbol}_{endpoint}_{param_str}.json"
        else:
            return f"{symbol}_{endpoint}.json"
    
    def _make_request(self, url: str, cache_key: Optional[str] = None) -> Optional[Dict]:
        """Make API request with caching and rate limiting"""
        
        # Check cache first
        if cache_key:
            cache_path = os.path.join(self.cache_dir, cache_key)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError):
                    os.remove(cache_path)  # Remove corrupted cache file
        
        # Rate limiting
        self._rate_limit()
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            if cache_key and data:
                cache_path = os.path.join(self.cache_dir, cache_key)
                with open(cache_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            return data
            
        except Exception as e:
            print(f"Error making request to {url}: {e}")
            return None
    
    def get_nasdaq_stocks(self) -> List[str]:
        """Get list of NASDAQ 100 stocks"""
        url = f"{self.base_url}/symbol/available-traded/list?apikey={self.api_key}"
        cache_key = "available_traded_list.json"
        
        data = self._make_request(url, cache_key)
        if not data:
            print("Failed to fetch stock list")
            return []
        
        # Filter for NASDAQ stocks (simplified approach)
        nasdaq_symbols = []
        
        # Common NASDAQ 100 symbols (backup list)
        nasdaq_100_symbols = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'NVDA', 'PYPL', 'NFLX',
            'ADBE', 'CMCSA', 'INTC', 'CSCO', 'PEP', 'COST', 'AMGN', 'AVGO', 'TXN', 'QCOM',
            'INTU', 'AMAT', 'BKNG', 'CHTR', 'SBUX', 'GILD', 'ISRG', 'MCHP', 'REGN', 'KLAC',
            'MAR', 'LRCX', 'JNJ', 'CRM', 'ORCL', 'BIIB'
        ]
        
        # Use the predefined list as it's more reliable
        for symbol in nasdaq_100_symbols:
            nasdaq_symbols.append(symbol)
        
        print(f"Found {len(nasdaq_symbols)} NASDAQ stocks")
        return nasdaq_symbols[:50]  # Limit to 50 for performance
    
    def get_historical_prices(self, symbol: str, start_date: str = "2005-01-01", 
                            end_date: str = "2025-04-07") -> Optional[pd.DataFrame]:
        """Get historical price data for a symbol"""
        
        cache_key = self._get_cache_filename(symbol, "prices", **{
            'start': start_date, 'end': end_date
        })
        
        url = f"{self.base_url}/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={self.api_key}"
        
        data = self._make_request(url, cache_key)
        if not data or 'historical' not in data:
            return None
        
        df = pd.DataFrame(data['historical'])
        if df.empty:
            return None
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Ensure we have the necessary columns
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            return None
        
        return df[required_columns]
    
    def get_market_cap_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical market cap data"""
        
        cache_key = self._get_cache_filename(symbol, "enterprise_values")
        url = f"{self.base_url}/enterprise-values/{symbol}?limit=120&apikey={self.api_key}"
        
        data = self._make_request(url, cache_key)
        if not data:
            return None
        
        df = pd.DataFrame(data)
        if df.empty:
            return None
            
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Market cap in billions for easier reading
        if 'marketCapitalization' in df.columns:
            df['market_cap_billions'] = df['marketCapitalization'] / 1e9
        
        return df[['date', 'marketCapitalization', 'market_cap_billions']] if 'marketCapitalization' in df.columns else None
    
    def calculate_momentum(self, df: pd.DataFrame, period: int = 252) -> pd.DataFrame:
        """Calculate momentum scores using open prices for execution"""
        df = df.copy()
        
        # Calculate momentum using close prices (for signal generation)
        df['momentum_signal'] = df['close'].pct_change(period)
        
        # For execution, we'll use next day's open price
        # Shift the open price forward by 1 day to simulate next-day execution
        df['execution_price'] = df['open'].shift(-1)
        
        # Calculate returns using open-to-open execution
        df['next_open'] = df['open'].shift(-1)
        df['open_to_open_return'] = (df['next_open'] / df['open']) - 1
        
        return df
    
    def run_backtest_with_open_prices(self, start_date: str = "2005-01-01", 
                                    end_date: str = "2024-12-31") -> Tuple[pd.DataFrame, Dict]:
        """
        Run backtest using open prices for more realistic execution
        """
        print("=" * 80)
        print("🚀 ENHANCED NASDAQ TOP-N MOMENTUM STRATEGY BACKTEST (Open Price Execution)")
        print("=" * 80)
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"📊 Top N stocks: {TOP_N_STOCKS}")
        print(f"🔄 Rebalance frequency: {REBALANCE_FREQUENCY_DAYS} days")
        print(f"💰 Execution: Next day open prices")
        print("=" * 80)
        
        # Get stock universe
        nasdaq_stocks = self.get_nasdaq_stocks()
        if not nasdaq_stocks:
            raise ValueError("Failed to get NASDAQ stock list")
        
        print(f"📈 Analyzing {len(nasdaq_stocks)} stocks...")
        
        # Get historical data for all stocks
        all_data = {}
        valid_stocks = []
        
        for i, symbol in enumerate(nasdaq_stocks):
            print(f"📥 Fetching data for {symbol} ({i+1}/{len(nasdaq_stocks)})")
            
            price_data = self.get_historical_prices(symbol, start_date, end_date)
            if price_data is None or len(price_data) < 300:  # Need sufficient data
                print(f"⚠️  Skipping {symbol} - insufficient data")
                continue
            
            # Calculate momentum with open price execution
            price_data = self.calculate_momentum(price_data)
            all_data[symbol] = price_data
            valid_stocks.append(symbol)
        
        print(f"✅ Successfully loaded data for {len(valid_stocks)} stocks")
        
        if len(valid_stocks) < TOP_N_STOCKS:
            raise ValueError(f"Not enough valid stocks ({len(valid_stocks)}) for top {TOP_N_STOCKS} strategy")
        
        # Run the backtest
        return self._execute_backtest_with_open_prices(all_data, valid_stocks, start_date, end_date)
    
    def _execute_backtest_with_open_prices(self, all_data: Dict[str, pd.DataFrame], 
                                         valid_stocks: List[str], start_date: str, 
                                         end_date: str) -> Tuple[pd.DataFrame, Dict]:
        """Execute the backtest with open price execution logic"""
        
        # Create rebalancing dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        rebalance_dates = pd.date_range(start=start_dt, end=end_dt, 
                                      freq=f"{REBALANCE_FREQUENCY_DAYS}D")
        
        print(f"🗓️  Generated {len(rebalance_dates)} rebalancing dates")
        
        # Initialize portfolio tracking
        portfolio_values = []
        portfolio_returns = []
        holdings_history = []
        
        initial_capital = 100000  # $100K initial investment
        current_value = initial_capital
        
        for i, rebalance_date in enumerate(rebalance_dates[:-1]):  # Exclude last date
            print(f"⚖️  Rebalancing on {rebalance_date.strftime('%Y-%m-%d')} ({i+1}/{len(rebalance_dates)-1})")
            
            # Get momentum scores for this date
            momentum_scores = {}
            
            for symbol in valid_stocks:
                df = all_data[symbol]
                # Find the closest date
                date_mask = df['date'] <= rebalance_date
                if not date_mask.any():
                    continue
                
                latest_idx = date_mask.idxmax()
                if latest_idx < 252:  # Need at least 252 days for momentum calculation
                    continue
                
                momentum_score = df.loc[latest_idx, 'momentum_signal']
                if pd.notna(momentum_score):
                    momentum_scores[symbol] = momentum_score
            
            # Select top N stocks by momentum
            if len(momentum_scores) < TOP_N_STOCKS:
                print(f"⚠️  Only {len(momentum_scores)} stocks available, continuing...")
                continue
            
            top_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N_STOCKS]
            selected_symbols = [stock[0] for stock in top_stocks]
            
            print(f"📈 Selected stocks: {selected_symbols}")
            
            # Calculate period return using open prices
            period_return = self._calculate_period_return_open_prices(
                all_data, selected_symbols, rebalance_date, rebalance_dates[i+1]
            )
            
            if period_return is not None:
                current_value *= (1 + period_return)
                portfolio_returns.append(period_return)
                portfolio_values.append(current_value)
                holdings_history.append({
                    'date': rebalance_date,
                    'holdings': selected_symbols,
                    'momentum_scores': [score for _, score in top_stocks],
                    'portfolio_value': current_value,
                    'period_return': period_return
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(holdings_history)
        
        # Calculate performance metrics
        portfolio_returns_array = np.array(portfolio_returns)
        
        metrics = {
            'total_return': (current_value / initial_capital - 1) * 100,
            'annualized_return': ((current_value / initial_capital) ** (252 / len(portfolio_returns)) - 1) * 100,
            'volatility': np.std(portfolio_returns_array) * np.sqrt(252 / REBALANCE_FREQUENCY_DAYS) * 100,
            'sharpe_ratio': np.mean(portfolio_returns_array) / np.std(portfolio_returns_array) * np.sqrt(252 / REBALANCE_FREQUENCY_DAYS) if np.std(portfolio_returns_array) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values) * 100,
            'final_value': current_value,
            'num_periods': len(portfolio_returns)
        }
        
        return results_df, metrics
    
    def _calculate_period_return_open_prices(self, all_data: Dict[str, pd.DataFrame], 
                                           symbols: List[str], start_date: pd.Timestamp, 
                                           end_date: pd.Timestamp) -> Optional[float]:
        """Calculate portfolio return for a period using open price execution"""
        
        period_returns = []
        
        for symbol in symbols:
            df = all_data[symbol]
            
            # Find start execution price (next day's open after signal)
            start_mask = df['date'] > start_date  # Next trading day
            if not start_mask.any():
                continue
            start_idx = start_mask.idxmax()
            start_price = df.loc[start_idx, 'open']
            
            # Find end execution price (open price on rebalancing day)
            end_mask = df['date'] >= end_date
            if not end_mask.any():
                continue
            end_idx = end_mask.idxmax()
            end_price = df.loc[end_idx, 'open']
            
            if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                stock_return = (end_price / start_price) - 1
                period_returns.append(stock_return)
        
        if not period_returns:
            return None
        
        # Equal weight portfolio
        return np.mean(period_returns)
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0.0
        
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def create_performance_charts(self, results_df: pd.DataFrame, metrics: Dict, 
                                save_path: str = None):
        """Create comprehensive performance visualization"""
        
        if CHART_DISPLAY_MODE == 'none':
            print("📊 Chart display disabled in config")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NASDAQ Top-N Momentum Strategy Performance (Open Price Execution)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Portfolio Value Over Time
        ax1 = axes[0, 0]
        ax1.plot(results_df['date'], results_df['portfolio_value'], 
                linewidth=2, color='steelblue', label='Strategy')
        ax1.set_title('Portfolio Value Growth', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Period Returns Distribution
        ax2 = axes[0, 1]
        returns = results_df['period_return'].dropna()
        ax2.hist(returns * 100, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_title('Period Returns Distribution', fontweight='bold')
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe Ratio (if enough data points)
        ax3 = axes[1, 0]
        if len(results_df) > 12:
            window = min(12, len(results_df) // 2)
            rolling_returns = results_df['period_return'].rolling(window=window)
            rolling_sharpe = rolling_returns.mean() / rolling_returns.std() * np.sqrt(252 / REBALANCE_FREQUENCY_DAYS)
            ax3.plot(results_df['date'].iloc[window-1:], rolling_sharpe.iloc[window:], 
                    linewidth=2, color='green')
            ax3.set_title(f'Rolling Sharpe Ratio ({window}-period)', fontweight='bold')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        else:
            ax3.text(0.5, 0.5, 'Insufficient data\nfor rolling Sharpe', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Rolling Sharpe Ratio', fontweight='bold')
        
        # 4. Performance Metrics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        metrics_text = f"""
📊 PERFORMANCE SUMMARY
{'=' * 30}
💰 Total Return: {metrics['total_return']:.2f}%
📈 Annualized Return: {metrics['annualized_return']:.2f}%
📉 Volatility: {metrics['volatility']:.2f}%
⚡ Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
📊 Max Drawdown: {metrics['max_drawdown']:.2f}%
💵 Final Value: ${metrics['final_value']:,.0f}
🔢 Rebalancing Periods: {metrics['num_periods']}

⚙️ STRATEGY PARAMETERS
{'=' * 30}
🎯 Top N Stocks: {TOP_N_STOCKS}
🔄 Rebalance Frequency: {REBALANCE_FREQUENCY_DAYS} days
💱 Execution Method: Open Prices
"""
        
        ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save chart if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Chart saved to: {save_path}")
        
        # Display based on config
        if CHART_DISPLAY_MODE == 'show':
            plt.show()
        elif CHART_DISPLAY_MODE == 'save':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nasdaq_open_price_strategy_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Chart saved as: {filename}")
        
        plt.close()
    
    def save_results(self, results_df: pd.DataFrame, metrics: Dict):
        """Save detailed results to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        os.makedirs("results/csv", exist_ok=True)
        
        # Save detailed results
        filename = f"results/csv/nasdaq_open_price_strategy_{timestamp}.csv"
        
        # Expand holdings and momentum scores into separate columns
        expanded_results = []
        for _, row in results_df.iterrows():
            base_row = {
                'date': row['date'],
                'portfolio_value': row['portfolio_value'],
                'period_return': row['period_return']
            }
            
            # Add individual stock holdings and scores
            for i, (stock, score) in enumerate(zip(row['holdings'], row['momentum_scores'])):
                base_row[f'stock_{i+1}'] = stock
                base_row[f'momentum_score_{i+1}'] = score
            
            expanded_results.append(base_row)
        
        expanded_df = pd.DataFrame(expanded_results)
        expanded_df.to_csv(filename, index=False)
        
        # Save summary metrics
        summary_filename = f"results/csv/nasdaq_open_price_summary_{timestamp}.csv"
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(summary_filename, index=False)
        
        print(f"💾 Results saved to: {filename}")
        print(f"💾 Summary saved to: {summary_filename}")

def main():
    """Main execution function"""
    if not FMP_API_KEY:
        print("❌ Error: FMP_API_KEY not found in config.py")
        print("Please add your Financial Modeling Prep API key to config.py")
        return
    
    try:
        print("🚀 Starting NASDAQ Top-N Momentum Strategy Analysis (Open Price Execution)")
        print(f"📊 Configuration: Top {TOP_N_STOCKS} stocks, {REBALANCE_FREQUENCY_DAYS}-day rebalancing")
        
        analyzer = NasdaqMomentumAnalyzer(FMP_API_KEY)
        
        # Run the backtest
        results_df, metrics = analyzer.run_backtest_with_open_prices()
        
        # Print results
        print("\n" + "=" * 80)
        print("📊 FINAL RESULTS SUMMARY")
        print("=" * 80)
        print(f"💰 Total Return: {metrics['total_return']:.2f}%")
        print(f"📈 Annualized Return: {metrics['annualized_return']:.2f}%")
        print(f"📉 Volatility: {metrics['volatility']:.2f}%")
        print(f"⚡ Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"📊 Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"💵 Final Portfolio Value: ${metrics['final_value']:,.0f}")
        print(f"🔢 Number of Rebalancing Periods: {metrics['num_periods']}")
        print("=" * 80)
        
        # Create visualizations
        analyzer.create_performance_charts(results_df, metrics)
        
        # Save results
        analyzer.save_results(results_df, metrics)
        
        print("✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
