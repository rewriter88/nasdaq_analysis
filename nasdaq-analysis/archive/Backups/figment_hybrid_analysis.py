"""
Hybrid Nasdaq Analysis combining C_Figment-Tindex threshold logic with FMP market cap data

This script implements the sophisticated threshold-based rebalancing from C_Figment-Tindex
but uses your validated FMP market cap data instead of requiring external data files.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FigmentHybridAnalyzer:
    def __init__(self, fmp_cache_dir='fmp_cache'):
        self.fmp_cache_dir = fmp_cache_dir
        self.market_cap_data = {}
        self.price_data = {}
        
    def load_fmp_market_cap_data(self, symbols: List[str]) -> pd.DataFrame:
        """Load market cap data from FMP cache files"""
        all_market_caps = {}
        
        for symbol in symbols:
            cache_file = os.path.join(self.fmp_cache_dir, f"{symbol}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to DataFrame and extract market cap
                df = pd.DataFrame(data)
                if not df.empty and 'date' in df.columns and 'marketCap' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date').sort_index()
                    # Get daily market cap data
                    all_market_caps[symbol] = df['marketCap']
        
        if not all_market_caps:
            print("No market cap data found in cache files")
            return pd.DataFrame()
            
        # Combine all market cap data
        market_cap_df = pd.DataFrame(all_market_caps)
        return market_cap_df.fillna(method='ffill').dropna(how='all')
    
    def detect_changes_with_threshold(self, market_cap_df: pd.DataFrame, 
                                    threshold: float = 0.005, 
                                    top_n: int = 10) -> Dict:
        """
        C_Figment-style threshold-based change detection
        
        Args:
            market_cap_df: DataFrame with market cap data
            threshold: Minimum relative change needed to trigger rebalancing (0.5% default)
            top_n: Number of top stocks to track
        
        Returns:
            Dictionary of rebalancing dates and their compositions
        """
        result_dict = {}
        
        # Get initial top N components
        first_row = market_cap_df.iloc[0].sort_values(ascending=False)
        previous_components = set(first_row.head(top_n).index)
        result_dict[market_cap_df.index[0]] = first_row.head(top_n)
        
        print(f"Initial composition ({market_cap_df.index[0].strftime('%Y-%m-%d')}): {list(previous_components)}")
        
        rebalance_count = 0
        
        for i in range(1, len(market_cap_df)):
            current_date = market_cap_df.index[i]
            current_row = market_cap_df.iloc[i].sort_values(ascending=False)
            current_components = set(current_row.head(top_n).index)
            
            # Check if composition changed
            if previous_components != current_components:
                # Get the market cap of the smallest component in current portfolio
                prev_row = market_cap_df.iloc[i-1]
                curr_last_value = prev_row.loc[list(previous_components)].sort_values(ascending=False).iloc[-1]
                
                # Get the market cap of the smallest component in new top N
                new_last_value = current_row.head(top_n).iloc[-1]
                new_last_symbol = current_row.head(top_n).index[-1]
                
                # Apply threshold logic: only rebalance if the change is significant
                if (new_last_value > curr_last_value * (1 + threshold) and 
                    new_last_symbol not in previous_components):
                    
                    previous_components = current_components
                    result_dict[current_date] = current_row.head(top_n)
                    rebalance_count += 1
                    
                    print(f"Rebalance #{rebalance_count} ({current_date.strftime('%Y-%m-%d')}): "
                          f"New component: {new_last_symbol}, "
                          f"Market cap change: {((new_last_value/curr_last_value - 1)*100):.2f}%")
        
        print(f"\nTotal rebalances: {rebalance_count}")
        return result_dict
    
    def get_price_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download price data for symbols"""
        print(f"Downloading price data for {len(symbols)} symbols...")
        
        try:
            data = yf.download(symbols, start=start_date, end=end_date, auto_adjust=True)
            if len(symbols) == 1:
                return pd.DataFrame({symbols[0]: data['Close']})
            else:
                return data['Close'].fillna(method='ffill')
        except Exception as e:
            print(f"Error downloading data: {e}")
            return pd.DataFrame()
    
    def calculate_portfolio_returns(self, compositions: Dict, price_data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate portfolio returns with event-driven rebalancing
        """
        rebal_dates = list(compositions.keys())
        all_returns = []
        all_weights = []
        
        for i, rebal_date in enumerate(rebal_dates):
            # Determine period for this composition
            start_date = rebal_date
            end_date = rebal_dates[i+1] if i+1 < len(rebal_dates) else price_data.index[-1]
            
            # Get period price data
            period_prices = price_data.loc[start_date:end_date]
            if period_prices.empty:
                continue
                
            # Get composition weights (equal weight)
            composition = compositions[rebal_date]
            n_stocks = len(composition)
            weight = 1.0 / n_stocks
            
            # Calculate returns for this period
            period_returns = period_prices.pct_change().fillna(0)
            
            # Only use stocks that are in the composition and have price data
            available_stocks = [stock for stock in composition.index if stock in period_returns.columns]
            
            if not available_stocks:
                continue
                
            # Equal weight portfolio returns
            portfolio_returns = period_returns[available_stocks].mean(axis=1) 
            
            all_returns.append(portfolio_returns)
            
            # Track weights (for analysis)
            period_weights = pd.DataFrame(0, index=period_returns.index, columns=period_returns.columns)
            for stock in available_stocks:
                period_weights[stock] = weight
            all_weights.append(period_weights)
        
        # Combine all periods
        total_returns = pd.concat(all_returns).sort_index()
        total_weights = pd.concat(all_weights).sort_index()
        
        return total_returns, total_weights
    
    def run_analysis(self, symbols: List[str], start_date: str, end_date: str, 
                    threshold: float = 0.005, top_n: int = 10):
        """Run complete analysis"""
        
        print(f"=== Figment Hybrid Analysis ===")
        print(f"Period: {start_date} to {end_date}")
        print(f"Threshold: {threshold*100:.1f}%")
        print(f"Top N stocks: {top_n}")
        print(f"Universe: {len(symbols)} symbols")
        
        # Load market cap data
        print("\n1. Loading market cap data from FMP cache...")
        market_cap_df = self.load_fmp_market_cap_data(symbols)
        
        if market_cap_df.empty:
            print("No market cap data available!")
            return None
            
        # Filter by date range
        market_cap_df = market_cap_df.loc[start_date:end_date]
        print(f"Market cap data: {len(market_cap_df)} days, {len(market_cap_df.columns)} symbols")
        
        # Detect composition changes with threshold
        print(f"\n2. Detecting composition changes with {threshold*100:.1f}% threshold...")
        compositions = self.detect_changes_with_threshold(market_cap_df, threshold, top_n)
        
        # Get all unique symbols that appear in compositions
        all_composition_symbols = set()
        for comp in compositions.values():
            all_composition_symbols.update(comp.index)
        
        print(f"\n3. Downloading price data for {len(all_composition_symbols)} symbols...")
        price_data = self.get_price_data(list(all_composition_symbols), start_date, end_date)
        
        if price_data.empty:
            print("No price data available!")
            return None
            
        # Calculate portfolio performance
        print("\n4. Calculating portfolio performance...")
        portfolio_returns, weights = self.calculate_portfolio_returns(compositions, price_data)
        
        # Calculate benchmark (QQQ)
        print("\n5. Downloading benchmark data (QQQ)...")
        benchmark_data = yf.download('QQQ', start=start_date, end=end_date, auto_adjust=True)
        if isinstance(benchmark_data, pd.DataFrame):
            benchmark_returns = benchmark_data['Close'].pct_change().fillna(0)
        else:
            benchmark_returns = benchmark_data.pct_change().fillna(0)
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        # Calculate performance metrics
        results = self.calculate_performance_metrics(portfolio_returns, benchmark_returns, compositions)
        
        # Plot results
        self.plot_results(portfolio_returns, benchmark_returns, compositions, threshold, top_n)
        
        return results
    
    def calculate_performance_metrics(self, portfolio_returns: pd.Series, 
                                    benchmark_returns: pd.Series, 
                                    compositions: Dict) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Portfolio metrics
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        portfolio_total_return = portfolio_cumulative.iloc[-1] - 1
        portfolio_cagr = (portfolio_cumulative.iloc[-1] ** (252 / len(portfolio_returns))) - 1
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        portfolio_sharpe = portfolio_cagr / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Benchmark metrics
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        benchmark_total_return = benchmark_cumulative.iloc[-1] - 1
        benchmark_cagr = (benchmark_cumulative.iloc[-1] ** (252 / len(benchmark_returns))) - 1
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        benchmark_sharpe = benchmark_cagr / benchmark_volatility if benchmark_volatility > 0 else 0
        
        # Relative metrics
        alpha = portfolio_cagr - benchmark_cagr
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0
        
        results = {
            'portfolio': {
                'total_return': portfolio_total_return,
                'cagr': portfolio_cagr,
                'volatility': portfolio_volatility,
                'sharpe_ratio': portfolio_sharpe,
                'final_value': portfolio_cumulative.iloc[-1] * 100000  # Assume $100k initial
            },
            'benchmark': {
                'total_return': benchmark_total_return,
                'cagr': benchmark_cagr,
                'volatility': benchmark_volatility,
                'sharpe_ratio': benchmark_sharpe,
                'final_value': benchmark_cumulative.iloc[-1] * 100000
            },
            'relative': {
                'alpha': alpha,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'outperformance': portfolio_total_return - benchmark_total_return
            },
            'rebalances': len(compositions),
            'period': f"{portfolio_returns.index[0].strftime('%Y-%m-%d')} to {portfolio_returns.index[-1].strftime('%Y-%m-%d')}"
        }
        
        # Print results
        print(f"\n=== PERFORMANCE RESULTS ===")
        print(f"Period: {results['period']}")
        print(f"Rebalances: {results['rebalances']}")
        print(f"\nPortfolio Performance:")
        print(f"  Total Return: {results['portfolio']['total_return']:.1%}")
        print(f"  CAGR: {results['portfolio']['cagr']:.1%}")
        print(f"  Volatility: {results['portfolio']['volatility']:.1%}")
        print(f"  Sharpe Ratio: {results['portfolio']['sharpe_ratio']:.2f}")
        print(f"  Final Value: ${results['portfolio']['final_value']:,.0f}")
        print(f"\nBenchmark (QQQ) Performance:")
        print(f"  Total Return: {results['benchmark']['total_return']:.1%}")
        print(f"  CAGR: {results['benchmark']['cagr']:.1%}")
        print(f"  Volatility: {results['benchmark']['volatility']:.1%}")
        print(f"  Sharpe Ratio: {results['benchmark']['sharpe_ratio']:.2f}")
        print(f"  Final Value: ${results['benchmark']['final_value']:,.0f}")
        print(f"\nRelative Performance:")
        print(f"  Alpha: {results['relative']['alpha']:.1%}")
        print(f"  Outperformance: {results['relative']['outperformance']:.1%}")
        print(f"  Information Ratio: {results['relative']['information_ratio']:.2f}")
        
        return results
    
    def plot_results(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, 
                    compositions: Dict, threshold: float, top_n: int):
        """Plot performance comparison"""
        
        portfolio_cumulative = (1 + portfolio_returns).cumprod() * 100000
        benchmark_cumulative = (1 + benchmark_returns).cumprod() * 100000
        
        plt.figure(figsize=(15, 10))
        
        # Main performance plot
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_cumulative.index, portfolio_cumulative.values, 
                label=f'Figment Top {top_n} ({threshold*100:.1f}% threshold)', linewidth=2, color='blue')
        plt.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                label='QQQ Benchmark', linewidth=2, color='red', alpha=0.7)
        
        # Mark rebalancing dates
        rebal_dates = list(compositions.keys())
        for date in rebal_dates[1:]:  # Skip first date
            if date in portfolio_cumulative.index:
                plt.axvline(x=date, color='gray', alpha=0.5, linestyle='--', linewidth=0.5)
        
        plt.title(f'Figment Hybrid Strategy Performance\nTop {top_n} Nasdaq Stocks with {threshold*100:.1f}% Rebalancing Threshold')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Rolling returns comparison
        plt.subplot(2, 1, 2)
        rolling_window = 60  # 60-day rolling returns
        portfolio_rolling = portfolio_returns.rolling(rolling_window).mean() * 252
        benchmark_rolling = benchmark_returns.rolling(rolling_window).mean() * 252
        
        plt.plot(portfolio_rolling.index, portfolio_rolling.values, 
                label=f'Portfolio (60d rolling CAGR)', linewidth=1.5, color='blue')
        plt.plot(benchmark_rolling.index, benchmark_rolling.values, 
                label=f'QQQ (60d rolling CAGR)', linewidth=1.5, color='red', alpha=0.7)
        
        plt.title('Rolling 60-Day Annualized Returns')
        plt.ylabel('Annualized Return')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'figment_hybrid_analysis_top{top_n}_threshold{int(threshold*1000)}.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main execution"""
    
    # Get list of symbols from FMP cache
    analyzer = FigmentHybridAnalyzer()
    cache_files = [f for f in os.listdir(analyzer.fmp_cache_dir) if f.endswith('.json')]
    symbols = [f.replace('.json', '') for f in cache_files if f != 'nasdaq_constituent.json']
    
    print(f"Found {len(symbols)} symbols in FMP cache")
    
    # Run analysis with different configurations
    configs = [
        {'threshold': 0.005, 'top_n': 6, 'name': 'Conservative (0.5% threshold, Top 6)'},
        {'threshold': 0.01, 'top_n': 6, 'name': 'Moderate (1.0% threshold, Top 6)'},
        {'threshold': 0.02, 'top_n': 6, 'name': 'Aggressive (2.0% threshold, Top 6)'},
        {'threshold': 0.005, 'top_n': 10, 'name': 'Conservative Diversified (0.5% threshold, Top 10)'},
    ]
    
    results_summary = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Running: {config['name']}")
        print(f"{'='*60}")
        
        results = analyzer.run_analysis(
            symbols=symbols,
            start_date='2019-01-01',
            end_date='2024-12-31',
            threshold=config['threshold'],
            top_n=config['top_n']
        )
        
        if results:
            results_summary[config['name']] = results
    
    # Summary comparison
    if results_summary:
        print(f"\n{'='*80}")
        print("CONFIGURATION COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        comparison_df = pd.DataFrame({
            name: {
                'CAGR': f"{results['portfolio']['cagr']:.1%}",
                'Total Return': f"{results['portfolio']['total_return']:.1%}",
                'Volatility': f"{results['portfolio']['volatility']:.1%}",
                'Sharpe Ratio': f"{results['portfolio']['sharpe_ratio']:.2f}",
                'Alpha vs QQQ': f"{results['relative']['alpha']:.1%}",
                'Rebalances': results['rebalances'],
                'Final Value': f"${results['portfolio']['final_value']:,.0f}"
            }
            for name, results in results_summary.items()
        })
        
        print(comparison_df.T)


if __name__ == "__main__":
    main()
