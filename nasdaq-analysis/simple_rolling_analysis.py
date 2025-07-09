"""
Simplified Rolling Period Analysis for Nasdaq Top-N Momentum Strategy
====================================================================

This script analyzes the performance of different portfolio sizes across multiple
time periods, starting from various historical dates up to the present day.

Creates CSV tables showing:
1. Performance by time period - which portfolio won in each period  
2. Summary table - how many times each portfolio won first place

Usage:
    python simple_rolling_analysis.py
"""

import os
import sys
import datetime as dt
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Import the main analysis components
from nasdaq_fmp_analysis_corrected import CorrectedMomentumAnalyzer

def load_config():
    """Load configuration from config.py"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", "config.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config
    except Exception as e:
        print(f"Error loading config.py: {e}")
        sys.exit(1)

def calculate_performance_metrics(price_series: pd.Series) -> Dict:
    """Calculate performance metrics from a price series"""
    if price_series.empty or len(price_series) < 2:
        return {
            'final_value': 0,
            'total_return': 0,
            'cagr': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
    
    # Calculate basic metrics
    initial_value = price_series.iloc[0]
    final_value = price_series.iloc[-1]
    total_return = (final_value / initial_value) - 1
    
    # Calculate annualized metrics
    years = len(price_series) / 252  # Approximate trading days per year
    if years > 0:
        cagr = (final_value / initial_value) ** (1/years) - 1
    else:
        cagr = 0
    
    # Calculate daily returns
    daily_returns = price_series.pct_change().dropna()
    
    # Volatility (annualized)
    volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    cumulative = price_series / price_series.cummax()
    max_drawdown = (cumulative.min() - 1) * -1
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def run_single_period_analysis(start_date: str, end_date: str, portfolio_sizes: List[int], 
                               use_shared_data: bool = True, verbose: bool = False) -> Dict:
    """Run analysis for a single time period and return results with metrics"""
    
    # Temporarily modify the global config for this run
    import nasdaq_fmp_analysis_corrected as main_module
    original_start = main_module.START_DATE
    original_end = main_module.END_DATE
    original_sizes = main_module.PORTFOLIO_SIZES
    
    # Get rolling analysis settings from config
    config = load_config()
    disable_charts = getattr(config, 'ROLLING_DISABLE_CHARTS', True)
    reuse_data = getattr(config, 'ROLLING_REUSE_DATA', True)
    
    try:
        # Set the dates for this analysis period
        main_module.START_DATE = start_date
        main_module.END_DATE = end_date
        main_module.PORTFOLIO_SIZES = portfolio_sizes
        
        # Create analyzer with shared data option
        analyzer = main_module.CorrectedMomentumAnalyzer(use_shared_data=reuse_data)
        
        # Run analysis with appropriate settings
        raw_results = analyzer.run_analysis(
            start_date, 
            end_date, 
            portfolio_sizes,
            enable_charts=not disable_charts,  # Disable charts during rolling analysis
            enable_exports=False,  # Don't export CSV for each period
            verbose=verbose
        )
        
        # Convert time series to performance metrics
        processed_results = {}
        for name, series in raw_results.items():
            metrics = calculate_performance_metrics(series)
            processed_results[name] = metrics
        
        return processed_results
        
    finally:
        # Always restore original config
        main_module.START_DATE = original_start
        main_module.END_DATE = original_end  
        main_module.PORTFOLIO_SIZES = original_sizes

def run_rolling_analysis(start_year: int, end_date: str, portfolio_sizes: List[int]) -> List[Dict]:
    """Run analysis for multiple rolling periods"""
    
    print("=" * 70)
    print("SIMPLIFIED ROLLING PERIOD ANALYSIS")
    print("=" * 70)
    print(f"Analyzing performance from {start_year} to present day")
    print(f"Portfolio sizes: {portfolio_sizes}")
    print("=" * 70)
    
    results = []
    end_year = int(end_date.split("-")[0])
    
    # Run analysis for each starting year
    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        
        print(f"\nAnalyzing period: {start_date} to {end_date}")
        print("-" * 50)
        
        try:
            # Run analysis for this period (with data reuse and no charts)
            period_results = run_single_period_analysis(
                start_date, end_date, portfolio_sizes, 
                use_shared_data=True, verbose=False
            )
            
            if period_results:
                # Find the winner
                best_return = -float('inf')
                winner = None
                
                for name, metrics in period_results.items():
                    if metrics['total_return'] > best_return:
                        best_return = metrics['total_return']
                        winner = name
                
                # Create period summary
                period_summary = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'period_years': (dt.datetime.strptime(end_date, "%Y-%m-%d") - 
                                   dt.datetime.strptime(start_date, "%Y-%m-%d")).days / 365.25,
                    'winner': winner,
                    'winner_return': best_return,
                    'results': period_results
                }
                
                results.append(period_summary)
                print(f"✅ Winner: {winner} ({best_return:.1%} return)")
                
            else:
                print(f"❌ No results for period {start_date} to {end_date}")
                
        except Exception as e:
            print(f"❌ Error analyzing {start_date} to {end_date}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

def create_tables(results: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create the two required tables from results"""
    
    # Table 1: Period-by-period results
    table1_data = []
    
    for result in results:
        row = {
            'Start_Date': result['start_date'],
            'End_Date': result['end_date'],
            'Period_Years': f"{result['period_years']:.1f}",
            'Winner': result['winner'],
            'Winner_Return_Pct': f"{result['winner_return']:.1%}",
        }
        
        # Add performance for each portfolio
        for name, metrics in result['results'].items():
            row[f'{name}_Return_Pct'] = f"{metrics['total_return']:.1%}"
            row[f'{name}_Final_Value'] = f"${metrics['final_value']:,.0f}"
        
        table1_data.append(row)
    
    table1_df = pd.DataFrame(table1_data)
    
    # Table 2: Summary of wins
    win_counts = {}
    total_profits = {}
    
    for result in results:
        winner = result['winner']
        if winner:
            win_counts[winner] = win_counts.get(winner, 0) + 1
            
            # Calculate total profit for this winner across all periods
            if winner not in total_profits:
                total_profits[winner] = 0
            
            winner_metrics = result['results'][winner]
            profit = winner_metrics['final_value'] - 100000  # Assuming $100k start
            total_profits[winner] += profit
    
    # Create summary table
    table2_data = []
    for portfolio in win_counts.keys():
        table2_data.append({
            'Portfolio': portfolio,
            'Times_Won_1st_Place': win_counts[portfolio],
            'Win_Percentage': f"{(win_counts[portfolio] / len(results)) * 100:.1f}%",
            'Total_Profit_All_Periods': f"${total_profits[portfolio]:,.0f}"
        })
    
    # Sort by wins first, then by total profit
    table2_df = pd.DataFrame(table2_data)
    table2_df = table2_df.sort_values(['Times_Won_1st_Place', 'Total_Profit_All_Periods'], 
                                     ascending=[False, False])
    
    return table1_df, table2_df

def save_tables(table1_df: pd.DataFrame, table2_df: pd.DataFrame) -> str:
    """Save both tables to a CSV file"""
    
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rolling_period_analysis_{timestamp}.csv"
    filepath = os.path.join("results", "csv", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        # Write Table 1
        f.write("TABLE 1: PERFORMANCE BY TIME PERIOD\n")
        f.write("="*50 + "\n")
        f.write("Shows which portfolio won in each time period\n\n")
        
        table1_df.to_csv(f, index=False)
        
        # Add separator
        f.write("\n\n" + "="*50 + "\n\n")
        
        # Write Table 2
        f.write("TABLE 2: SUMMARY - FIRST PLACE WINS BY PORTFOLIO\n")
        f.write("="*50 + "\n")
        f.write("Shows how many times each portfolio won first place\n")
        f.write("Sorted by: (1) Number of wins, (2) Total profit\n\n")
        
        table2_df.to_csv(f, index=False)
    
    return filepath

def main():
    """Main execution function"""
    
    # Load configuration
    config = load_config()
    
    # Check if rolling analysis is enabled
    enable_analysis = getattr(config, 'ENABLE_ROLLING_ANALYSIS', True)
    
    if not enable_analysis:
        print("Rolling period analysis is disabled in config.")
        print("Set ENABLE_ROLLING_ANALYSIS = True in config.py to enable.")
        return
    
    # Validate API key
    if not hasattr(config, 'FMP_API_KEY') or config.FMP_API_KEY == "YOUR_FMP_API_KEY_HERE":
        print("ERROR: FMP API KEY NOT SET!")
        print("Please set your FMP API key in config.py")
        return
    
    # Get configuration settings
    disable_charts = getattr(config, 'ROLLING_DISABLE_CHARTS', True)
    reuse_data = getattr(config, 'ROLLING_REUSE_DATA', True)
    start_year = int(config.START_DATE.split("-")[0])
    end_year = int(config.END_DATE.split("-")[0])
    num_periods = end_year - start_year + 1
    
    print("=" * 70)
    print("NASDAQ ROLLING PERIOD ANALYSIS")
    print("=" * 70)
    print("This analysis will test portfolio performance across multiple time periods")
    print("and identify which portfolios are most consistent winners.")
    print()
    print("ANALYSIS CONFIGURATION:")
    print(f"  • Time periods to analyze: {num_periods} periods")
    print(f"  • From {start_year} to {end_year}")
    print(f"  • Portfolio sizes: {config.PORTFOLIO_SIZES}")
    print(f"  • Chart creation: {'DISABLED' if disable_charts else 'ENABLED'} (for speed)")
    print(f"  • Data reuse: {'ENABLED' if reuse_data else 'DISABLED'} (for efficiency)")
    print()
    print("OUTPUT WILL BE CREATED:")
    print("  • CSV file with two tables:")
    print("    - Table 1: Performance by time period (which portfolio won each period)")
    print("    - Table 2: Summary of wins (how many times each portfolio won first place)")
    print(f"  • Saved to: results/csv/rolling_period_analysis_[timestamp].csv")
    print()
    
    if disable_charts:
        print("NOTE: Chart creation is disabled during rolling analysis for faster execution.")
        print("      To see charts, run the main analysis: python run_analysis.py")
        print()
    
    if reuse_data:
        print("EFFICIENCY: Data will be downloaded once and reused across all periods.")
        print("           This significantly reduces API calls and execution time.")
        print()
    
    # User confirmation
    print("=" * 70)
    response = input("Press ENTER to continue with rolling analysis, or Ctrl+C to cancel: ")
    print("=" * 70)
    
    # Extract start year from config
    start_year = int(config.START_DATE.split("-")[0])
    
    # Run rolling analysis
    results = run_rolling_analysis(
        start_year=start_year,
        end_date=config.END_DATE,
        portfolio_sizes=config.PORTFOLIO_SIZES
    )
    
    if not results:
        print("No results generated. Check your configuration and API connection.")
        return
    
    # Create tables
    table1_df, table2_df = create_tables(results)
    
    # Save results
    filepath = save_tables(table1_df, table2_df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: {filepath}")
    print(f"Analyzed {len(results)} time periods")
    
    # Display top performers
    print("\nTOP PERFORMING PORTFOLIOS:")
    print("-" * 40)
    for idx, row in table2_df.head().iterrows():
        print(f"{row['Portfolio']}: {row['Times_Won_1st_Place']} wins "
              f"({row['Win_Percentage']} win rate)")
    
    print(f"\nDetailed results available in: {filepath}")

if __name__ == "__main__":
    main()
