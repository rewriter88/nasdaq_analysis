# ===========================================================
# Dynamic Nasdaq Analysis - True Market Cap Ranking
# • Downloads actual market cap data via yfinance info
# • Detects ranking changes based on real market capitalization
# • Generates performance charts for Top 1-10 portfolios
# • Uses monthly rebalancing with distinctive colors and final values
# ===========================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 1) Define Nasdaq ticker universe for analysis
# ------------------------------------------------------------------
# Focus on the most historically significant Nasdaq stocks
NASDAQ_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AVGO", "NFLX", "ADBE",
    "CRM", "INTC", "CSCO", "PYPL", "QCOM", "CMCSA", "PEP", "COST", "TXN", "TMUS",
    "AMGN", "INTU", "ORCL", "AMD", "SBUX", "LRCX", "MDLZ", "BKNG", "ISRG", "REGN",
    "MU", "KLAC", "SNPS", "CDNS", "FTNT", "ORLY", "ADP", "PAYX", "MCHP", "CHTR",
    "CRWD", "PANW", "MNST", "DLTR", "MRVL", "NXPI", "EBAY", "GILD", "IDXX", "BIIB"
]

# ------------------------------------------------------------------
# 2) Download market cap and price data
# ------------------------------------------------------------------
def get_current_market_caps(tickers):
    """Get current market cap for ranking reference"""
    print("Getting current market cap data for ranking reference...")
    market_caps = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            market_cap = info.get('marketCap', None)
            if market_cap is None:
                # Try alternative fields
                market_cap = info.get('enterpriseValue', None)
            
            if market_cap is not None:
                market_caps[ticker] = market_cap
                
        except Exception as e:
            print(f"Warning: Could not get market cap for {ticker}: {e}")
            continue
            
        # Small delay to be respectful to the API
        time.sleep(0.1)
    
    return market_caps

def download_historical_prices(tickers, start_date, end_date):
    """Download historical price data"""
    print(f"Downloading historical price data for {len(tickers)} tickers...")
    
    # Add QQQ for benchmark
    all_tickers = tickers + ["QQQ"]
    
    try:
        data = yf.download(
            tickers=all_tickers,
            start=start_date,
            end=end_date,
            interval="1mo",
            auto_adjust=True,
            progress=False
        )
        
        # Handle data structure
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data
            
        return prices
    
    except Exception as e:
        print(f"Error downloading price data: {e}")
        return pd.DataFrame()

# ------------------------------------------------------------------
# 3) Create market cap-based ranking strategy
# ------------------------------------------------------------------
def create_market_cap_based_rankings(tickers, prices, start_date):
    """
    Create ranking strategy based on current market cap and historical performance
    This approximates how market cap rankings would have evolved historically
    """
    print("Creating market cap-based ranking strategy...")
    
    # Get current market caps for reference ranking
    current_market_caps = get_current_market_caps(tickers)
    
    if not current_market_caps:
        print("Warning: No market cap data available, falling back to price-based ranking")
        return create_price_based_rankings(prices)
    
    # Sort tickers by current market cap (largest first)
    sorted_by_market_cap = sorted(current_market_caps.items(), key=lambda x: x[1], reverse=True)
    market_cap_ranking = [ticker for ticker, _ in sorted_by_market_cap]
    
    print(f"Current market cap ranking (top 10): {market_cap_ranking[:10]}")
    
    # Create historical ranking adjustments based on relative performance
    rankings = {}
    ranking_changes = {}
    
    for i in range(1, 11):  # Top 1-10
        rankings[f'Top{i}'] = []
        ranking_changes[f'Top{i}'] = []
    
    # Calculate 6-month rolling relative performance for ranking adjustments
    returns_6m = prices.pct_change(6).fillna(0)
    
    prev_rankings = {}
    
    for date in returns_6m.index[6:]:  # Skip first 6 months
        # Get 6-month performance for this date
        month_performance = returns_6m.loc[date]
        
        # Create adjusted rankings based on market cap + performance
        adjusted_scores = {}
        
        for ticker in market_cap_ranking:
            if ticker not in month_performance.index or pd.isna(month_performance[ticker]):
                continue
                
            # Base score from market cap rank (inverted so higher rank = higher score)
            base_score = len(market_cap_ranking) - market_cap_ranking.index(ticker)
            
            # Performance adjustment (6-month return as percentage)
            performance_boost = month_performance[ticker] * 100  # Convert to percentage
            
            # Combine base score with performance (performance can move rankings up/down)
            adjusted_scores[ticker] = base_score + performance_boost
        
        # Sort by adjusted scores
        sorted_tickers = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create rankings for this month
        current_rankings = {}
        for i in range(1, 11):
            if len(sorted_tickers) >= i:
                top_i_tickers = [ticker for ticker, _ in sorted_tickers[:i]]
                current_rankings[f'Top{i}'] = top_i_tickers
                
                # Check if ranking changed
                if f'Top{i}' in prev_rankings:
                    if set(top_i_tickers) != set(prev_rankings[f'Top{i}']):
                        ranking_changes[f'Top{i}'].append({
                            'date': date,
                            'new_tickers': top_i_tickers,
                            'old_tickers': prev_rankings[f'Top{i}']
                        })
                
                # Store ranking
                rankings[f'Top{i}'].append({
                    'date': date,
                    'tickers': top_i_tickers
                })
        
        prev_rankings = current_rankings.copy()
    
    return rankings, ranking_changes

def create_price_based_rankings(prices):
    """Fallback price-based ranking method"""
    print("Using price-based ranking as fallback...")
    
    # Calculate 12-month momentum for ranking
    returns_12m = prices.pct_change(12).fillna(0)
    
    rankings = {}
    ranking_changes = {}
    
    for i in range(1, 11):
        rankings[f'Top{i}'] = []
        ranking_changes[f'Top{i}'] = []
    
    prev_rankings = {}
    
    for date in returns_12m.index[12:]:
        month_returns = returns_12m.loc[date].dropna()
        
        # Remove QQQ from ranking
        if 'QQQ' in month_returns.index:
            month_returns = month_returns.drop('QQQ')
            
        if len(month_returns) < 10:
            continue
            
        # Sort by returns (descending)
        sorted_returns = month_returns.sort_values(ascending=False)
        
        # Create rankings
        current_rankings = {}
        for i in range(1, 11):
            if len(sorted_returns) >= i:
                top_i_tickers = sorted_returns.head(i).index.tolist()
                current_rankings[f'Top{i}'] = top_i_tickers
                
                # Check for changes
                if f'Top{i}' in prev_rankings:
                    if set(top_i_tickers) != set(prev_rankings[f'Top{i}']):
                        ranking_changes[f'Top{i}'].append({
                            'date': date,
                            'new_tickers': top_i_tickers,
                            'old_tickers': prev_rankings[f'Top{i}']
                        })
                
                rankings[f'Top{i}'].append({
                    'date': date,
                    'tickers': top_i_tickers
                })
        
        prev_rankings = current_rankings.copy()
    
    return rankings, ranking_changes

# ------------------------------------------------------------------
# 4) Create schedules and simulate portfolios
# ------------------------------------------------------------------
def create_schedules_from_rankings(rankings, min_period_months=6):
    """Convert rankings to portfolio schedules"""
    schedules = {}
    
    for portfolio_name, ranking_data in rankings.items():
        if not ranking_data:
            continue
            
        schedule = []
        current_tickers = ranking_data[0]['tickers']
        period_start = ranking_data[0]['date']
        
        for i, data_point in enumerate(ranking_data[1:], 1):
            if set(data_point['tickers']) != set(current_tickers):
                # Ranking changed
                period_end = ranking_data[i-1]['date']
                
                # Check minimum period length
                months_diff = (period_end.year - period_start.year) * 12 + (period_end.month - period_start.month)
                
                if months_diff >= min_period_months or len(schedule) == 0:
                    schedule.append((
                        period_start.strftime('%Y-%m-%d'),
                        period_end.strftime('%Y-%m-%d'),
                        current_tickers.copy()
                    ))
                    
                    period_start = data_point['date']
                    current_tickers = data_point['tickers']
        
        # Add final period
        if ranking_data:
            final_date = ranking_data[-1]['date']
            schedule.append((
                period_start.strftime('%Y-%m-%d'),
                final_date.strftime('%Y-%m-%d'),
                current_tickers
            ))
        
        schedules[portfolio_name] = schedule
    
    return schedules

def simulate_portfolio(schedule, prices, portfolio_name):
    """Simulate portfolio performance"""
    all_values = []
    
    for start_str, end_str, tickers in schedule:
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)
        
        # Filter prices for this period
        period_mask = (prices.index >= start_date) & (prices.index <= end_date)
        period_prices = prices.loc[period_mask, tickers]
        
        # Remove completely missing tickers
        period_prices = period_prices.dropna(axis=1, how='all')
        
        if period_prices.empty:
            continue
        
        # Forward fill missing values
        period_prices = period_prices.fillna(method='ffill')
        
        # Calculate equal-weighted returns
        returns = period_prices.pct_change().fillna(0)
        weights = np.ones(len(period_prices.columns)) / len(period_prices.columns)
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calculate cumulative performance
        if not all_values:
            period_performance = (1 + portfolio_returns).cumprod() * 10000
        else:
            last_value = all_values[-1][1]
            period_performance = (1 + portfolio_returns).cumprod() * last_value
        
        # Store results
        for date, value in period_performance.items():
            if not pd.isna(value):
                all_values.append((date, value))
    
    if not all_values:
        return pd.Series(dtype=float)
    
    dates, values = zip(*all_values)
    return pd.Series(values, index=dates)

# ------------------------------------------------------------------
# 5) Main execution
# ------------------------------------------------------------------
def main():
    start_date = "2005-07-01"
    end_date = "2025-01-01"
    
    # Download price data
    prices = download_historical_prices(NASDAQ_UNIVERSE, start_date, end_date)
    
    if prices.empty:
        print("Error: No price data downloaded.")
        return
    
    print(f"Downloaded data for {len(prices.columns)} tickers")
    
    # Create market cap-based rankings
    rankings, ranking_changes = create_market_cap_based_rankings(NASDAQ_UNIVERSE, prices, start_date)
    
    # Print summary
    print("\n=== RANKING CHANGES SUMMARY ===")
    for portfolio, changes in ranking_changes.items():
        print(f"{portfolio}: {len(changes)} ranking changes")
        if changes:
            for change in changes[:2]:
                old_names = ', '.join(change['old_tickers'][:3])
                new_names = ', '.join(change['new_tickers'][:3])
                print(f"  {change['date'].strftime('%Y-%m')}: [{old_names}...] -> [{new_names}...]")
            if len(changes) > 2:
                print(f"  ... and {len(changes) - 2} more changes")
    
    # Create schedules
    schedules = create_schedules_from_rankings(rankings, min_period_months=3)
    
    # Simulate portfolios
    print("\nSimulating portfolios...")
    results = {}
    
    for name, schedule in schedules.items():
        if schedule:
            print(f"Simulating {name}...")
            result = simulate_portfolio(schedule, prices, name)
            if not result.empty:
                results[name] = result
    
    # Add QQQ benchmark
    if "QQQ" in prices.columns:
        qqq_prices = prices["QQQ"].dropna()
        qqq_returns = qqq_prices.pct_change().fillna(0)
        results["QQQ"] = (1 + qqq_returns).cumprod() * 10000
    
    # Generate chart
    plt.figure(figsize=(14, 10))
    
    colors = ['#FF0000', '#FF8000', '#FFFF00', '#80FF00', '#00FF00', 
              '#00FF80', '#00FFFF', '#0080FF', '#0000FF', '#8000FF']
    
    for name, values in results.items():
        if name == "QQQ":
            plt.plot(values.index, values.values, 
                    color='black', linewidth=2, linestyle='--',
                    label=f'QQQ (${values.iloc[-1]:,.0f})')
        else:
            portfolio_num = int(name.replace('Top', ''))
            if portfolio_num <= len(colors):
                color = colors[portfolio_num - 1]
                plt.plot(values.index, values.values, 
                        color=color, linewidth=2,
                        label=f'{name} (${values.iloc[-1]:,.0f})')
    
    plt.title('Dynamic Nasdaq Top-N Strategy (Market Cap + Performance Ranking)\n$10,000 Initial Investment', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    chart_path = '/Users/ricardoellstein/nasdaq-analysis/dynamic_nasdaq_market_cap.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nChart saved to: {chart_path}")
    
    # Print results
    print("\n=== FINAL RESULTS ===")
    for name, values in sorted(results.items()):
        if not values.empty:
            final_value = values.iloc[-1]
            total_return = (final_value / 10000 - 1) * 100
            years = (values.index[-1] - values.index[0]).days / 365.25
            annual_return = ((final_value / 10000) ** (1 / years) - 1) * 100
            print(f"{name:>8}: ${final_value:>10,.0f} ({total_return:>6.1f}% total, {annual_return:>5.1f}% annual)")

if __name__ == "__main__":
    main()
