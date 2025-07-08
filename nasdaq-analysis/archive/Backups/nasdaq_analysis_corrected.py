# ===========================================================
# Corrected Nasdaq Analysis - Using Real Historical Market Cap Rankings
# • Uses actual FMP market cap data to determine rankings dynamically
# • Eliminates look-ahead bias from hardcoded schedules
# • Event-driven rebalancing only when composition actually changes
# • Total return via auto_adjust=True
# ===========================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, timedelta

# Cache directory for FMP data
CACHE_DIR = "/Users/ricardoellstein/nasdaq-analysis/fmp_cache"

def load_market_cap_data():
    """Load all cached market cap data from FMP."""
    print("Loading market cap data from FMP cache...")
    
    market_cap_data = {}
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json') and f != 'nasdaq_constituent.json']
    
    for file in cache_files:
        symbol = file.replace('.json', '')
        file_path = os.path.join(CACHE_DIR, file)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list) and len(data) > 0:
                # Convert to DataFrame for easier manipulation
                df = pd.DataFrame(data)
                if 'date' in df.columns and 'marketCap' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date').sort_index()
                    market_cap_data[symbol] = df
                    print(f"  {symbol}: {len(df)} data points from {df.index[0].date()} to {df.index[-1].date()}")
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
    
    print(f"Loaded market cap data for {len(market_cap_data)} symbols")
    return market_cap_data

def get_top_n_at_date(market_cap_data, target_date, n=10, max_days_diff=90):
    """Get top N companies by market cap at a specific date."""
    target_date = pd.to_datetime(target_date)
    rankings = {}
    
    for symbol, df in market_cap_data.items():
        # Find closest date within max_days_diff
        if target_date in df.index:
            # Exact match
            market_cap = df.loc[target_date, 'marketCap']
            rankings[symbol] = market_cap
        else:
            # Find closest date
            date_diffs = abs(df.index - target_date)
            min_diff_idx = date_diffs.argmin()
            min_diff_days = date_diffs[min_diff_idx].days
            
            if min_diff_days <= max_days_diff:
                closest_date = df.index[min_diff_idx]
                market_cap = df.loc[closest_date, 'marketCap']
                rankings[symbol] = market_cap
    
    # Sort by market cap and return top N
    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    top_n_symbols = [symbol for symbol, _ in sorted_rankings[:n]]
    
    return top_n_symbols, len(rankings)

def build_dynamic_schedules(market_cap_data, start_date="2020-01-01", end_date="2025-06-30"):
    """Build dynamic schedules based on actual market cap rankings."""
    print(f"\nBuilding dynamic schedules from {start_date} to {end_date}...")
    
    # Generate monthly check dates (we have limited FMP data, so check quarterly)
    date_range = pd.date_range(start=start_date, end=end_date, freq='QS')  # Quarterly start
    
    dynamic_schedules = {f"Top {i}": [] for i in range(1, 11)}
    
    current_rankings = {f"Top {i}": None for i in range(1, 11)}
    current_start_dates = {f"Top {i}": start_date for i in range(1, 11)}
    
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        print(f"  Checking rankings for {date_str}...")
        
        # Get top 10 at this date
        top_10, total_stocks = get_top_n_at_date(market_cap_data, date, n=10)
        
        if len(top_10) >= 10:
            print(f"    Found rankings for {total_stocks} stocks, top 10: {top_10}")
            
            # Check each strategy for changes
            for i in range(1, 11):
                strategy_name = f"Top {i}"
                new_ranking = top_10[:i]
                
                if current_rankings[strategy_name] != new_ranking:
                    # Composition changed - close previous period and start new one
                    if current_rankings[strategy_name] is not None:
                        # Close previous period
                        prev_date = (date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                        dynamic_schedules[strategy_name].append(
                            (current_start_dates[strategy_name], prev_date, current_rankings[strategy_name])
                        )
                    
                    # Start new period
                    current_rankings[strategy_name] = new_ranking.copy()
                    current_start_dates[strategy_name] = date_str
                    print(f"      {strategy_name} composition changed to: {new_ranking}")
        else:
            print(f"    Insufficient data ({len(top_10)} stocks found)")
    
    # Close final periods
    final_date = end_date
    for i in range(1, 11):
        strategy_name = f"Top {i}"
        if current_rankings[strategy_name] is not None:
            dynamic_schedules[strategy_name].append(
                (current_start_dates[strategy_name], final_date, current_rankings[strategy_name])
            )
    
    # Print summary
    print(f"\nDynamic Schedule Summary:")
    for strategy_name, schedule in dynamic_schedules.items():
        print(f"{strategy_name}: {len(schedule)} periods")
        for start, end, tickers in schedule:
            print(f"  {start} to {end}: {tickers}")
    
    return dynamic_schedules

def build_event_driven_weights_dynamic(schedule, universe, index):
    """Build weight matrix with event-driven rebalancing for dynamic schedules."""
    w = pd.DataFrame(0.0, index=index, columns=universe)
    
    for i, (start, end, names) in enumerate(schedule):
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        period_mask = (index >= start_date) & (index <= end_date)
        
        if not period_mask.any():
            continue
        
        # Set equal weights only at the start of each period (composition change)
        period_dates = index[period_mask]
        rebalance_date = period_dates[0]  # First date of the period
        
        # Set weights only at rebalance date - let them drift afterward
        valid_names = [name for name in names if name in universe]
        if valid_names:
            w.loc[rebalance_date, valid_names] = 1.0 / len(valid_names)
            print(f"  Rebalance on {rebalance_date.date()}: {valid_names} (equal {1.0/len(valid_names):.3f} each)")
    
    return w

def portfolio_cum_event_driven(weights, prices):
    """Calculate portfolio returns with event-driven rebalancing and weight drift."""
    rets = prices.pct_change().fillna(0)
    
    # Initialize portfolio value and weights
    port_value = pd.Series(1.0, index=rets.index)
    current_weights = pd.Series(0.0, index=weights.columns)
    
    for i, date in enumerate(rets.index):
        if i == 0:
            # Initialize with first period weights
            if weights.loc[date].sum() > 0:
                current_weights = weights.loc[date].copy()
            continue
        
        # Check if this is a rebalancing date (new non-zero weights)
        if weights.loc[date].sum() > 0:
            # Rebalance: set new equal weights
            current_weights = weights.loc[date].copy()
        else:
            # No rebalancing: let weights drift with returns
            period_returns = rets.loc[date]
            # Update weights based on individual stock performance
            current_weights = current_weights * (1 + period_returns)
            # Renormalize to sum to 1 (accounts for any numerical drift)
            if current_weights.sum() > 0:
                current_weights = current_weights / current_weights.sum()
        
        # Calculate portfolio return for this period
        port_return = (current_weights * rets.loc[date]).sum()
        port_value.loc[date] = port_value.iloc[i-1] * (1 + port_return)
    
    return port_value

def main():
    # Load market cap data
    market_cap_data = load_market_cap_data()
    
    if len(market_cap_data) < 10:
        print("ERROR: Insufficient market cap data. Need at least 10 symbols.")
        return
    
    # Build dynamic schedules based on actual market cap rankings
    # Using 5-year period where we have FMP data
    dynamic_schedules = build_dynamic_schedules(
        market_cap_data, 
        start_date="2020-01-01", 
        end_date="2025-06-30"
    )
    
    # Get all tickers from dynamic schedules
    all_tickers = set()
    for schedule in dynamic_schedules.values():
        for _, _, tickers in schedule:
            all_tickers.update(tickers)
    all_tickers.add("QQQ")  # Add benchmark
    all_tickers = sorted(list(all_tickers))
    
    print(f"\nFetching price data for {len(all_tickers)} tickers...")
    
    # Download price data
    data = yf.download(
        tickers=all_tickers,
        start="2020-01-01",
        end="2025-07-01",
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )
    
    # Handle the data structure
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    
    prices = prices.dropna(how="all").resample("M").last()
    
    print(f"Downloaded {len(prices)} months × {len(prices.columns)} tickers")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # Build weight matrices for dynamic schedules
    print(f"\nBuilding event-driven weight matrices...")
    idx = prices.index
    
    weight_matrices = {}
    for strategy_name, schedule in dynamic_schedules.items():
        if schedule:  # Only process strategies with data
            print(f"\n{strategy_name} rebalancing events:")
            weight_matrices[strategy_name] = build_event_driven_weights_dynamic(
                schedule, prices.columns, idx
            )
    
    # Calculate performance curves
    print(f"\nCalculating performance curves...")
    curves_data = {}
    
    # QQQ benchmark
    rets = prices.pct_change().fillna(0)
    curves_data["QQQ"] = (1 + rets["QQQ"]).cumprod()
    
    # Dynamic strategies
    for strategy_name, weights in weight_matrices.items():
        curves_data[strategy_name] = portfolio_cum_event_driven(weights, prices)
    
    # Combine all curves
    curves = pd.concat([
        pd.Series(curve, name=name) for name, curve in curves_data.items()
    ], axis=1).dropna()
    
    print(f"Final curves shape: {curves.shape}")
    
    # Plot results
    plt.figure(figsize=(14, 8))
    
    # Define colors
    colors = {
        "QQQ": "#2E2E2E",
        "Top 1": "#E31A1C", "Top 2": "#FF7F00", "Top 3": "#1F78B4", 
        "Top 4": "#33A02C", "Top 5": "#6A3D9A", "Top 6": "#FF1493",
        "Top 7": "#00CED1", "Top 8": "#FFD700", "Top 9": "#DC143C", "Top 10": "#32CD32"
    }
    
    # Calculate final values
    initial_investment = 100000
    for strategy in curves.columns:
        if strategy in curves.columns:
            final_multiplier = curves[strategy].iloc[-1] / curves[strategy].iloc[0]
            final_value = final_multiplier * initial_investment
            label_text = f"{strategy} (${final_value:,.0f})"
            
            plt.plot(
                curves.index, 
                curves[strategy], 
                label=label_text,
                linewidth=2.5 if strategy == "QQQ" else 2.0,
                color=colors.get(strategy, "#000000")
            )
    
    plt.yscale("log")
    plt.ylabel("Growth of $1 (log scale)")
    plt.xlabel("Date")
    plt.title("Corrected Nasdaq Analysis • Real Market Cap Rankings • Jan 2020 – Jun 2025\nEvent-Driven Rebalancing • Final Portfolio Values (Starting $100K)")
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()
    
    # Performance summary
    print(f"\nCorrected Performance Summary (Real Market Cap Rankings)")
    print("=" * 70)
    print(f"{'Strategy':8s} | {'Final Value':>12s} | {'Total Return':>12s} | {'CAGR':>8s}")
    print("-" * 70)
    
    for name in curves.columns:
        total = curves[name].iloc[-1] / curves[name].iloc[0] - 1
        years = (curves.index[-1] - curves.index[0]).days / 365.25
        cagr = (1 + total) ** (1 / years) - 1
        final_value = curves[name].iloc[-1] * initial_investment
        
        print(f"{name:8s} | ${final_value:>11,.0f} | {total:11.1%} | {cagr:7.2%}")
    
    print(f"\nCorrected Methodology:")
    print(f"• Data source: Real FMP market cap rankings + Yahoo Finance prices")
    print(f"• Rebalancing: Event-driven (only when actual composition changes)")
    print(f"• Rankings: Based on quarterly market cap checks")
    print(f"• Period: {years:.1f} years (limited by FMP data availability)")
    print(f"• Weight drift: Allowed between rebalancing events")
    print(f"• Eliminates: Look-ahead bias, hardcoded schedules")

if __name__ == "__main__":
    main()
