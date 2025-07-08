# ===========================================================
# Nasdaq Analysis - Recreating the Chart Results
# • Simple event-driven rebalancing (original approach)
# • Using GOOG instead of GOOGL for full data history
# • Minimal fixes to data issues while preserving original logic
# ===========================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# --------------------------- USER SETTINGS ----------------------------
START = "2005-07-01"
END = "2025-07-01"
INITIAL_INVESTMENT = 100_000     # Initial investment in USD
# ----------------------------------------------------------------------

# Original schedules with GOOG fix only
top2_schedule = [
    ("2005-07-01", ["MSFT", "INTC"]),
    ("2010-01-01", ["AAPL", "MSFT"]),  
    ("2024-05-01", ["NVDA", "MSFT"]),  # NVDA overtakes AAPL
]

top3_schedule = [
    ("2005-07-01", ["MSFT", "INTC", "CSCO"]),
    ("2010-01-01", ["AAPL", "MSFT", "GOOG"]),  # Using GOOG instead of GOOGL
    ("2015-07-01", ["AAPL", "MSFT", "GOOG"]),   # Keep same 
    ("2023-01-01", ["AAPL", "MSFT", "GOOG"]),   # Keep same
]

top4_schedule = [
    ("2005-07-01", ["MSFT", "INTC", "CSCO", "ORCL"]),
    ("2010-01-01", ["AAPL", "MSFT", "GOOG", "INTC"]),
    ("2015-07-01", ["AAPL", "MSFT", "AMZN", "GOOG"]),
    ("2023-01-01", ["AAPL", "MSFT", "NVDA", "GOOG"]),
]

top5_schedule = [
    ("2005-07-01", ["MSFT", "INTC", "CSCO", "ORCL", "QCOM"]),
    ("2010-01-01", ["AAPL", "MSFT", "GOOG", "INTC", "CSCO"]),
    ("2015-07-01", ["AAPL", "MSFT", "AMZN", "GOOG", "META"]),
    ("2023-01-01", ["AAPL", "MSFT", "NVDA", "GOOG", "META"]),
]

larger_schedules = {
    "Top 6": [
        ("2005-07-01", ["MSFT", "INTC", "CSCO", "ORCL", "QCOM", "ADBE"]),
        ("2010-01-01", ["AAPL", "MSFT", "GOOG", "INTC", "CSCO", "ORCL"]),
        ("2015-07-01", ["AAPL", "MSFT", "AMZN", "GOOG", "META", "INTC"]),
        ("2023-01-01", ["AAPL", "MSFT", "NVDA", "GOOG", "META", "AMZN"]),
    ],
    "Top 8": [
        ("2005-07-01", ["MSFT", "INTC", "CSCO", "ORCL", "QCOM", "ADBE", "AMAT", "TXN"]),
        ("2010-01-01", ["AAPL", "MSFT", "GOOG", "INTC", "CSCO", "ORCL", "QCOM", "ADBE"]),
        ("2015-07-01", ["AAPL", "MSFT", "AMZN", "GOOG", "META", "INTC", "CSCO", "ADBE"]),
        ("2023-01-01", ["AAPL", "MSFT", "NVDA", "GOOG", "META", "AMZN", "ADBE", "AVGO"]),
    ],
    "Top 10": [
        ("2005-07-01", ["MSFT", "INTC", "CSCO", "ORCL", "QCOM", "ADBE", "AMAT", "TXN", "INTU", "GILD"]),
        ("2010-01-01", ["AAPL", "MSFT", "GOOG", "INTC", "CSCO", "ORCL", "QCOM", "ADBE", "INTU", "GILD"]),
        ("2015-07-01", ["AAPL", "MSFT", "AMZN", "GOOG", "META", "INTC", "CSCO", "ADBE", "NFLX", "INTU"]),
        ("2023-01-01", ["AAPL", "MSFT", "NVDA", "GOOG", "META", "AMZN", "ADBE", "AVGO", "CSCO", "INTU"]),
    ],
}

def build_weights(schedule, universe, index):
    """Build weight matrix from an event-driven schedule - ORIGINAL approach."""
    w = pd.DataFrame(0.0, index=index, columns=universe)
    
    for i, (start, names) in enumerate(schedule):
        start_date = pd.to_datetime(start)
        
        # Find actual trading date on or after start_date
        valid_dates = index[index >= start_date]
        if valid_dates.empty:
            continue
        actual_start = valid_dates[0]
        
        # Determine end date (next rebalance date or end of period)
        if i < len(schedule) - 1:
            end_date = pd.to_datetime(schedule[i + 1][0])
            valid_end_dates = index[index >= end_date]
            if not valid_end_dates.empty:
                actual_end = valid_end_dates[0]
            else:
                actual_end = index[-1]
        else:
            actual_end = index[-1]
        
        # Create mask for this period (inclusive start, exclusive end)
        mask = (index >= actual_start) & (index < actual_end)
        
        # Only assign weights to tickers that exist in the universe
        available_names = [name for name in names if name in universe]
        
        if available_names:
            equal_weight = 1.0 / len(available_names)
            w.loc[mask, available_names] = equal_weight
    
    return w

def calculate_performance(prices_df):
    """Calculate performance for all portfolio strategies - ORIGINAL approach."""
    idx = prices_df.index
    rets = prices_df.pct_change().fillna(0)
    
    def portfolio_cum(weights):
        port_rets = (weights * rets).sum(axis=1)
        return (1 + port_rets).cumprod()
    
    # Calculate curves for all portfolio sizes
    curves_dict = {
        "QQQ": (1 + rets["QQQ"]).cumprod(),
        "Top 2": portfolio_cum(build_weights(top2_schedule, prices_df.columns, idx)),
        "Top 3": portfolio_cum(build_weights(top3_schedule, prices_df.columns, idx)),
        "Top 4": portfolio_cum(build_weights(top4_schedule, prices_df.columns, idx)),
        "Top 5": portfolio_cum(build_weights(top5_schedule, prices_df.columns, idx))
    }
    
    # Add larger portfolios
    for size, schedule in larger_schedules.items():
        curves_dict[size] = portfolio_cum(build_weights(schedule, prices_df.columns, idx))
    
    return pd.concat(curves_dict, axis=1).dropna()

def perf_table(df: pd.DataFrame, label: str):
    """Print performance summary table."""
    print(f"\n{label} Performance")
    print("=" * 80)
    print(f"{'Strategy':15s} | {'Final Value':>15s} | {'Total Return':>12s} | {'CAGR':>8s}")
    print("-" * 80)
    for col in df.columns:
        final_value = df[col].iloc[-1] * INITIAL_INVESTMENT
        total = df[col].iloc[-1] / df[col].iloc[0] - 1
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (1 + total) ** (1 / yrs) - 1
        print(f"{col:15s} | ${final_value:>14,.0f} | {total:11.1%} | {cagr:7.2%}")

def main():
    print("Nasdaq Analysis - Chart Matching Version")
    print("=" * 50)
    print("Using original approach with GOOG fix only...")
    
    # Get all tickers needed - using GOOG instead of GOOGL
    all_tickers = sorted(
        {t for _, lst in top2_schedule + top3_schedule + top4_schedule + top5_schedule for t in lst} |
        {t for schedule in larger_schedules.values() for _, lst in schedule for t in lst} |
        {"QQQ"}
    )
    
    print(f"Tickers: {all_tickers}")
    
    # Fetch daily data 
    prices_daily = (
        yf.download(
            tickers=all_tickers,
            start=START,
            end=END,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        ["Close"]
        .dropna(how="all")
    )
    
    # Handle MultiIndex if present
    if isinstance(prices_daily.columns, pd.MultiIndex):
        prices_daily.columns = prices_daily.columns.droplevel(0)
    
    # Basic data cleanup - forward fill missing data
    print("Basic data cleanup...")
    for t in prices_daily.columns:
        first = prices_daily[t].first_valid_index()
        if first is not None:
            prices_daily.loc[first:, t] = prices_daily.loc[first:, t].ffill()
    
    # Drop tickers with severe issues (>90% missing)
    missing_pct = prices_daily.isna().mean()
    severe_problems = missing_pct[missing_pct > 0.9]
    if len(severe_problems) > 0:
        prices_daily = prices_daily.drop(columns=severe_problems.index)
        print(f"Dropped severely problematic tickers: {severe_problems.index.tolist()}")
    
    # Check what we have
    print(f"\nData: {len(prices_daily)} days × {len(prices_daily.columns)} tickers")
    print("Key tickers check:")
    for t in ["AAPL", "MSFT", "GOOG", "META", "NVDA", "QQQ"]:
        if t in prices_daily.columns:
            first_date = prices_daily[t].first_valid_index()
            print(f"  {t:6s}: {first_date.date() if first_date else 'MISSING'}")
    
    # Calculate performance 
    print("\nCalculating performance...")
    curves_daily = calculate_performance(prices_daily)
    
    # Create chart matching the original
    colors = {
        "QQQ": "#000000",      # Black
        "Top 2": "#FF0000",    # Red
        "Top 3": "#0000FF",    # Blue
        "Top 4": "#00CC00",    # Green
        "Top 5": "#FF6600",    # Orange
        "Top 6": "#9933CC",    # Purple
        "Top 8": "#FF66B2",    # Pink
        "Top 10": "#00CCCC",   # Cyan
    }
    
    plt.figure(figsize=(16, 10))
    for name in curves_daily.columns:
        lw = 3.0 if name == "QQQ" else 2.0
        line = '-' if name == "QQQ" or int(name.split()[1]) <= 3 else '--'
        plt.plot(curves_daily.index, curves_daily[name] * INITIAL_INVESTMENT,
                 label=f"{name} (${curves_daily[name].iloc[-1] * INITIAL_INVESTMENT:,.0f})",
                 linewidth=lw, linestyle=line,
                 color=colors.get(name, None))

    plt.grid(True, linestyle="--", linewidth=0.4)
    plt.title(f"Portfolio Values • {START} – {END}\nChart Matching Version • Initial Investment: ${INITIAL_INVESTMENT:,}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD)")

    # Format y-axis
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    
    plt.show()

    # Performance summary
    perf_table(curves_daily, "Daily")

if __name__ == "__main__":
    main()
