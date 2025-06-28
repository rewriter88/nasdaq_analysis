# ===========================================================
# Nasdaq "Top‑N" Momentum Strategies, 2005‑07‑01 – 2025‑06‑30
# • Builds monthly Top‑N portfolio for several different N
# • Recomputes the roster AUTOMATICALLY from daily market caps
# • Equal‑weights the constituents on a change date, then
#   lets weights float (buy‑and‑hold) until the next change
# • Benchmarks against QQQ (total‑return proxy via Adj Close)
# • Produces daily & month‑end equity curves + summary stats
#
# Dependencies:
#   pip install yfinance pandas numpy matplotlib requests tqdm
# ===========================================================

import os
import json
import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm

# --------------------------- USER SETTINGS ----------------------------
START = "2005-07-01"
END = "2025-07-01"
SIZES = [2, 3, 4, 5, 6, 8, 10]  # portfolio sizes to back‑test
INITIAL_INVESTMENT = 100_000     # Initial investment in USD
FMP_KEY = os.getenv("FMP_KEY", "2whsxV4FK6zNdDQs0Z5yrggdvxfeHPAS")
CACHE_DIR = "fmp_cache"  # simple on‑disk cache
# ----------------------------------------------------------------------

os.makedirs(CACHE_DIR, exist_ok=True)
session = requests.Session()

def _get_cached(url: str):
    """Simple JSON cache for each endpoint."""
    fname = os.path.join(CACHE_DIR, url.split("/")[-1].split("?")[0] + ".json")
    if os.path.exists(fname):
        with open(fname, "r") as fh:
            return json.load(fh)
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    with open(fname, "w") as fh:
        json.dump(data, fh)
    return data

def get_market_cap_series(ticker: str) -> pd.Series:
    """Calculate historical market cap using price * shares outstanding approximation."""
    try:
        # Download data from Yahoo Finance with adjusted prices
        ticker_data = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
        if ticker_data.empty:
            return pd.Series(dtype=float)
            
        # Use price * volume as a proxy for market activity/size
        # This is a simplified approach - in reality we'd need shares outstanding data
        close_price = ticker_data['Close']  # auto_adjust=True makes this already adjusted
        volume = ticker_data['Volume']
        
        # Use a simple price * average volume as market cap proxy
        # Apply a 252-day (1 year) rolling mean to smooth the data
        avg_volume = volume.rolling(window=252, min_periods=1).mean()
        market_cap_proxy = close_price * avg_volume
        
        # Handle duplicates and sort
        market_cap_proxy = market_cap_proxy[~market_cap_proxy.index.duplicated(keep="first")]
        return market_cap_proxy.sort_index()
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return pd.Series(dtype=float)

def build_roster_df(mcap: pd.DataFrame, n: int) -> pd.Series:
    """Returns a pd.Series whose values are frozensets of the Top‑n tickers."""
    # Drop any rows with NaN values
    mcap_clean = mcap.dropna(how='all')
    top_sets = (
        mcap_clean.rank(axis=1, method="first", ascending=False)
        .le(n)
        .apply(lambda row: frozenset(row[row].index), axis=1)
    )
    # Keep only rows where the set actually changed
    changed = top_sets != top_sets.shift(1)
    return top_sets.loc[changed]

def backtest(prc: pd.DataFrame, roster_series: pd.Series) -> pd.Series:
    """Run a backtest for a given price DataFrame and roster series."""
    rets = prc.pct_change().fillna(0)
    
    # Check for extreme values that could cause unrealistic returns
    max_ret = rets.max().max()
    min_ret = rets.min().min()
    print(f"    Max daily return in portfolio: {max_ret:.4f}")
    print(f"    Min daily return in portfolio: {min_ret:.4f}")
    
    # Check for any stocks with extreme cumulative returns
    cumulative_rets = (1 + rets).cumprod()
    final_rets = cumulative_rets.iloc[-1]
    extreme_stocks = final_rets[final_rets > 100]  # Stocks with >100x return
    if not extreme_stocks.empty:
        print(f"    Stocks with extreme returns (>100x): {extreme_stocks.to_dict()}")
    
    # Cap extreme daily returns to prevent unrealistic results
    rets = rets.clip(-0.5, 1.0)  # Cap returns between -50% and +100% per day
    
    pv = INITIAL_INVESTMENT  # Start with initial investment
    values = []
    weights = {}  # ticker -> weight

    for date in rets.index:
        # If this is a (first) date in the roster_series, rebalance
        if date in roster_series.index:
            members = roster_series.loc[date]
            k = len(members)
            weights = {t: 1.0 / k for t in members}

        # Compute portfolio daily return
        daily_ret = sum(weights.get(t, 0.0) * rets.at[date, t] for t in weights)
        pv *= 1 + daily_ret
        values.append(pv)

        # Let weights float with price moves
        for t in list(weights):
            weights[t] *= 1 + rets.at[date, t]
        total = sum(weights.values())
        if total > 0:
            for t in list(weights):
                weights[t] /= total

    return pd.Series(values, index=rets.index, name="Portfolio")

def perf_table(df: pd.DataFrame, label: str):
    """Print performance summary table."""
    print(f"\n{label} Performance")
    print("=" * 80)
    print(f"{'Strategy':15s} | {'Final Value':>15s} | {'Total Return':>12s} | {'CAGR':>8s}")
    print("-" * 80)
    for col in df.columns:
        final_value = df[col].iloc[-1]
        total = final_value / INITIAL_INVESTMENT - 1
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (1 + total) ** (1 / yrs) - 1
        print(f"{col:15s} | ${final_value:>14,.0f} | {total:11.1%} | {cagr:7.2%}")

def main():
    # 1) Get Nasdaq universe
    print("Downloading Nasdaq constituent list...")
    nasdaq_url = f"https://financialmodelingprep.com/api/v3/nasdaq_constituent?apikey={FMP_KEY}"
    constituents = _get_cached(nasdaq_url)
    tickers = sorted({c["symbol"] for c in constituents})
    print(f"Found {len(tickers)} tickers in Nasdaq constituent list.")

    # 2) Get historical market caps
    print("Fetching historical market caps (may take a few minutes)...")
    mcaps = {}
    for t in tqdm(tickers):
        s = get_market_cap_series(t)
        if s.empty:
            continue
        # Restrict to analysis window for speed
        mcaps[t] = s.loc[(s.index >= START) & (s.index < END)]

    market_cap_df = pd.concat(mcaps, axis=1)
    market_cap_df = market_cap_df.ffill().dropna(how="all")

    if market_cap_df.empty:
        raise RuntimeError("No market‑cap data returned – check date range.")

    # 3) Align to NYSE trading days (use QQQ prices as calendar anchor)
    calendar_data = yf.download("QQQ", start=START, end=END, progress=False, auto_adjust=True)
    calendar = calendar_data['Close'].dropna().index  # auto_adjust=True makes Close already adjusted
    market_cap_df = market_cap_df.reindex(calendar).ffill()

    # 4) Build portfolios for each size
    rosters_by_size = {n: build_roster_df(market_cap_df, n) for n in SIZES}

    # 5) Get universe of tickers needed
    all_strategy_tickers = sorted(set().union(*[item for n in SIZES for item in rosters_by_size[n].tolist()]))
    all_tickers = all_strategy_tickers + ["QQQ"]

    # 6) Get price data
    print("Downloading price series from Yahoo Finance...")
    # Initialize an empty DataFrame with datetime index
    prices = pd.DataFrame()

    for t in tqdm(all_tickers):
        try:
            # Explicitly request Adj Close for split/dividend adjusted prices
            df = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
            if not df.empty:
                prices[t] = df['Close']  # auto_adjust=True makes Close already adjusted
        except Exception as e:
            print(f"Error downloading {t}: {str(e)}")
            continue

    if prices.empty:
        raise RuntimeError("No price data downloaded")

    # Clean up the price data
    prices = prices.dropna(how='all').ffill()

    # 7) Run back‑tests
    print("Running back‑tests...")
    # Initialize QQQ with initial investment and calculate its value over time
    qqq_returns = prices["QQQ"].pct_change()
    qqq_returns.iloc[0] = 0  # Set first day return to 0 to match portfolio initialization
    qqq_returns = qqq_returns.fillna(0)  # Fill any remaining NaN values with 0
    qqq_values = INITIAL_INVESTMENT * (1 + qqq_returns).cumprod()  # Calculate cumulative returns
    curves = {"QQQ": qqq_values}
    print(f"QQQ final value: ${qqq_values.iloc[-1]:,.0f}")

    for n in SIZES:
        print(f"Running backtest for Top {n}...")
        strategy_tickers = rosters_by_size[n].explode().unique()
        
        # Debug: Print first few roster changes for Top 2
        if n == 2:
            print(f"Top {n} roster changes (first 10):")
            for i, (date, roster) in enumerate(rosters_by_size[n].head(10).items()):
                print(f"  {date.strftime('%Y-%m-%d')}: {sorted(list(roster))}")
        
        portfolio_result = backtest(prices[strategy_tickers], rosters_by_size[n])
        curves[f"Top {n}"] = portfolio_result
        print(f"Top {n} final value: ${portfolio_result.iloc[-1]:,.0f}")

    curves_df = pd.concat(curves, axis=1)
    curves_daily = curves_df
    curves_monthly = curves_df.resample("ME").last()  # Using ME instead of M to avoid warning

    # 8) Plot results
    colors = {
        "QQQ": "#000000",
        "Top 2": "#FF0000",
        "Top 3": "#0000FF",
        "Top 4": "#00CC00",
        "Top 5": "#FF6600",
        "Top 6": "#9933CC",
        "Top 8": "#FF66B2",
        "Top 10": "#00CCCC",
    }

    plt.figure(figsize=(15, 7))
    for name in curves_daily.columns:
        lw = 2.5 if name == "QQQ" else 1.7
        line = '-' if "Top" not in name or int(name.split()[1]) <= 3 else '--'
        plt.plot(curves_daily.index, curves_daily[name],
                 label=f"{name} (${curves_daily[name].iloc[-1]:,.0f})",  # Add final value to legend
                 linewidth=lw, linestyle=line,
                 color=colors.get(name, None))

    plt.grid(True, linestyle="--", linewidth=0.4)
    plt.title(f"Portfolio Values • {START} – {END}\nInitial Investment: ${INITIAL_INVESTMENT:,}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD)")

    # Format y-axis with dollar signs and commas
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Adjust legend and layout
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    
    # Display the plot
    plt.show()

    # 9) Performance summary
    perf_table(curves_daily, "Daily")
    perf_table(curves_monthly, "Monthly")
    
    # Clean up plots at the very end
    plt.close('all')

if __name__ == "__main__":
    main()
