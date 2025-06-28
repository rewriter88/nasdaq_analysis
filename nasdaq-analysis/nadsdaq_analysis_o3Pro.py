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
FMP_KEY = os.getenv("FMP_KEY", "2whsxV4FK6zNdDQs0Z5yrggdvxfeHPAS")
CACHE_DIR = "fmp_cache"  # simple on‑disk cache
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 3) Price data (total return via Adj Close) ---------------------------
# ----------------------------------------------------------------------
print("Downloading price series from Yahoo Finance...")
# Download tickers one by one to avoid any string handling issues
prices_dict = {}
for t in tqdm(all_tickers):
    try:
        df = yf.download(t, start=START, end=END, progress=False)
        if not df.empty:
            prices_dict[t] = df["Close"]
    except Exception as e:
        print(f"Error downloading {t}: {str(e)}")
        continue
prices = pd.DataFrame(prices_dict).dropna(how="all")
prices = prices.ffill()

# Union of all tickers ever needed, plus QQQ
all_strategy_tickers = sorted(set().union(*[item for n in SIZES for item in rosters_by_size[n].tolist()]))
all_tickers = all_strategy_tickers + ["QQQ"]

# ----------------------------------------------------------------------
# 3) Price data (total return via Adj Close) ---------------------------
# ----------------------------------------------------------------------
print("Downloading price series from Yahoo Finance...")
all_strategy_tickers = sorted(
    set().union(*[set(item) for n in SIZES for item in rosters_by_size[n].tolist()])
)
all_tickers = all_strategy_tickers + ["QQQ"]

# ----------------------------------------------------------------------
# 3) Price data (total return via Adj Close) ---------------------------
# ----------------------------------------------------------------------
print("Downloading price series from Yahoo Finance...")
# Download tickers one by one to avoid any string handling issues
prices_dict = {}
for t in tqdm(all_tickers):
    try:
        df = yf.download(t, start=START, end=END, progress=False)
        if not df.empty:
            prices_dict[t] = df["Close"]
    except Exception as e:
        print(f"Error downloading {t}: {str(e)}")
        continue
prices = pd.DataFrame(prices_dict).dropna(how="all")the roster AUTOMATICALLY from daily market caps
# • Equal‑weights the constituents on a change date, then
#   lets weights float (buy‑and‑hold) until the next change
# • Benchmarks against QQQ (total‑return proxy via Adj Close)
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
START     = "2005-07-01"
END       = "2025-07-01"
SIZES     = [2, 3, 4, 5, 6, 8, 10]          # portfolio sizes to back‑test
FMP_KEY   = os.getenv("FMP_KEY", "2whsxV4FK6zNdDQs0Z5yrggdvxfeHPAS")
CACHE_DIR = "fmp_cache"                     # simple on‑disk cache
# ----------------------------------------------------------------------

os.makedirs(CACHE_DIR, exist_ok=True)
session = requests.Session()

# ----------------------------------------------------------------------
# Helper #1 : Simple JSON cache for each endpoint ----------------------
# ----------------------------------------------------------------------
def _get_cached(url: str):
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

# ----------------------------------------------------------------------
# 1) Retrieve Nasdaq universe and historical market caps ---------------
# ----------------------------------------------------------------------
print("Downloading Nasdaq constituent list...")
nasdaq_url = f"https://financialmodelingprep.com/api/v3/nasdaq_constituent?apikey={FMP_KEY}"
constituents = _get_cached(nasdaq_url)
tickers = sorted({c["symbol"] for c in constituents})

print(f"Found {len(tickers)} tickers in Nasdaq constituent list.")

def get_market_cap_series(ticker: str) -> pd.Series:
    """Calculate historical market cap using normalized volume-weighted market value from Yahoo Finance."""
    try:
        # Download data from Yahoo Finance
        ticker_data = yf.download(ticker, start=START, end=END, progress=False)
        if ticker_data.empty:
            return pd.Series(dtype=float)
            
        # Calculate normalized volume-weighted market value as a proxy for market cap
        # We use a 21-day rolling mean of volume to smooth out volume spikes
        volume_smoothed = ticker_data['Volume'].rolling(window=21, min_periods=1).mean()
        market_cap = ticker_data['Close'] * ticker_data['Volume'] / volume_smoothed
        
        # Handle duplicates and sort
        market_cap = market_cap[~market_cap.index.duplicated(keep="first")]
        return market_cap.sort_index()
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return pd.Series(dtype=float)

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
    raise RuntimeError("No market‑cap data returned – check FMP key or date range.")

# Align to NYSE trading days (use QQQ prices as calendar anchor)
calendar = (
    yf.download("QQQ", start=START, end=END, progress=False)["Close"]
    .dropna()
    .index
)
market_cap_df = market_cap_df.reindex(calendar).ffill()

# ----------------------------------------------------------------------
# 2) Determine daily Top‑N rosters & rebalance dates -------------------
# ----------------------------------------------------------------------
def build_roster_df(mcap: pd.DataFrame, n: int) -> pd.Series:
    """
    Returns a pd.Series whose values are frozensets of the Top‑n tickers.
    """
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

rosters_by_size = {n: build_roster_df(market_cap_df, n) for n in SIZES}

# Union of all tickers ever needed, plus QQQ
all_strategy_tickers = sorted(
    set().union(*[item for n in SIZES for item in rosters_by_size[n].tolist()])
)
all_tickers = all_strategy_tickers + ["QQQ"]

# ----------------------------------------------------------------------
# 3) Price data (total return via Adj Close) ---------------------------
# ----------------------------------------------------------------------
print("Downloading price series from Yahoo Finance...")
prices = (
    yf.download(all_tickers, start=START, end=END, progress=False)["Close"]
    .dropna(how="all")
)
prices = prices.ffill()

# ----------------------------------------------------------------------
# 4) Portfolio back‑tester (buy‑and‑hold between changes) --------------
# ----------------------------------------------------------------------
def backtest(prc: pd.DataFrame, roster_series: pd.Series) -> pd.Series:
    rets = prc.pct_change().fillna(0)
    pv = 1.0
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

# ----------------------------------------------------------------------
# 5) Run back‑tests for every N and for QQQ ----------------------------
# ----------------------------------------------------------------------
print("Running back‑tests...")
curves = {"QQQ": (1 + prices["QQQ"].pct_change().fillna(0)).cumprod()}
for n in SIZES:
    curves[f"Top {n}"] = backtest(prices[rosters_by_size[n].explode().unique()], rosters_by_size[n])

curves_df = pd.concat(curves, axis=1)

# Daily and month‑end versions
curves_daily   = curves_df
curves_monthly = curves_df.resample("M").last()

# ----------------------------------------------------------------------
# 6) Plot results ------------------------------------------------------
# ----------------------------------------------------------------------
colors = {
    "QQQ":    "#000000",
    "Top 2":  "#FF0000",
    "Top 3":  "#0000FF",
    "Top 4":  "#00CC00",
    "Top 5":  "#FF6600",
    "Top 6":  "#9933CC",
    "Top 8":  "#FF66B2",
    "Top 10": "#00CCCC",
}
plt.figure(figsize=(15, 7))
for name in curves_daily.columns:
    lw   = 2.5 if name == "QQQ" else 1.7
    line = '-' if "Top" not in name or int(name.split()[1]) <= 3 else '--'
    plt.plot(curves_daily.index, curves_daily[name],
             label=name, linewidth=lw, linestyle=line,
             color=colors.get(name, None))
plt.yscale("log")
plt.grid(True, linestyle="--", linewidth=0.4)
plt.title("Daily equity curves  •  Jul 2005 – Jun 2025")
plt.xlabel("Date"); plt.ylabel("Growth of $1 (log)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# 7) Performance summary ----------------------------------------------
def perf_table(df: pd.DataFrame, label: str):
    print(f"\n{label} Performance")
    print("=" * 60)
    for col in df.columns:
        total = df[col].iloc[-1] / df[col].iloc[0] - 1
        yrs   = (df.index[-1] - df.index[0]).days / 365.25
        cagr  = (1 + total) ** (1 / yrs) - 1
        print(f"{col:6s} | Total: {total:8.1%} | CAGR: {cagr:6.2%}")

perf_table(curves_daily,   "Daily")
perf_table(curves_monthly, "Monthly")
