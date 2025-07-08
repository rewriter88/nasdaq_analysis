# ===========================================================
# True month-by-month equity curves
#   • QQQ (total-return proxy via yfinance “Adj Close”)
#   • “Top 2” strategy – equal-weight the #1 & #2 Nasdaq
#   • “Top 3” strategy – equal-weight the #1-#3 Nasdaq
#
# Roster is based on historic market-cap tables.
# Re-rank happens on the FIRST trading day *after* a change.
# Adjust the schedules below if you have finer-grained data.
# ===========================================================

# pip install yfinance pandas numpy matplotlib

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1)  Roster schedules  (yyyy-mm-dd  →  yyyy-mm-dd  →  constituent list)
# ------------------------------------------------------------------
top2_schedule = [
    # start        , end          , tickers  (equal-weight 50/50)
    ("2005-07-01",  "2009-12-31", ["MSFT", "INTC"]),
    ("2010-01-01",  "2024-04-30", ["AAPL", "MSFT"]),
    ("2024-05-01",  "2025-06-30", ["MSFT", "NVDA"]),
]

top3_schedule = [
    # start        , end          , tickers  (equal-weight 33 ⅓ % each)
    ("2005-07-01",  "2009-12-31", ["MSFT", "INTC", "CSCO"]),
    ("2010-01-01",  "2015-06-30", ["AAPL", "MSFT", "GOOGL"]),
    ("2015-07-01",  "2022-12-31", ["AAPL", "MSFT", "AMZN"]),
    ("2023-01-01",  "2025-06-30", ["AAPL", "MSFT", "NVDA"]),
]

# Additional schedules for top 4-10
top4_schedule = [
    ("2005-07-01",  "2009-12-31", ["MSFT", "INTC", "CSCO", "ORCL"]),
    ("2010-01-01",  "2015-06-30", ["AAPL", "MSFT", "GOOGL", "INTC"]),
    ("2015-07-01",  "2022-12-31", ["AAPL", "MSFT", "AMZN", "GOOGL"]),
    ("2023-01-01",  "2025-06-30", ["AAPL", "MSFT", "NVDA", "GOOGL"]),
]

top5_schedule = [
    ("2005-07-01",  "2009-12-31", ["MSFT", "INTC", "CSCO", "ORCL", "QCOM"]),
    ("2010-01-01",  "2015-06-30", ["AAPL", "MSFT", "GOOGL", "INTC", "CSCO"]),
    ("2015-07-01",  "2022-12-31", ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]),
    ("2023-01-01",  "2025-06-30", ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]),
]

# Larger portfolios
larger_schedules = {
    "Top 6": [
        ("2005-07-01",  "2009-12-31", ["MSFT", "INTC", "CSCO", "ORCL", "QCOM", "AMZN"]),
        ("2010-01-01",  "2015-06-30", ["AAPL", "MSFT", "GOOGL", "INTC", "CSCO", "ORCL"]),
        ("2015-07-01",  "2022-12-31", ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA"]),
        ("2023-01-01",  "2025-06-30", ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"]),
    ],
    "Top 8": [
        ("2005-07-01",  "2009-12-31", ["MSFT", "INTC", "CSCO", "ORCL", "QCOM", "AMZN", "AMAT", "TXN"]),
        ("2010-01-01",  "2015-06-30", ["AAPL", "MSFT", "GOOGL", "INTC", "CSCO", "ORCL", "QCOM", "AMZN"]),
        ("2015-07-01",  "2022-12-31", ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "INTC", "CSCO"]),
        ("2023-01-01",  "2025-06-30", ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "ADBE"]),
    ],
    "Top 10": [
        ("2005-07-01",  "2009-12-31", ["MSFT", "INTC", "CSCO", "ORCL", "QCOM", "AMZN", "AMAT", "TXN", "ADBE", "INTU"]),
        ("2010-01-01",  "2015-06-30", ["AAPL", "MSFT", "GOOGL", "INTC", "CSCO", "ORCL", "QCOM", "AMZN", "ADBE", "INTU"]),
        ("2015-07-01",  "2022-12-31", ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "INTC", "CSCO", "ADBE", "NFLX"]),
        ("2023-01-01",  "2025-06-30", ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "ADBE", "AVGO", "CSCO"]),
    ],
}

# ------------------------------------------------------------------
# 2)  Fetch monthly price history (adjusted for splits & dividends)
# ------------------------------------------------------------------
all_tickers = sorted(
    {t for _, _, lst in top2_schedule + top3_schedule + top4_schedule + top5_schedule for t in lst} |
    {t for schedule in larger_schedules.values() for _, _, lst in schedule for t in lst} |
    {"QQQ"}
)
# Fetch both daily and monthly data
prices_daily = (
    yf.download(
        tickers=all_tickers,
        start="2005-07-01",
        end="2025-07-01",
        interval="1d",
        progress=False,
    )
    ["Close"]
    .dropna(how="all")
)

prices_monthly = prices_daily.resample("M").last()

# ------------------------------------------------------------------
# 3)  Helper: build weight matrix from a schedule
# ------------------------------------------------------------------
def build_weights(schedule, universe, index):
    w = pd.DataFrame(0.0, index=index, columns=universe)
    for start, end, names in schedule:
        mask = (index >= pd.to_datetime(start)) & (index <= pd.to_datetime(end))
        w.loc[mask, names] = 1.0 / len(names)
    return w

def calculate_performance(prices_df):
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

# Calculate both daily and monthly performance
curves_daily = calculate_performance(prices_daily)
curves_monthly = calculate_performance(prices_monthly)

# ------------------------------------------------------------------
# 5)  Plot both daily and monthly results
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))

# Set up distinct colors for different portfolios
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

# Daily plot
for name in curves_daily.columns:
    linewidth = 2.5 if name == "QQQ" else 1.8
    style = '-' if name in ["QQQ", "Top 2", "Top 3"] else '--'
    ax1.plot(curves_daily.index, curves_daily[name],
             label=name, linewidth=linewidth, color=colors[name],
             linestyle=style)

ax1.set_yscale("log")
ax1.set_ylabel("Growth of $1 (log scale)")
ax1.set_xlabel("Date")
ax1.set_title("Daily equity curves  •  Jul 2005 – Jun 2025")
ax1.grid(True, linestyle="--", linewidth=0.4)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

# Monthly plot
for name in curves_monthly.columns:
    linewidth = 2.5 if name == "QQQ" else 1.8
    style = '-' if name in ["QQQ", "Top 2", "Top 3"] else '--'
    ax2.plot(curves_monthly.index, curves_monthly[name], 
             label=name, linewidth=linewidth, color=colors[name],
             linestyle=style)

ax2.set_yscale("log")
ax2.set_ylabel("Growth of $1 (log scale)")
ax2.set_xlabel("Date")
ax2.set_title("Monthly equity curves  •  Jul 2005 – Jun 2025")
ax2.grid(True, linestyle="--", linewidth=0.4)
ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 6)  Performance summary tables
# ------------------------------------------------------------------
print("\nDaily Performance:")
print("=" * 50)
for name in curves_daily.columns:
    total = curves_daily[name].iloc[-1] / curves_daily[name].iloc[0] - 1
    years = (curves_daily.index[-1] - curves_daily.index[0]).days / 365.25
    cagr = (1 + total) ** (1 / years) - 1
    print(f"{name:5s}  |  Total: {total:7.1%}  |  CAGR: {cagr:5.2%}")

print("\nMonthly Performance:")
print("=" * 50)
for name in curves_monthly.columns:
    total = curves_monthly[name].iloc[-1] / curves_monthly[name].iloc[0] - 1
    years = (curves_monthly.index[-1] - curves_monthly.index[0]).days / 365.25
    cagr = (1 + total) ** (1 / years) - 1
    print(f"{name:5s}  |  Total: {total:7.1%}  |  CAGR: {cagr:5.2%}")
