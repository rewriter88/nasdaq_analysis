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

# ------------------------------------------------------------------
# 2)  Fetch monthly price history (adjusted for splits & dividends)
# ------------------------------------------------------------------
all_tickers = sorted(
    {t for _, _, lst in top2_schedule + top3_schedule for t in lst} | {"QQQ"}
)
prices = (
    yf.download(
        tickers=all_tickers,
        start="2005-07-01",
        end="2025-07-01",             # 1 extra month for latest month-end
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )
    .loc[:, "Adj Close"]
    .dropna(how="all")               # remove rows with all NaNs (e.g., partial month)
    .resample("M").last()            # force month-end alignment
)

# ------------------------------------------------------------------
# 3)  Helper: build weight matrix from a schedule
# ------------------------------------------------------------------
def build_weights(schedule, universe, index):
    w = pd.DataFrame(0.0, index=index, columns=universe)
    for start, end, names in schedule:
        mask = (index >= pd.to_datetime(start)) & (index <= pd.to_datetime(end))
        w.loc[mask, names] = 1.0 / len(names)
    return w

idx = prices.index
weights_top2 = build_weights(top2_schedule, prices.columns, idx)
weights_top3 = build_weights(top3_schedule, prices.columns, idx)

# ------------------------------------------------------------------
# 4)  Compute monthly total returns & cumulative growth
# ------------------------------------------------------------------
rets = prices.pct_change().fillna(0)

def portfolio_cum(weights):
    port_rets = (weights * rets).sum(axis=1)
    return (1 + port_rets).cumprod()

curve_top2 = portfolio_cum(weights_top2)
curve_top3 = portfolio_cum(weights_top3)
curve_qqq  = (1 + rets["QQQ"]).cumprod()

# Align all three to same index & starting value $1
curves = pd.concat(
    [curve_qqq.rename("QQQ"), curve_top2.rename("Top 2"), curve_top3.rename("Top 3")],
    axis=1,
).dropna()

# ------------------------------------------------------------------
# 5)  Plot
# ------------------------------------------------------------------
plt.figure(figsize=(11, 6))
plt.plot(curves.index, curves["QQQ"],  label="QQQ",      linewidth=2)
plt.plot(curves.index, curves["Top 2"], label="Top 2",   linewidth=1.8)
plt.plot(curves.index, curves["Top 3"], label="Top 3",   linewidth=1.8)
plt.yscale("log")
plt.ylabel("Growth of $1 (log scale)")
plt.xlabel("Date")
plt.title("True month-by-month equity curves  •  Jul 2005 – Jun 2025")
plt.grid(True, linestyle="--", linewidth=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 6)  Optional summary table
# ------------------------------------------------------------------
for name in curves.columns:
    total   = curves[name].iloc[-1] / curves[name].iloc[0] - 1
    years   = (curves.index[-1] - curves.index[0]).days / 365.25
    cagr    = (1 + total) ** (1 / years) - 1
    print(f"{name:5s}  |  Total: {total:7.1%}  |  CAGR: {cagr:5.2%}")
