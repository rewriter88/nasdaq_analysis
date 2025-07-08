# ===========================================================
# Simple Monthly Nasdaq Analysis - Learning from Reference Code
# • Uses Yahoo Finance monthly data instead of FMP
# • Clean start/end date ranges for each roster period
# • Simple equal-weight rebalancing at month-end
# • Total return via auto_adjust=True
# ===========================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1) Roster schedules with start/end date ranges (Top 1-10)
# ------------------------------------------------------------------
top1_schedule = [
    # start        , end          , tickers  (100% weight)
    ("2005-07-01",  "2011-12-31", ["MSFT"]),
    ("2012-01-01",  "2024-04-30", ["AAPL"]),
    ("2024-05-01",  "2025-06-30", ["NVDA"]),
]

top2_schedule = [
    # start        , end          , tickers  (equal-weight 50/50)
    ("2005-07-01",  "2009-12-31", ["MSFT", "INTC"]),
    ("2010-01-01",  "2024-04-30", ["AAPL", "MSFT"]),
    ("2024-05-01",  "2025-06-30", ["MSFT", "NVDA"]),
]

top3_schedule = [
    # start        , end          , tickers  (equal-weight 33⅓% each)
    ("2005-07-01",  "2009-12-31", ["MSFT", "INTC", "CSCO"]),
    ("2010-01-01",  "2015-06-30", ["AAPL", "MSFT", "GOOG"]),  # Using GOOG for full history
    ("2015-07-01",  "2022-12-31", ["AAPL", "MSFT", "AMZN"]),
    ("2023-01-01",  "2025-06-30", ["AAPL", "MSFT", "NVDA"]),
]

top4_schedule = [
    # start        , end          , tickers  (equal-weight 25% each)
    ("2005-07-01",  "2009-12-31", ["MSFT", "INTC", "CSCO", "ORCL"]),
    ("2010-01-01",  "2015-06-30", ["AAPL", "MSFT", "GOOG", "INTC"]),
    ("2015-07-01",  "2022-12-31", ["AAPL", "MSFT", "AMZN", "GOOG"]),
    ("2023-01-01",  "2025-06-30", ["AAPL", "MSFT", "NVDA", "GOOG"]),
]

top5_schedule = [
    # start        , end          , tickers  (equal-weight 20% each)
    ("2005-07-01",  "2009-12-31", ["MSFT", "INTC", "CSCO", "ORCL", "QCOM"]),
    ("2010-01-01",  "2015-06-30", ["AAPL", "MSFT", "GOOG", "INTC", "CSCO"]),
    ("2015-07-01",  "2022-12-31", ["AAPL", "MSFT", "AMZN", "GOOG", "META"]),
    ("2023-01-01",  "2025-06-30", ["AAPL", "MSFT", "NVDA", "GOOG", "META"]),
]

top6_schedule = [
    # start        , end          , tickers  (equal-weight ~16.7% each)
    ("2005-07-01",  "2009-12-31", ["MSFT", "INTC", "CSCO", "ORCL", "QCOM", "DELL"]),
    ("2010-01-01",  "2015-06-30", ["AAPL", "MSFT", "GOOG", "INTC", "CSCO", "ORCL"]),
    ("2015-07-01",  "2022-12-31", ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA"]),
    ("2023-01-01",  "2025-06-30", ["AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA"]),
]

top7_schedule = [
    # start        , end          , tickers  (equal-weight ~14.3% each)
    ("2005-07-01",  "2009-12-31", ["MSFT", "INTC", "CSCO", "ORCL", "QCOM", "DELL", "AMGN"]),
    ("2010-01-01",  "2015-06-30", ["AAPL", "MSFT", "GOOG", "INTC", "CSCO", "ORCL", "QCOM"]),
    ("2015-07-01",  "2022-12-31", ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NFLX"]),
    ("2023-01-01",  "2025-06-30", ["AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA", "NFLX"]),
]

top8_schedule = [
    # start        , end          , tickers  (equal-weight 12.5% each)
    ("2005-07-01",  "2009-12-31", ["MSFT", "INTC", "CSCO", "ORCL", "QCOM", "DELL", "AMGN", "EBAY"]),
    ("2010-01-01",  "2015-06-30", ["AAPL", "MSFT", "GOOG", "INTC", "CSCO", "ORCL", "QCOM", "AMGN"]),
    ("2015-07-01",  "2022-12-31", ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NFLX", "ADBE"]),
    ("2023-01-01",  "2025-06-30", ["AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA", "NFLX", "ADBE"]),
]

top9_schedule = [
    # start        , end          , tickers  (equal-weight ~11.1% each)
    ("2005-07-01",  "2009-12-31", ["MSFT", "INTC", "CSCO", "ORCL", "QCOM", "DELL", "AMGN", "EBAY", "ADBE"]),
    ("2010-01-01",  "2015-06-30", ["AAPL", "MSFT", "GOOG", "INTC", "CSCO", "ORCL", "QCOM", "AMGN", "EBAY"]),
    ("2015-07-01",  "2022-12-31", ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NFLX", "ADBE", "PYPL"]),
    ("2023-01-01",  "2025-06-30", ["AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA", "NFLX", "ADBE", "CRM"]),
]

top10_schedule = [
    # start        , end          , tickers  (equal-weight 10% each)
    ("2005-07-01",  "2009-12-31", ["MSFT", "INTC", "CSCO", "ORCL", "QCOM", "DELL", "AMGN", "EBAY", "ADBE", "GILD"]),
    ("2010-01-01",  "2015-06-30", ["AAPL", "MSFT", "GOOG", "INTC", "CSCO", "ORCL", "QCOM", "AMGN", "EBAY", "ADBE"]),
    ("2015-07-01",  "2022-12-31", ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NFLX", "ADBE", "PYPL", "CRM"]),
    ("2023-01-01",  "2025-06-30", ["AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA", "NFLX", "ADBE", "CRM", "AVGO"]),
]

# ------------------------------------------------------------------
# 2) Fetch monthly price history (adjusted for splits & dividends)
# ------------------------------------------------------------------
print("Fetching monthly data from Yahoo Finance...")
all_schedules = [
    top1_schedule, top2_schedule, top3_schedule, top4_schedule, top5_schedule,
    top6_schedule, top7_schedule, top8_schedule, top9_schedule, top10_schedule
]

all_tickers = sorted(
    {t for schedule in all_schedules for _, _, lst in schedule for t in lst} | {"QQQ"}
)

print(f"Downloading {len(all_tickers)} tickers: {all_tickers}")

# Download monthly data with total returns
data = yf.download(
    tickers=all_tickers,
    start="2005-07-01",
    end="2025-07-01",             # 1 extra month for latest month-end
    interval="1mo",               # Monthly data like reference code
    auto_adjust=True,             # Total returns (includes dividends)
    progress=False,
)

# Handle the data structure - with auto_adjust=True, we get "Close" instead of "Adj Close"
if isinstance(data.columns, pd.MultiIndex):
    prices = data["Close"]
else:
    prices = data  # Single ticker case

prices = (
    prices
    .dropna(how="all")               # Remove rows with all NaNs
    .resample("M").last()            # Force month-end alignment
)

print(f"Downloaded {len(prices)} months × {len(prices.columns)} tickers")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# Check data availability
print("\nData availability check:")
for ticker in ["AAPL", "MSFT", "GOOG", "META", "NVDA", "QQQ"]:
    if ticker in prices.columns:
        first_valid = prices[ticker].first_valid_index()
        print(f"  {ticker}: starts {first_valid.date()}")
    else:
        print(f"  {ticker}: NOT FOUND")

# ------------------------------------------------------------------
# 3) Helper: build weight matrix from a schedule (exactly like reference)
# ------------------------------------------------------------------
def build_weights(schedule, universe, index):
    """Build weight matrix from schedule with start/end date ranges."""
    w = pd.DataFrame(0.0, index=index, columns=universe)
    for start, end, names in schedule:
        mask = (index >= pd.to_datetime(start)) & (index <= pd.to_datetime(end))
        # Equal weight among the tickers in this period
        w.loc[mask, names] = 1.0 / len(names)
    return w

idx = prices.index
weights_top1 = build_weights(top1_schedule, prices.columns, idx)
weights_top2 = build_weights(top2_schedule, prices.columns, idx)
weights_top3 = build_weights(top3_schedule, prices.columns, idx)
weights_top4 = build_weights(top4_schedule, prices.columns, idx)
weights_top5 = build_weights(top5_schedule, prices.columns, idx)
weights_top6 = build_weights(top6_schedule, prices.columns, idx)
weights_top7 = build_weights(top7_schedule, prices.columns, idx)
weights_top8 = build_weights(top8_schedule, prices.columns, idx)
weights_top9 = build_weights(top9_schedule, prices.columns, idx)
weights_top10 = build_weights(top10_schedule, prices.columns, idx)

# ------------------------------------------------------------------
# 4) Compute monthly total returns & cumulative growth
# ------------------------------------------------------------------
rets = prices.pct_change().fillna(0)

def portfolio_cum(weights):
    """Calculate portfolio cumulative returns."""
    port_rets = (weights * rets).sum(axis=1)
    return (1 + port_rets).cumprod()

# Calculate performance curves
curve_top1 = portfolio_cum(weights_top1)
curve_top2 = portfolio_cum(weights_top2)
curve_top3 = portfolio_cum(weights_top3)
curve_top4 = portfolio_cum(weights_top4)
curve_top5 = portfolio_cum(weights_top5)
curve_top6 = portfolio_cum(weights_top6)
curve_top7 = portfolio_cum(weights_top7)
curve_top8 = portfolio_cum(weights_top8)
curve_top9 = portfolio_cum(weights_top9)
curve_top10 = portfolio_cum(weights_top10)
curve_qqq = (1 + rets["QQQ"]).cumprod()

# Align all curves to same index & starting value $1
curves = pd.concat([
    curve_qqq.rename("QQQ"),
    curve_top1.rename("Top 1"),
    curve_top2.rename("Top 2"),
    curve_top3.rename("Top 3"),
    curve_top4.rename("Top 4"),
    curve_top5.rename("Top 5"),
    curve_top6.rename("Top 6"),
    curve_top7.rename("Top 7"),
    curve_top8.rename("Top 8"),
    curve_top9.rename("Top 9"),
    curve_top10.rename("Top 10"),
], axis=1).dropna()

print(f"\nFinal curves shape: {curves.shape}")

# ------------------------------------------------------------------
# 5) Plot with distinctive colors and final portfolio values
# ------------------------------------------------------------------
plt.figure(figsize=(14, 8))

# Define distinctive colors for each strategy
colors = {
    "QQQ": "#2E2E2E",       # Dark gray
    "Top 1": "#E31A1C",     # Red
    "Top 2": "#FF7F00",     # Orange  
    "Top 3": "#1F78B4",     # Blue
    "Top 4": "#33A02C",     # Green
    "Top 5": "#6A3D9A",     # Purple
    "Top 6": "#FF1493",     # Deep pink
    "Top 7": "#00CED1",     # Dark turquoise
    "Top 8": "#FFD700",     # Gold
    "Top 9": "#DC143C",     # Crimson
    "Top 10": "#32CD32",    # Lime green
}

# Calculate final portfolio values for labels
initial_investment = 100000  # $100K initial
final_values = {}

for strategy in curves.columns:
    final_multiplier = curves[strategy].iloc[-1] / curves[strategy].iloc[0]
    final_value = final_multiplier * initial_investment
    final_values[strategy] = final_value

# Plot each strategy with final value in label
for strategy in curves.columns:
    final_val = final_values[strategy]
    label_text = f"{strategy} (${final_val:,.0f})"
    
    plt.plot(
        curves.index, 
        curves[strategy], 
        label=label_text,
        linewidth=2.5 if strategy == "QQQ" else 2.0,
        color=colors[strategy]
    )

plt.yscale("log")
plt.ylabel("Growth of $1 (log scale)")
plt.xlabel("Date")
plt.title("Nasdaq Top 1-10 Monthly Strategies • Jul 2005 – Jun 2025\nFinal Portfolio Values (Starting $100K)")
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
plt.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 6) Summary table (like reference code)
# ------------------------------------------------------------------
print(f"\nPerformance Summary (Monthly Rebalancing)")
print("=" * 60)
print(f"{'Strategy':8s} | {'Final Value':>12s} | {'Total Return':>12s} | {'CAGR':>8s}")
print("-" * 60)

initial_investment = 100000  # $100K initial

for name in curves.columns:
    total = curves[name].iloc[-1] / curves[name].iloc[0] - 1
    years = (curves.index[-1] - curves.index[0]).days / 365.25
    cagr = (1 + total) ** (1 / years) - 1
    final_value = curves[name].iloc[-1] * initial_investment
    
    print(f"{name:8s} | ${final_value:>11,.0f} | {total:11.1%} | {cagr:7.2%}")

print(f"\nMethodology:")
print(f"• Data source: Yahoo Finance monthly data")
print(f"• Rebalancing: Monthly, equal-weight within each strategy")
print(f"• Returns: Total returns (auto_adjust=True)")
print(f"• Period: {years:.1f} years")
print(f"• Schedule: Start/end date ranges (like reference code)")

# ------------------------------------------------------------------
# 7) Weight verification (debugging - sample strategies)
# ------------------------------------------------------------------
print(f"\nWeight Verification (sample dates):")
sample_dates = [curves.index[0], curves.index[len(curves)//2], curves.index[-1]]

weight_sets = [
    ("Top 1", weights_top1),
    ("Top 2", weights_top2), 
    ("Top 5", weights_top5),
    ("Top 10", weights_top10)
]

for date in sample_dates:
    print(f"\n{date.date()}:")
    for strategy_name, weights in weight_sets:
        if date in weights.index:
            active_weights = weights.loc[date][weights.loc[date] > 0]
            if len(active_weights) > 0:
                print(f"  {strategy_name}: {dict(active_weights.round(3))}")
            else:
                print(f"  {strategy_name}: No active positions")
