#!/usr/bin/env python3
"""
Test script for intelligent cache reuse optimization
====================================================

This script tests whether the cache optimization correctly reuses existing
cache files when the requested date range is smaller than what's already cached.

Test scenario:
- Cached data: 2005-01-01 to 2025-04-07
- Requested data: 2009-01-01 to 2025-04-07
- Expected: Should reuse cache and filter to requested range
"""

import sys
import os
sys.path.append('.')

from nasdaq_fmp_analysis_corrected import CorrectedFMPDataProvider
from config import FMP_API_KEY, START_DATE, END_DATE

def test_cache_optimization():
    """Test the intelligent cache reuse optimization"""
    
    print("=" * 60)
    print("TESTING INTELLIGENT CACHE REUSE OPTIMIZATION")
    print("=" * 60)
    print(f"Requested date range: {START_DATE} to {END_DATE}")
    print(f"Cached data range: 2005-01-01 to 2025-04-07")
    print(f"Expected: Should reuse cache and filter data")
    print("=" * 60)
    
    # Create data provider
    provider = CorrectedFMPDataProvider(FMP_API_KEY)
    
    # Test with a few major stocks
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in test_symbols:
        print(f"\n🧪 Testing {symbol}...")
        
        # Test historical prices (should use cache optimization)
        print(f"  📊 Requesting price data for {symbol}...")
        price_data = provider.get_historical_prices(symbol, START_DATE, END_DATE)
        
        if price_data and 'open' in price_data:
            open_prices = price_data['open']
            print(f"  ✅ Got {len(open_prices)} days of price data")
            print(f"      Date range: {open_prices.index.min()} to {open_prices.index.max()}")
        else:
            print(f"  ❌ Failed to get price data for {symbol}")
        
        # Test shares outstanding (should use cache optimization for enterprise values)
        print(f"  📈 Requesting shares outstanding for {symbol}...")
        shares_data = provider.get_historical_shares_outstanding(symbol)
        
        if shares_data is not None and len(shares_data) > 0:
            print(f"  ✅ Got shares outstanding data ({len(shares_data)} records)")
        else:
            print(f"  ❌ Failed to get shares outstanding for {symbol}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("Look for '♻️ ' messages above to confirm cache reuse!")
    print("=" * 60)

if __name__ == "__main__":
    test_cache_optimization()
