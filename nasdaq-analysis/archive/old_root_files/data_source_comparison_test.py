#!/usr/bin/env python3
"""
Dat    def __init__(self):
        self.fmp_api_key = FMP_API_KEY
        self.start_date = pd.to_datetime(START_DATE)
        self.end_date = pd.to_datetime(END_DATE)
        self.sample_size = 5  # Testing with 5 days firstrce Comparison Test: FMP vs Yahoo

This script samples 1000 random days over the analysis period and compares:
1. Adjusted open prices
2. Shares outstanding

Outputs:
- CSV with detailed comparison for each sampled day
- Summary statistics table at the bottom analyzing differences

Usage: python data_source_comparison_test.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time

# Import config
from config import FMP_API_KEY, START_DATE, END_DATE

class DataSourceComparisonTest:
    def __init__(self):
        self.fmp_api_key = FMP_API_KEY
        self.start_date = pd.to_datetime(START_DATE)
        self.end_date = pd.to_datetime(END_DATE)
        self.sample_size = 100  # Testing with 100 random days
        
        # Create cache directories
        os.makedirs("fmp_cache_test", exist_ok=True)
        os.makedirs("yahoo_cache_test", exist_ok=True)
        os.makedirs("results/csv", exist_ok=True)
        
        # Test symbols (major NASDAQ stocks that should be available in both sources)
        self.test_symbols = [
            "AAPL", "MSFT", "GOOGL"  # Just 3 symbols for initial test
        ]
        
    def generate_random_sample_dates(self) -> List[pd.Timestamp]:
        """Generate random business days within the analysis period."""
        print(f"Generating {self.sample_size} random sample dates between {self.start_date} and {self.end_date}")
        
        # For testing, let's use a recent date (last 2 years) to avoid old stock split issues
        recent_start = pd.to_datetime("2023-01-01") 
        business_days = pd.bdate_range(start=recent_start, end=self.end_date)
        print(f"Total business days available: {len(business_days)}")
        
        # Sample random dates
        if len(business_days) < self.sample_size:
            print(f"Warning: Only {len(business_days)} business days available, using all of them")
            return list(business_days)
        
        random_dates = random.sample(list(business_days), self.sample_size)
        random_dates.sort()
        
        print(f"Selected {len(random_dates)} random dates: {[d.strftime('%Y-%m-%d') for d in random_dates]}")
        return random_dates
    
    def fetch_fmp_data(self, symbol: str, date: pd.Timestamp) -> Tuple[Optional[float], Optional[float]]:
        """Fetch adjusted open price and shares outstanding from FMP for a specific date."""
        cache_file = f"fmp_cache_test/{symbol}_{date.strftime('%Y-%m-%d')}.json"
        
        # Check cache first
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('adj_open'), data.get('shares_outstanding')
        
        try:
            # Get historical price data for the specific date
            date_str = date.strftime('%Y-%m-%d')
            price_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
            price_params = {
                'apikey': self.fmp_api_key,
                'from': date_str,
                'to': date_str
            }
            
            price_response = requests.get(price_url, params=price_params)
            if price_response.status_code != 200:
                print(f"FMP price API error for {symbol} on {date_str}: {price_response.status_code}")
                return None, None
            
            price_data = price_response.json()
            adj_open = None
            
            if 'historical' in price_data and len(price_data['historical']) > 0:
                day_data = price_data['historical'][0]
                adj_open = day_data.get('adjOpen', day_data.get('open'))
            
            # Get shares outstanding (use enterprise value endpoint for historical data)
            shares_url = f"https://financialmodelingprep.com/api/v3/enterprise-values/{symbol}"
            shares_params = {
                'apikey': self.fmp_api_key,
                'period': 'quarter'  # Use quarterly data for better historical coverage
            }
            
            shares_response = requests.get(shares_url, params=shares_params)
            shares_outstanding = None
            
            if shares_response.status_code == 200:
                shares_data = shares_response.json()
                if len(shares_data) > 0:
                    # Find the most recent quarterly report before or on our date
                    for share_info in shares_data:
                        share_date = pd.to_datetime(share_info.get('date'))
                        if share_date <= date:
                            shares_outstanding = share_info.get('numberOfShares')
                            break
            
            # Cache the result
            cache_data = {
                'adj_open': adj_open,
                'shares_outstanding': shares_outstanding
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            # Rate limiting
            time.sleep(0.2)
            
            return adj_open, shares_outstanding
            
        except Exception as e:
            print(f"Error fetching FMP data for {symbol} on {date}: {e}")
            return None, None
    
    def fetch_yahoo_data(self, symbol: str, date: pd.Timestamp) -> Tuple[Optional[float], Optional[float]]:
        """Fetch adjusted open price and shares outstanding from Yahoo for a specific date."""
        cache_file = f"yahoo_cache_test/{symbol}_{date.strftime('%Y-%m-%d')}.json"
        
        # Check cache first
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('adj_open'), data.get('shares_outstanding')
        
        try:
            # Get stock info
            stock = yf.Ticker(symbol)
            
            # Get historical data around the date (need a range for Yahoo)
            start_fetch = date - timedelta(days=5)
            end_fetch = date + timedelta(days=5)
            
            # Get data with auto_adjust=False to get both raw and adjusted prices
            hist_data = stock.history(start=start_fetch, end=end_fetch, auto_adjust=False)
            
            adj_open = None
            if not hist_data.empty:
                # Convert both to timezone-naive for comparison
                hist_dates_naive = hist_data.index.tz_localize(None) if hist_data.index.tz is not None else hist_data.index
                date_naive = date.tz_localize(None) if date.tz is not None else date
                
                # Find the closest date
                closest_idx = hist_dates_naive.get_indexer([date_naive], method='nearest')[0]
                closest_date = hist_dates_naive[closest_idx]
                
                if abs((closest_date.date() - date_naive.date()).days) <= 2:  # Within 2 days
                    # Calculate adjusted open price using the same method as yfinance
                    # adj_open = open * (adj_close / close)
                    row = hist_data.iloc[closest_idx]
                    
                    # Check if required columns exist and are valid
                    required_cols = ['Open', 'Close', 'Adj Close']
                    if all(col in hist_data.columns for col in required_cols):
                        open_val = row['Open']
                        close_val = row['Close'] 
                        adj_close_val = row['Adj Close']
                        
                        if (pd.notna(open_val) and pd.notna(close_val) and pd.notna(adj_close_val) 
                            and close_val != 0):
                            adj_open = open_val * (adj_close_val / close_val)
                        else:
                            print(f"Invalid data for {symbol} on {closest_date}: Open={open_val}, Close={close_val}, Adj Close={adj_close_val}")
                    else:
                        print(f"Missing required columns for {symbol}. Available: {list(hist_data.columns)}")
            
            # Get shares outstanding from info
            shares_outstanding = None
            try:
                info = stock.info
                shares_outstanding = info.get('sharesOutstanding', info.get('impliedSharesOutstanding'))
            except:
                pass
            
            # Cache the result
            cache_data = {
                'adj_open': adj_open,
                'shares_outstanding': shares_outstanding
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            # Rate limiting
            time.sleep(0.1)
            
            return adj_open, shares_outstanding
            
        except Exception as e:
            print(f"Error fetching Yahoo data for {symbol} on {date}: {e}")
            return None, None
    
    def run_comparison(self) -> pd.DataFrame:
        """Run the comparison test and return results DataFrame."""
        print("Starting data source comparison test...")
        
        # Generate random sample dates
        sample_dates = self.generate_random_sample_dates()
        
        results = []
        total_comparisons = len(sample_dates) * len(self.test_symbols)
        completed = 0
        
        for date in sample_dates:
            for symbol in self.test_symbols:
                print(f"Processing {symbol} on {date.strftime('%Y-%m-%d')} ({completed+1}/{total_comparisons})")
                
                # Fetch data from both sources
                fmp_adj_open, fmp_shares = self.fetch_fmp_data(symbol, date)
                yahoo_adj_open, yahoo_shares = self.fetch_yahoo_data(symbol, date)
                
                # Calculate differences
                price_diff = None
                price_pct_diff = None
                if fmp_adj_open is not None and yahoo_adj_open is not None:
                    price_diff = fmp_adj_open - yahoo_adj_open
                    if yahoo_adj_open != 0:
                        price_pct_diff = (price_diff / yahoo_adj_open) * 100
                
                shares_diff = None
                shares_pct_diff = None
                if fmp_shares is not None and yahoo_shares is not None:
                    shares_diff = fmp_shares - yahoo_shares
                    if yahoo_shares != 0:
                        shares_pct_diff = (shares_diff / yahoo_shares) * 100
                
                results.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Time': datetime.now().strftime('%H:%M:%S'),
                    'Symbol': symbol,
                    'FMP_Adj_Open_Price': fmp_adj_open,
                    'Yahoo_Adj_Open_Price': yahoo_adj_open,
                    'Price_Difference': price_diff,
                    'Price_Pct_Difference': price_pct_diff,
                    'FMP_Shares_Outstanding': fmp_shares,
                    'Yahoo_Shares_Outstanding': yahoo_shares,
                    'Shares_Difference': shares_diff,
                    'Shares_Pct_Difference': shares_pct_diff
                })
                
                completed += 1
        
        return pd.DataFrame(results)
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for the differences."""
        print("Generating summary statistics...")
        
        # Filter rows where both sources have data
        price_valid = df.dropna(subset=['Price_Difference'])
        shares_valid = df.dropna(subset=['Shares_Difference'])
        
        summary_stats = []
        
        # Price difference statistics
        if not price_valid.empty:
            summary_stats.extend([
                {'Metric': 'PRICE DIFFERENCES', 'Value': ''},
                {'Metric': 'Valid Price Comparisons', 'Value': len(price_valid)},
                {'Metric': 'Mean Price Difference ($)', 'Value': f"{price_valid['Price_Difference'].mean():.4f}"},
                {'Metric': 'Median Price Difference ($)', 'Value': f"{price_valid['Price_Difference'].median():.4f}"},
                {'Metric': 'Std Dev Price Difference ($)', 'Value': f"{price_valid['Price_Difference'].std():.4f}"},
                {'Metric': 'Max Price Difference ($)', 'Value': f"{price_valid['Price_Difference'].max():.4f}"},
                {'Metric': 'Min Price Difference ($)', 'Value': f"{price_valid['Price_Difference'].min():.4f}"},
                {'Metric': 'Mean Price Difference (%)', 'Value': f"{price_valid['Price_Pct_Difference'].mean():.4f}%"},
                {'Metric': 'Median Price Difference (%)', 'Value': f"{price_valid['Price_Pct_Difference'].median():.4f}%"},
                {'Metric': 'Std Dev Price Difference (%)', 'Value': f"{price_valid['Price_Pct_Difference'].std():.4f}%"},
                {'Metric': '', 'Value': ''},
            ])
        
        # Shares outstanding statistics
        if not shares_valid.empty:
            summary_stats.extend([
                {'Metric': 'SHARES OUTSTANDING DIFFERENCES', 'Value': ''},
                {'Metric': 'Valid Shares Comparisons', 'Value': len(shares_valid)},
                {'Metric': 'Mean Shares Difference', 'Value': f"{shares_valid['Shares_Difference'].mean():.0f}"},
                {'Metric': 'Median Shares Difference', 'Value': f"{shares_valid['Shares_Difference'].median():.0f}"},
                {'Metric': 'Std Dev Shares Difference', 'Value': f"{shares_valid['Shares_Difference'].std():.0f}"},
                {'Metric': 'Max Shares Difference', 'Value': f"{shares_valid['Shares_Difference'].max():.0f}"},
                {'Metric': 'Min Shares Difference', 'Value': f"{shares_valid['Shares_Difference'].min():.0f}"},
                {'Metric': 'Mean Shares Difference (%)', 'Value': f"{shares_valid['Shares_Pct_Difference'].mean():.4f}%"},
                {'Metric': 'Median Shares Difference (%)', 'Value': f"{shares_valid['Shares_Pct_Difference'].median():.4f}%"},
                {'Metric': 'Std Dev Shares Difference (%)', 'Value': f"{shares_valid['Shares_Pct_Difference'].std():.4f}%"},
                {'Metric': '', 'Value': ''},
            ])
        
        # Data availability statistics
        total_attempts = len(df)
        fmp_price_available = df['FMP_Adj_Open_Price'].notna().sum()
        yahoo_price_available = df['Yahoo_Adj_Open_Price'].notna().sum()
        fmp_shares_available = df['FMP_Shares_Outstanding'].notna().sum()
        yahoo_shares_available = df['Yahoo_Shares_Outstanding'].notna().sum()
        
        summary_stats.extend([
            {'Metric': 'DATA AVAILABILITY', 'Value': ''},
            {'Metric': 'Total Attempted Comparisons', 'Value': total_attempts},
            {'Metric': 'FMP Price Data Available', 'Value': f"{fmp_price_available} ({fmp_price_available/total_attempts*100:.1f}%)"},
            {'Metric': 'Yahoo Price Data Available', 'Value': f"{yahoo_price_available} ({yahoo_price_available/total_attempts*100:.1f}%)"},
            {'Metric': 'FMP Shares Data Available', 'Value': f"{fmp_shares_available} ({fmp_shares_available/total_attempts*100:.1f}%)"},
            {'Metric': 'Yahoo Shares Data Available', 'Value': f"{yahoo_shares_available} ({yahoo_shares_available/total_attempts*100:.1f}%)"},
        ])
        
        return pd.DataFrame(summary_stats)
    
    def save_results(self, df: pd.DataFrame, summary_df: pd.DataFrame):
        """Save results to CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/csv/data_source_comparison_{timestamp}.csv"
        
        print(f"Saving results to {output_file}")
        
        # Combine main results and summary
        with open(output_file, 'w') as f:
            # Write main results
            df.to_csv(f, index=False)
            
            # Add separator
            f.write('\n\n')
            f.write('SUMMARY STATISTICS\n')
            
            # Write summary statistics
            summary_df.to_csv(f, index=False)
        
        print(f"Results saved to {output_file}")
        return output_file

def main():
    """Run the data source comparison test."""
    print("=" * 80)
    print("DATA SOURCE COMPARISON TEST: FMP vs Yahoo")
    print("=" * 80)
    print("Starting test initialization...")
    
    # Create and run the test
    test = DataSourceComparisonTest()
    print("Test object created successfully")
    
    # Run comparison
    print("Running comparison...")
    results_df = test.run_comparison()
    print("Comparison completed")
    
    # Generate summary statistics
    print("Generating summary statistics...")
    summary_df = test.generate_summary_statistics(results_df)
    
    # Save results
    print("Saving results...")
    output_file = test.save_results(results_df, summary_df)
    
    print()
    print("=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"Total comparisons attempted: {len(results_df)}")
    print(f"Valid price comparisons: {results_df['Price_Difference'].notna().sum()}")
    print(f"Valid shares comparisons: {results_df['Shares_Difference'].notna().sum()}")
    print()
    
    # Display summary statistics
    print("SUMMARY STATISTICS PREVIEW:")
    print("-" * 40)
    for _, row in summary_df.iterrows():
        if row['Value'] != '':
            print(f"{row['Metric']}: {row['Value']}")
        else:
            print()

if __name__ == "__main__":
    main()
