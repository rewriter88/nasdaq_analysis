#!/usr/bin/env python3
"""
Test script to check FMP API data availability for 20 years
"""

import requests
import pandas as pd
import json
from datetime import datetime

# Your API key
API_KEY = "2whsxV4FK6zNdDQs0Z5yrggdvxfeHPAS"
BASE_URL = "https://financialmodelingprep.com/api/v3"

def test_market_cap_data(symbol="AAPL"):
    """Test market cap data for a symbol"""
    print(f"Testing market cap data for {symbol}...")
    
    endpoint = f"{BASE_URL}/historical-market-capitalization/{symbol}"
    params = {
        'apikey': API_KEY,
        'from': '2005-01-01',
        'to': '2025-01-01',
        'serietype': 'line',
        'limit': 10000
    }
    
    response = requests.get(endpoint, params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Data type: {type(data)}")
        
        if isinstance(data, list) and len(data) > 0:
            print(f"Number of records: {len(data)}")
            print(f"First record: {data[0]}")
            print(f"Last record: {data[-1]}")
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"Total days: {len(df)}")
            
        elif isinstance(data, dict):
            print(f"Dictionary keys: {data.keys()}")
            if 'historical' in data:
                hist = data['historical']
                print(f"Historical records: {len(hist)}")
        else:
            print(f"Unexpected data format: {data}")
    else:
        print(f"Error: {response.text}")

def test_price_data(symbol="AAPL"):
    """Test price data for a symbol"""
    print(f"\nTesting price data for {symbol}...")
    
    endpoint = f"{BASE_URL}/historical-price-full/{symbol}"
    params = {
        'apikey': API_KEY,
        'from': '2005-01-01',
        'to': '2025-01-01',
        'serietype': 'line',
        'limit': 10000
    }
    
    response = requests.get(endpoint, params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict) and 'historical' in data:
            hist = data['historical']
            print(f"Number of records: {len(hist)}")
            
            if len(hist) > 0:
                print(f"First record: {hist[0]}")
                print(f"Last record: {hist[-1]}")
                
                df = pd.DataFrame(hist)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                print(f"Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"Total days: {len(df)}")
        else:
            print(f"Unexpected data format or keys: {data.keys() if isinstance(data, dict) else type(data)}")
    else:
        print(f"Error: {response.text}")

def test_api_info():
    """Test API info to see account limits"""
    print(f"\nTesting API account info...")
    
    endpoint = f"{BASE_URL}/account-info"
    params = {'apikey': API_KEY}
    
    response = requests.get(endpoint, params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Account info: {json.dumps(data, indent=2)}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_api_info()
    test_market_cap_data("AAPL")
    test_price_data("AAPL")
    
    # Test a few more symbols
    for symbol in ["MSFT", "GOOGL", "QQQ"]:
        test_market_cap_data(symbol)
