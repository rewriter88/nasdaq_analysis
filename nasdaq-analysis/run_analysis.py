#!/usr/bin/env python3
"""
Nasdaq Top-N Momentum Strategy Analysis Launcher

This script reads the DATA_SOURCE configuration from config.py and runs
the appropriate analysis script (FMP or Yahoo Finance).

Usage: python run_analysis.py
"""

import sys
import subprocess
import importlib.util

def load_config():
    """Load configuration from config.py"""
    try:
        spec = importlib.util.spec_from_file_location("config", "config.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config
    except Exception as e:
        print(f"Error loading config.py: {e}")
        sys.exit(1)

def main():
    """Main launcher function"""
    print("Nasdaq Top-N Momentum Strategy Analysis Launcher")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Get data source setting
    data_source = getattr(config, 'DATA_SOURCE', 'FMP').upper()
    
    print(f"Data Source: {data_source}")
    
    if data_source == "FMP":
        script_name = "nasdaq_fmp_analysis_corrected.py"
        print("Running FMP analysis with corrected methodology...")
        print("Features: Fixed survivorship bias, accurate market cap, premium data")
        
    elif data_source == "YAHOO":
        script_name = "nasdaq_yahoo_analysis.py"
        print("Running Yahoo Finance analysis for comparison...")
        print("Features: Free data source, good for verification")
        
    else:
        print(f"Error: Unknown DATA_SOURCE '{data_source}'")
        print("Valid options: 'FMP' or 'YAHOO'")
        sys.exit(1)
    
    print(f"Executing: {script_name}")
    print("-" * 50)
    
    # Run the selected analysis script
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print(f"\n{script_name} completed successfully!")
        else:
            print(f"\n{script_name} failed with return code {result.returncode}")
            sys.exit(result.returncode)
            
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
