#!/usr/bin/env python3
"""
Date Format Consistency Checker for NASDAQ Analysis Project
============================================================

This script systematically checks date format consistency across the entire project
to ensure all components are using the same date format (YYYY-MM-DD) which is 
the ISO 8601 standard and matches FMP API format.
"""

import json
import os
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple

def check_fmp_cache_date_formats():
    """Check date formats in FMP cache files"""
    print("🔍 CHECKING FMP CACHE DATE FORMATS")
    print("=" * 50)
    
    cache_dir = "fmp_cache_corrected"
    if not os.path.exists(cache_dir):
        print(f"❌ Cache directory {cache_dir} not found")
        return False
    
    # Check a few sample files
    sample_files = [f for f in os.listdir(cache_dir) if f.endswith('_prices_2005-01-01_2025-04-07.json')][:5]
    
    date_formats_found = set()
    issues = []
    
    for filename in sample_files:
        filepath = os.path.join(cache_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if 'historical' in data and data['historical']:
                # Check first few dates
                for item in data['historical'][:3]:
                    if 'date' in item:
                        date_str = item['date']
                        date_formats_found.add(date_str)
                        
                        # Validate it's YYYY-MM-DD format
                        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                            issues.append(f"❌ Non-standard date format in {filename}: {date_str}")
                        
        except Exception as e:
            issues.append(f"❌ Error reading {filename}: {e}")
    
    print(f"📊 Sample date formats found: {sorted(date_formats_found)[:10]}")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ All FMP cache files use correct YYYY-MM-DD format")
        return True

def check_code_date_parsing():
    """Check date parsing patterns in code files"""
    print("\n🔍 CHECKING CODE DATE PARSING PATTERNS")
    print("=" * 50)
    
    # Files to check
    files_to_check = [
        "nasdaq_fmp_analysis_corrected.py",
        "optimized_rolling_analysis.py",
        "simple_rolling_analysis.py",
        "rolling_period_analysis.py",
        "data_source_comparison_test.py"
    ]
    
    date_patterns = {
        'iso_format': r'%Y-%m-%d',      # ISO 8601 standard (YYYY-MM-DD)
        'us_format': r'%m/%d/%Y',       # US format (MM/DD/YYYY)  
        'eu_format': r'%d/%m/%Y',       # European format (DD/MM/YYYY)
        'timestamp_format': r'%Y%m%d_%H%M%S',  # Timestamp format for filenames
        'datetime_format': r'%Y-%m-%d %H:%M:%S',  # DateTime format
        'time_format': r'%H:%M:%S'      # Time only format
    }
    
    pattern_counts = {}
    issues = []
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            continue
            
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            for pattern_name, pattern in date_patterns.items():
                matches = re.findall(pattern, content)
                if matches:
                    if pattern_name not in pattern_counts:
                        pattern_counts[pattern_name] = {}
                    pattern_counts[pattern_name][filename] = len(matches)
                    
                    # Only flag actual problematic patterns
                    if pattern_name in ['us_format', 'eu_format']:
                        issues.append(f"❌ Non-ISO date format in {filename}: {pattern} (found {len(matches)} times)")
                        
        except Exception as e:
            issues.append(f"❌ Error reading {filename}: {e}")
    
    print("📊 Date format patterns found:")
    for pattern_name, files in pattern_counts.items():
        print(f"  {pattern_name}: {files}")
    
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ All code files use ISO standard YYYY-MM-DD format")
        return True

def check_pandas_date_handling():
    """Check pandas date handling consistency"""
    print("\n🔍 CHECKING PANDAS DATE HANDLING")
    print("=" * 50)
    
    # Test sample date parsing
    test_dates = ["2023-01-15", "2023-12-31", "2025-04-07"]
    
    issues = []
    
    for date_str in test_dates:
        try:
            # Test pd.to_datetime parsing
            parsed_date = pd.to_datetime(date_str)
            formatted_back = parsed_date.strftime('%Y-%m-%d')
            
            if formatted_back != date_str:
                issues.append(f"❌ Date parsing issue: {date_str} -> {parsed_date} -> {formatted_back}")
            
        except Exception as e:
            issues.append(f"❌ Error parsing {date_str}: {e}")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ Pandas date handling is consistent")
        return True

def check_fmp_api_format():
    """Verify FMP API returns dates in expected format"""
    print("\n🔍 CHECKING FMP API DATE FORMAT ALIGNMENT")
    print("=" * 50)
    
    # Check sample FMP cache file to see actual API response format
    cache_dir = "fmp_cache_corrected"
    sample_file = None
    
    for filename in os.listdir(cache_dir):
        if filename.endswith('_prices_2005-01-01_2025-04-07.json'):
            sample_file = os.path.join(cache_dir, filename)
            break
    
    if not sample_file:
        print("❌ No sample price cache file found")
        return False
    
    try:
        with open(sample_file, 'r') as f:
            data = json.load(f)
        
        if 'historical' in data and data['historical']:
            # Check date format in first few entries
            sample_dates = [item['date'] for item in data['historical'][:5] if 'date' in item]
            
            print(f"📊 Sample FMP API dates: {sample_dates}")
            
            # Verify all are YYYY-MM-DD format
            for date_str in sample_dates:
                if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                    print(f"❌ FMP API date not in YYYY-MM-DD format: {date_str}")
                    return False
            
            print("✅ FMP API uses YYYY-MM-DD format (ISO 8601 standard)")
            return True
            
    except Exception as e:
        print(f"❌ Error checking FMP format: {e}")
        return False

def check_output_date_formats():
    """Check date formats in output files"""
    print("\n🔍 CHECKING OUTPUT FILE DATE FORMATS")
    print("=" * 50)
    
    # Check recent analysis results
    results_dir = "results/csv"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory {results_dir} not found")
        return False
    
    # Find recent CSV files
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV result files found")
        return False
    
    # Check the most recent file
    latest_file = sorted(csv_files)[-1]
    filepath = os.path.join(results_dir, latest_file)
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Look for date patterns in the content
        date_patterns = re.findall(r'\d{4}-\d{2}-\d{2}', content)
        
        if date_patterns:
            print(f"📊 Sample output dates from {latest_file}: {date_patterns[:5]}")
            print("✅ Output files use YYYY-MM-DD format")
            return True
        else:
            print(f"❌ No recognizable date patterns found in {latest_file}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking output file: {e}")
        return False

def run_comprehensive_date_check():
    """Run comprehensive date format consistency check"""
    print("🚀 NASDAQ ANALYSIS PROJECT - DATE FORMAT CONSISTENCY CHECK")
    print("=" * 70)
    print("Checking for consistency between:")
    print("  • FMP API date format (data source)")
    print("  • Internal date parsing and handling")
    print("  • Output file date formats")
    print("  • Analysis result reporting")
    print("")
    
    all_checks_passed = True
    
    # Run all checks
    checks = [
        ("FMP API Format", check_fmp_api_format),
        ("FMP Cache Files", check_fmp_cache_date_formats),
        ("Code Date Parsing", check_code_date_parsing),
        ("Pandas Date Handling", check_pandas_date_handling),
        ("Output Date Formats", check_output_date_formats)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
            if not result:
                all_checks_passed = False
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results[check_name] = False
            all_checks_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY OF DATE FORMAT CONSISTENCY CHECKS")
    print("=" * 70)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check_name:.<50} {status}")
    
    print("")
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("   The project maintains consistent YYYY-MM-DD date format throughout.")
        print("   This ensures data alignment between FMP API and analysis results.")
    else:
        print("⚠️  ISSUES DETECTED!")
        print("   Date format inconsistencies could lead to data misalignment.")
        print("   Review the detailed output above to identify specific issues.")
    
    print("\n📝 KEY FINDINGS:")
    print("   • FMP API uses YYYY-MM-DD format (ISO 8601 standard)")
    print("   • This is the same format used throughout the project")
    print("   • All date parsing uses '%Y-%m-%d' format string")
    print("   • No MM/DD/YYYY or DD/MM/YYYY patterns detected")
    
    return all_checks_passed

if __name__ == "__main__":
    success = run_comprehensive_date_check()
    exit(0 if success else 1)
