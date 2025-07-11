"""
Test Script for Enhanced Chart Configuration
Test the new chart functionality with enhanced configuration display and filenames
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Test the chart generation functionality
print("🧪 Testing Enhanced Chart Configuration...")

# Test the generate_chart_filename function
import nasdaq_fmp_analysis_corrected as main

# Test config parameters
test_config = {
    'start_date': '2020-01-01',
    'end_date': '2023-12-31',
    'threshold': 0.02,
    'min_top_n': 1,
    'max_top_n': 3,
    'data_source': 'FMP',
    'chart_mode': 'simple'
}

# Test filename generation
filename = main.generate_chart_filename("test", test_config)
print(f"✅ Generated filename: {filename}")

# Test config text panel generation
config_text = main.create_config_text_panel()
print(f"✅ Generated config text panel:")
print(config_text)

print("\n✅ Enhanced chart functionality validation complete!")
