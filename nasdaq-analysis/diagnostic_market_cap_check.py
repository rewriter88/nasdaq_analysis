#!/usr/bin/env python3
"""
Diagnostic Market Cap Check for NASDAQ Analysis

This module provides diagnostic tools to check market cap data quality,
consistency, and identify potential issues with the data sources.

Author: Enhanced by AI Assistant
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MarketCapDiagnostics:
    """Diagnostic tools for market cap data analysis"""
    
    def __init__(self, market_cap_data: Dict):
        self.market_cap_data = market_cap_data
        self.symbols = list(market_cap_data.keys())
        
    def check_data_completeness(self) -> Dict:
        """Check completeness of market cap data across symbols and dates"""
        
        print("🔍 Checking market cap data completeness...")
        
        completeness_report = {
            'total_symbols': len(self.symbols),
            'symbols_with_data': 0,
            'symbols_without_data': [],
            'date_coverage': {},
            'missing_data_summary': {}
        }
        
        all_dates = set()
        symbol_date_counts = {}
        
        for symbol in self.symbols:
            data = self.market_cap_data.get(symbol, [])
            
            if data:
                completeness_report['symbols_with_data'] += 1
                dates = [item['date'] for item in data if 'date' in item]
                symbol_date_counts[symbol] = len(dates)
                all_dates.update(dates)
            else:
                completeness_report['symbols_without_data'].append(symbol)
                symbol_date_counts[symbol] = 0
        
        # Analyze date coverage
        if all_dates:
            min_date = min(all_dates)
            max_date = max(all_dates)
            total_dates = len(all_dates)
            
            completeness_report['date_coverage'] = {
                'earliest_date': min_date,
                'latest_date': max_date,
                'total_unique_dates': total_dates,
                'average_dates_per_symbol': np.mean(list(symbol_date_counts.values())),
                'min_dates_per_symbol': min(symbol_date_counts.values()),
                'max_dates_per_symbol': max(symbol_date_counts.values())
            }
        
        # Missing data summary
        completeness_report['missing_data_summary'] = {
            'symbols_with_no_data': len(completeness_report['symbols_without_data']),
            'percentage_complete': (completeness_report['symbols_with_data'] / 
                                  completeness_report['total_symbols']) * 100
        }
        
        return completeness_report
    
    def analyze_market_cap_distributions(self) -> Dict:
        """Analyze market cap value distributions and identify outliers"""
        
        print("📊 Analyzing market cap distributions...")
        
        all_market_caps = []
        symbol_stats = {}
        
        for symbol in self.symbols:
            data = self.market_cap_data.get(symbol, [])
            
            if data:
                market_caps = []
                for item in data:
                    if 'marketCap' in item and item['marketCap'] is not None:
                        try:
                            market_cap = float(item['marketCap'])
                            if market_cap > 0:  # Filter out zero/negative values
                                market_caps.append(market_cap)
                                all_market_caps.append(market_cap)
                        except (ValueError, TypeError):
                            continue
                
                if market_caps:
                    symbol_stats[symbol] = {
                        'count': len(market_caps),
                        'mean': np.mean(market_caps),
                        'median': np.median(market_caps),
                        'min': np.min(market_caps),
                        'max': np.max(market_caps),
                        'std': np.std(market_caps),
                        'cv': np.std(market_caps) / np.mean(market_caps)  # Coefficient of variation
                    }
        
        # Overall distribution statistics
        distribution_stats = {
            'total_observations': len(all_market_caps),
            'overall_mean': np.mean(all_market_caps) if all_market_caps else 0,
            'overall_median': np.median(all_market_caps) if all_market_caps else 0,
            'overall_std': np.std(all_market_caps) if all_market_caps else 0,
            'percentiles': {
                '5th': np.percentile(all_market_caps, 5) if all_market_caps else 0,
                '25th': np.percentile(all_market_caps, 25) if all_market_caps else 0,
                '75th': np.percentile(all_market_caps, 75) if all_market_caps else 0,
                '95th': np.percentile(all_market_caps, 95) if all_market_caps else 0
            }
        }
        
        return {
            'symbol_stats': symbol_stats,
            'distribution_stats': distribution_stats,
            'all_market_caps': all_market_caps
        }
    
    def detect_anomalies(self, z_threshold: float = 3.0) -> Dict:
        """Detect anomalies in market cap data using statistical methods"""
        
        print(f"🚨 Detecting anomalies with Z-score threshold: {z_threshold}")
        
        anomalies = {
            'extreme_values': [],
            'missing_data_patterns': [],
            'suspicious_changes': []
        }
        
        for symbol in self.symbols:
            data = self.market_cap_data.get(symbol, [])
            
            if not data:
                continue
                
            # Extract market cap values with dates
            time_series = []
            for item in data:
                if 'marketCap' in item and 'date' in item and item['marketCap'] is not None:
                    try:
                        market_cap = float(item['marketCap'])
                        if market_cap > 0:
                            time_series.append({
                                'date': item['date'],
                                'market_cap': market_cap
                            })
                    except (ValueError, TypeError):
                        continue
            
            if len(time_series) < 2:
                continue
                
            # Sort by date
            time_series.sort(key=lambda x: x['date'])
            market_caps = [item['market_cap'] for item in time_series]
            
            # Detect extreme values using Z-score
            if len(market_caps) > 1:
                z_scores = np.abs((market_caps - np.mean(market_caps)) / np.std(market_caps))
                extreme_indices = np.where(z_scores > z_threshold)[0]
                
                for idx in extreme_indices:
                    anomalies['extreme_values'].append({
                        'symbol': symbol,
                        'date': time_series[idx]['date'],
                        'market_cap': time_series[idx]['market_cap'],
                        'z_score': z_scores[idx]
                    })
            
            # Detect suspicious changes (large percentage changes)
            for i in range(1, len(time_series)):
                prev_cap = time_series[i-1]['market_cap']
                curr_cap = time_series[i]['market_cap']
                
                pct_change = abs((curr_cap - prev_cap) / prev_cap)
                
                # Flag changes > 50% as suspicious
                if pct_change > 0.5:
                    anomalies['suspicious_changes'].append({
                        'symbol': symbol,
                        'date': time_series[i]['date'],
                        'prev_date': time_series[i-1]['date'],
                        'prev_market_cap': prev_cap,
                        'curr_market_cap': curr_cap,
                        'pct_change': pct_change
                    })
        
        return anomalies
    
    def plot_market_cap_diagnostics(self, analysis_results: Dict, 
                                   save_path: Optional[str] = None):
        """Create diagnostic plots for market cap data"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Market Cap Data Diagnostics', fontsize=16, fontweight='bold')
        
        # Plot 1: Market cap distribution
        ax1 = axes[0, 0]
        market_caps = analysis_results['all_market_caps']
        if market_caps:
            # Log scale for better visualization
            log_caps = np.log10([cap for cap in market_caps if cap > 0])
            ax1.hist(log_caps, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_xlabel('Log10(Market Cap)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Market Cap Distribution (Log Scale)')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Symbols by data completeness
        ax2 = axes[0, 1]
        symbol_stats = analysis_results['symbol_stats']
        if symbol_stats:
            counts = [stats['count'] for stats in symbol_stats.values()]
            symbols = list(symbol_stats.keys())
            
            # Show top 20 symbols with most data
            top_indices = np.argsort(counts)[-20:]
            top_symbols = [symbols[i] for i in top_indices]
            top_counts = [counts[i] for i in top_indices]
            
            y_pos = np.arange(len(top_symbols))
            ax2.barh(y_pos, top_counts, alpha=0.7, color='green')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(top_symbols, fontsize=8)
            ax2.set_xlabel('Number of Data Points')
            ax2.set_title('Top 20 Symbols by Data Completeness')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Market cap variability (CV)
        ax3 = axes[1, 0]
        if symbol_stats:
            cvs = [stats['cv'] for stats in symbol_stats.values()]
            ax3.hist(cvs, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax3.set_xlabel('Coefficient of Variation')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Market Cap Variability Distribution')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Data completeness summary
        ax4 = axes[1, 1]
        completeness = analysis_results.get('completeness', {})
        if completeness:
            labels = ['With Data', 'Without Data']
            sizes = [
                completeness.get('symbols_with_data', 0),
                len(completeness.get('symbols_without_data', []))
            ]
            colors = ['green', 'red']
            
            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Data Completeness by Symbol')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Diagnostic plots saved to: {save_path}")
        
        plt.show()
    
    def generate_diagnostic_report(self, completeness: Dict, 
                                 analysis: Dict, anomalies: Dict) -> str:
        """Generate a comprehensive diagnostic report"""
        
        report = "=" * 80 + "\n"
        report += "MARKET CAP DATA DIAGNOSTIC REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Completeness section
        report += "DATA COMPLETENESS ANALYSIS\n"
        report += "-" * 40 + "\n"
        report += f"Total Symbols: {completeness['total_symbols']}\n"
        report += f"Symbols with Data: {completeness['symbols_with_data']}\n"
        report += f"Symbols without Data: {len(completeness['symbols_without_data'])}\n"
        report += f"Completeness Rate: {completeness['missing_data_summary']['percentage_complete']:.1f}%\n\n"
        
        if completeness['date_coverage']:
            coverage = completeness['date_coverage']
            report += f"Date Range: {coverage['earliest_date']} to {coverage['latest_date']}\n"
            report += f"Total Unique Dates: {coverage['total_unique_dates']}\n"
            report += f"Avg Dates per Symbol: {coverage['average_dates_per_symbol']:.1f}\n\n"
        
        # Distribution analysis
        report += "DISTRIBUTION ANALYSIS\n"
        report += "-" * 40 + "\n"
        dist_stats = analysis['distribution_stats']
        report += f"Total Observations: {dist_stats['total_observations']:,}\n"
        report += f"Mean Market Cap: ${dist_stats['overall_mean']:,.0f}\n"
        report += f"Median Market Cap: ${dist_stats['overall_median']:,.0f}\n"
        report += f"Standard Deviation: ${dist_stats['overall_std']:,.0f}\n\n"
        
        percentiles = dist_stats['percentiles']
        report += "Market Cap Percentiles:\n"
        for p, value in percentiles.items():
            report += f"  {p}: ${value:,.0f}\n"
        report += "\n"
        
        # Anomaly detection
        report += "ANOMALY DETECTION\n"
        report += "-" * 40 + "\n"
        report += f"Extreme Values Detected: {len(anomalies['extreme_values'])}\n"
        report += f"Suspicious Changes Detected: {len(anomalies['suspicious_changes'])}\n\n"
        
        if anomalies['extreme_values']:
            report += "Top 5 Extreme Values:\n"
            sorted_extreme = sorted(anomalies['extreme_values'], 
                                  key=lambda x: x['z_score'], reverse=True)
            for item in sorted_extreme[:5]:
                report += f"  {item['symbol']} on {item['date']}: "
                report += f"${item['market_cap']:,.0f} (Z-score: {item['z_score']:.2f})\n"
            report += "\n"
        
        if anomalies['suspicious_changes']:
            report += "Top 5 Suspicious Changes:\n"
            sorted_changes = sorted(anomalies['suspicious_changes'], 
                                  key=lambda x: x['pct_change'], reverse=True)
            for item in sorted_changes[:5]:
                report += f"  {item['symbol']} on {item['date']}: "
                report += f"{item['pct_change']:.1%} change\n"
            report += "\n"
        
        return report

def run_market_cap_diagnostics(market_cap_data: Dict, 
                             save_plots: bool = True) -> Dict:
    """
    Run complete market cap diagnostics
    
    Args:
        market_cap_data: Dictionary with market cap data by symbol
        save_plots: Whether to save diagnostic plots
    
    Returns:
        Dictionary with diagnostic results
    """
    
    print("🔍 Starting Market Cap Data Diagnostics...")
    
    diagnostics = MarketCapDiagnostics(market_cap_data)
    
    # Run all diagnostic checks
    completeness = diagnostics.check_data_completeness()
    analysis = diagnostics.analyze_market_cap_distributions()
    anomalies = diagnostics.detect_anomalies()
    
    # Combine results
    results = {
        'completeness': completeness,
        'analysis': analysis,
        'anomalies': anomalies
    }
    
    # Add analysis to results for plotting
    results['symbol_stats'] = analysis['symbol_stats']
    results['all_market_caps'] = analysis['all_market_caps']
    
    # Generate and print report
    report = diagnostics.generate_diagnostic_report(completeness, analysis, anomalies)
    print(report)
    
    # Create diagnostic plots
    if save_plots:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = f"results/charts/market_cap_diagnostics_{timestamp}.png"
        import os
        os.makedirs("results/charts", exist_ok=True)
        diagnostics.plot_market_cap_diagnostics(results, plot_path)
    else:
        diagnostics.plot_market_cap_diagnostics(results)
    
    return results

if __name__ == "__main__":
    # Example usage
    print("Market Cap Diagnostics module loaded successfully!")
    print("Use run_market_cap_diagnostics(market_cap_data) to analyze data quality.")
