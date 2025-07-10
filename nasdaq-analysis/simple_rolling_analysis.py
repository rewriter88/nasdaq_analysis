#!/usr/bin/env python3
"""
Simple Rolling Analysis for NASDAQ Top-N Momentum Strategy

A simplified version of rolling period analysis for quick performance evaluation
across different time windows.

Author: Enhanced by AI Assistant
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import warnings
warnings.filterwarnings('ignore')

class SimpleRollingAnalyzer:
    """Simple rolling period performance analyzer"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df.copy()
        self.results_df['date'] = pd.to_datetime(self.results_df['date'])
        
    def calculate_simple_rolling_metrics(self, window_months: int = 12) -> Dict:
        """Calculate simple rolling metrics for a given window"""
        
        print(f"📊 Calculating {window_months}-month rolling metrics...")
        
        # Sort by date
        df = self.results_df.sort_values('date').copy()
        
        # Calculate rolling returns
        df['rolling_return'] = df['portfolio_value'].pct_change(periods=window_months)
        df['rolling_volatility'] = df['portfolio_value'].pct_change().rolling(window_months).std() * np.sqrt(252)
        df['rolling_sharpe'] = (df['portfolio_value'].pct_change().rolling(window_months).mean() * 252) / df['rolling_volatility']
        
        # Calculate rolling max drawdown
        rolling_peak = df['portfolio_value'].rolling(window_months).max()
        rolling_drawdown = (df['portfolio_value'] - rolling_peak) / rolling_peak
        df['rolling_max_drawdown'] = rolling_drawdown.rolling(window_months).min()
        
        # Summary statistics
        metrics = {
            'avg_rolling_return': df['rolling_return'].mean(),
            'avg_rolling_volatility': df['rolling_volatility'].mean(),
            'avg_rolling_sharpe': df['rolling_sharpe'].mean(),
            'avg_rolling_max_drawdown': df['rolling_max_drawdown'].mean(),
            'best_rolling_period': {
                'date': df.loc[df['rolling_return'].idxmax(), 'date'],
                'return': df['rolling_return'].max()
            },
            'worst_rolling_period': {
                'date': df.loc[df['rolling_return'].idxmin(), 'date'],
                'return': df['rolling_return'].min()
            }
        }
        
        return metrics, df
    
    def analyze_multiple_windows(self, windows: List[int] = [6, 12, 24, 36]) -> Dict:
        """Analyze performance across multiple rolling windows"""
        
        results = {}
        
        for window in windows:
            print(f"\n🔄 Analyzing {window}-month rolling window...")
            metrics, df = self.calculate_simple_rolling_metrics(window)
            results[f"{window}_months"] = {
                'metrics': metrics,
                'data': df[['date', 'portfolio_value', 'rolling_return', 
                           'rolling_volatility', 'rolling_sharpe', 'rolling_max_drawdown']].copy()
            }
        
        return results
    
    def plot_rolling_analysis(self, results: Dict, save_path: Optional[str] = None):
        """Plot rolling analysis results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Simple Rolling Period Analysis', fontsize=16, fontweight='bold')
        
        # Colors for different windows
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        # Plot 1: Rolling Returns
        ax1 = axes[0, 0]
        for i, (window, data) in enumerate(results.items()):
            df = data['data'].dropna()
            ax1.plot(df['date'], df['rolling_return'], 
                    label=window.replace('_', ' '), color=colors[i % len(colors)], alpha=0.7)
        
        ax1.set_title('Rolling Returns by Window')
        ax1.set_ylabel('Rolling Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rolling Sharpe Ratio
        ax2 = axes[0, 1]
        for i, (window, data) in enumerate(results.items()):
            df = data['data'].dropna()
            ax2.plot(df['date'], df['rolling_sharpe'], 
                    label=window.replace('_', ' '), color=colors[i % len(colors)], alpha=0.7)
        
        ax2.set_title('Rolling Sharpe Ratio by Window')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Rolling Volatility
        ax3 = axes[1, 0]
        for i, (window, data) in enumerate(results.items()):
            df = data['data'].dropna()
            ax3.plot(df['date'], df['rolling_volatility'], 
                    label=window.replace('_', ' '), color=colors[i % len(colors)], alpha=0.7)
        
        ax3.set_title('Rolling Volatility by Window')
        ax3.set_ylabel('Annualized Volatility')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Rolling Max Drawdown
        ax4 = axes[1, 1]
        for i, (window, data) in enumerate(results.items()):
            df = data['data'].dropna()
            ax4.plot(df['date'], df['rolling_max_drawdown'], 
                    label=window.replace('_', ' '), color=colors[i % len(colors)], alpha=0.7)
        
        ax4.set_title('Rolling Max Drawdown by Window')
        ax4.set_ylabel('Max Drawdown')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Chart saved to: {save_path}")
        
        plt.show()
        
    def generate_summary_report(self, results: Dict) -> str:
        """Generate a summary report of rolling analysis"""
        
        report = "=" * 80 + "\n"
        report += "SIMPLE ROLLING PERIOD ANALYSIS SUMMARY\n"
        report += "=" * 80 + "\n\n"
        
        for window, data in results.items():
            metrics = data['metrics']
            
            report += f"{window.replace('_', ' ').title()} Rolling Analysis:\n"
            report += "-" * 40 + "\n"
            report += f"Average Rolling Return: {metrics['avg_rolling_return']:.2%}\n"
            report += f"Average Rolling Volatility: {metrics['avg_rolling_volatility']:.2%}\n"
            report += f"Average Rolling Sharpe: {metrics['avg_rolling_sharpe']:.2f}\n"
            report += f"Average Rolling Max Drawdown: {metrics['avg_rolling_max_drawdown']:.2%}\n"
            report += f"Best Period: {metrics['best_rolling_period']['date'].strftime('%Y-%m-%d')} "
            report += f"({metrics['best_rolling_period']['return']:.2%})\n"
            report += f"Worst Period: {metrics['worst_rolling_period']['date'].strftime('%Y-%m-%d')} "
            report += f"({metrics['worst_rolling_period']['return']:.2%})\n\n"
        
        return report

def run_simple_rolling_analysis(results_df: pd.DataFrame, 
                               windows: List[int] = [6, 12, 24, 36],
                               save_chart: bool = True) -> Dict:
    """
    Run complete simple rolling analysis
    
    Args:
        results_df: DataFrame with portfolio performance data
        windows: List of rolling window sizes in months
        save_chart: Whether to save the analysis chart
    
    Returns:
        Dictionary with rolling analysis results
    """
    
    print("🔄 Starting Simple Rolling Period Analysis...")
    
    analyzer = SimpleRollingAnalyzer(results_df)
    results = analyzer.analyze_multiple_windows(windows)
    
    # Generate and print summary report
    report = analyzer.generate_summary_report(results)
    print(report)
    
    # Create chart
    if save_chart:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        chart_path = f"results/charts/simple_rolling_analysis_{timestamp}.png"
        os.makedirs("results/charts", exist_ok=True)
        analyzer.plot_rolling_analysis(results, chart_path)
    else:
        analyzer.plot_rolling_analysis(results)
    
    return results

if __name__ == "__main__":
    # Example usage - this would typically be called from main analysis scripts
    print("Simple Rolling Analysis module loaded successfully!")
    print("Use run_simple_rolling_analysis(results_df) to analyze portfolio performance.")
