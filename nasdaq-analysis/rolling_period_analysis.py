#!/usr/bin/env python3
"""
Rolling Period Analysis for NASDAQ Top-N Momentum Strategy

This module performs comprehensive rolling period analysis to evaluate strategy performance
across different time periods and market conditions.

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

class RollingPeriodAnalyzer:
    """Analyze strategy performance across rolling time periods"""
    
    def __init__(self, results_df: pd.DataFrame, metrics: Dict):
        self.results_df = results_df.copy()
        self.metrics = metrics
        self.results_df['date'] = pd.to_datetime(self.results_df['date'])
        
    def analyze_rolling_periods(self, window_years: List[int] = [1, 2, 3, 5]) -> Dict:
        """Analyze performance across different rolling time windows"""
        
        rolling_results = {}
        
        for years in window_years:
            print(f"📊 Analyzing {years}-year rolling periods...")
            
            window_days = years * 365
            period_results = []
            
            # Generate rolling windows
            start_date = self.results_df['date'].min()
            end_date = self.results_df['date'].max()
            
            current_start = start_date
            while current_start + timedelta(days=window_days) <= end_date:
                window_end = current_start + timedelta(days=window_days)
                
                # Get data for this window
                window_mask = (self.results_df['date'] >= current_start) & \
                             (self.results_df['date'] <= window_end)
                window_data = self.results_df[window_mask].copy()
                
                if len(window_data) < 4:  # Need minimum data points
                    current_start += timedelta(days=30)  # Move window by 1 month
                    continue
                
                # Calculate metrics for this window
                window_metrics = self._calculate_window_metrics(window_data, years)
                window_metrics['start_date'] = current_start
                window_metrics['end_date'] = window_end
                
                period_results.append(window_metrics)
                current_start += timedelta(days=30)  # Move window by 1 month
            
            rolling_results[f'{years}_year'] = period_results
            print(f"✅ Completed {len(period_results)} rolling {years}-year periods")
        
        return rolling_results
    
    def _calculate_window_metrics(self, window_data: pd.DataFrame, years: int) -> Dict:
        """Calculate performance metrics for a specific time window"""
        
        if len(window_data) < 2:
            return {
                'total_return': np.nan,
                'annualized_return': np.nan,
                'volatility': np.nan,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan,
                'num_periods': 0
            }
        
        # Calculate returns
        returns = window_data['period_return'].dropna()
        
        if len(returns) == 0:
            return {
                'total_return': np.nan,
                'annualized_return': np.nan,
                'volatility': np.nan,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan,
                'num_periods': 0
            }
        
        # Portfolio values for drawdown calculation
        portfolio_values = window_data['portfolio_value'].tolist()
        
        # Calculate metrics
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        annualized_return = ((portfolio_values[-1] / portfolio_values[0]) ** (1/years) - 1) * 100
        
        volatility = returns.std() * np.sqrt(252 / 30) * 100  # Assuming monthly rebalancing
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 / 30) if returns.std() > 0 else 0
        
        max_drawdown = self._calculate_max_drawdown(portfolio_values) * 100
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_periods': len(returns)
        }
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0.0
        
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def create_rolling_analysis_report(self, rolling_results: Dict) -> pd.DataFrame:
        """Create comprehensive rolling analysis report"""
        
        all_periods = []
        
        for window_type, periods in rolling_results.items():
            for period in periods:
                period_data = period.copy()
                period_data['window_type'] = window_type
                all_periods.append(period_data)
        
        report_df = pd.DataFrame(all_periods)
        
        # Summary statistics by window type
        print("\n📊 ROLLING PERIOD ANALYSIS SUMMARY")
        print("=" * 80)
        
        for window_type in report_df['window_type'].unique():
            window_data = report_df[report_df['window_type'] == window_type]
            
            print(f"\n{window_type.replace('_', '-').upper()} WINDOWS:")
            print(f"   Annualized Return: {window_data['annualized_return'].mean():.2f}% ± {window_data['annualized_return'].std():.2f}%")
            print(f"   Volatility: {window_data['volatility'].mean():.2f}% ± {window_data['volatility'].std():.2f}%")
            print(f"   Sharpe Ratio: {window_data['sharpe_ratio'].mean():.3f} ± {window_data['sharpe_ratio'].std():.3f}")
            print(f"   Max Drawdown: {window_data['max_drawdown'].mean():.2f}% ± {window_data['max_drawdown'].std():.2f}%")
            print(f"   Number of Periods: {len(window_data)}")
        
        return report_df
    
    def visualize_rolling_analysis(self, rolling_results: Dict, save_path: str = None):
        """Create comprehensive rolling analysis visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Rolling Period Performance Analysis', fontsize=16, fontweight='bold')
        
        # Colors for different window types
        colors = ['steelblue', 'lightcoral', 'green', 'orange']
        
        # 1. Rolling Annualized Returns
        ax1 = axes[0, 0]
        for i, (window_type, periods) in enumerate(rolling_results.items()):
            if not periods:
                continue
            
            df = pd.DataFrame(periods)
            dates = pd.to_datetime(df['start_date'])
            returns = df['annualized_return']
            
            ax1.plot(dates, returns, label=f'{window_type.replace("_", "-")} windows',
                    color=colors[i % len(colors)], linewidth=2, alpha=0.8)
        
        ax1.set_title('Rolling Annualized Returns', fontweight='bold')
        ax1.set_xlabel('Period Start Date')
        ax1.set_ylabel('Annualized Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling Sharpe Ratios
        ax2 = axes[0, 1]
        for i, (window_type, periods) in enumerate(rolling_results.items()):
            if not periods:
                continue
            
            df = pd.DataFrame(periods)
            dates = pd.to_datetime(df['start_date'])
            sharpe = df['sharpe_ratio']
            
            ax2.plot(dates, sharpe, label=f'{window_type.replace("_", "-")} windows',
                    color=colors[i % len(colors)], linewidth=2, alpha=0.8)
        
        ax2.set_title('Rolling Sharpe Ratios', fontweight='bold')
        ax2.set_xlabel('Period Start Date')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 3. Rolling Maximum Drawdowns
        ax3 = axes[1, 0]
        for i, (window_type, periods) in enumerate(rolling_results.items()):
            if not periods:
                continue
            
            df = pd.DataFrame(periods)
            dates = pd.to_datetime(df['start_date'])
            drawdowns = df['max_drawdown']
            
            ax3.plot(dates, drawdowns, label=f'{window_type.replace("_", "-")} windows',
                    color=colors[i % len(colors)], linewidth=2, alpha=0.8)
        
        ax3.set_title('Rolling Maximum Drawdowns', fontweight='bold')
        ax3.set_xlabel('Period Start Date')
        ax3.set_ylabel('Max Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Risk-Return Scatter
        ax4 = axes[1, 1]
        for i, (window_type, periods) in enumerate(rolling_results.items()):
            if not periods:
                continue
            
            df = pd.DataFrame(periods)
            returns = df['annualized_return']
            volatility = df['volatility']
            
            ax4.scatter(volatility, returns, label=f'{window_type.replace("_", "-")} windows',
                       color=colors[i % len(colors)], alpha=0.6, s=50)
        
        ax4.set_title('Risk-Return Profile', fontweight='bold')
        ax4.set_xlabel('Volatility (%)')
        ax4.set_ylabel('Annualized Return (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Rolling analysis chart saved to: {save_path}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rolling_period_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Rolling analysis chart saved as: {filename}")
        
        plt.show()
        plt.close()

def analyze_strategy_robustness(results_df: pd.DataFrame, metrics: Dict) -> Dict:
    """Perform comprehensive robustness analysis of the strategy"""
    
    print("🔍 Starting Strategy Robustness Analysis...")
    
    analyzer = RollingPeriodAnalyzer(results_df, metrics)
    
    # Perform rolling period analysis
    rolling_results = analyzer.analyze_rolling_periods([1, 2, 3, 5])
    
    # Create comprehensive report
    report_df = analyzer.create_rolling_analysis_report(rolling_results)
    
    # Create visualizations
    analyzer.visualize_rolling_analysis(rolling_results)
    
    # Calculate robustness metrics
    robustness_metrics = {}
    
    for window_type in report_df['window_type'].unique():
        window_data = report_df[report_df['window_type'] == window_type]
        
        # Consistency metrics
        positive_periods = (window_data['total_return'] > 0).sum()
        total_periods = len(window_data)
        win_rate = positive_periods / total_periods * 100 if total_periods > 0 else 0
        
        robustness_metrics[window_type] = {
            'win_rate': win_rate,
            'return_consistency': 1 / (window_data['annualized_return'].std() + 1e-6),
            'sharpe_consistency': 1 / (window_data['sharpe_ratio'].std() + 1e-6),
            'worst_period_return': window_data['total_return'].min(),
            'best_period_return': window_data['total_return'].max()
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_df.to_csv(f"rolling_analysis_report_{timestamp}.csv", index=False)
    
    print("✅ Robustness Analysis Complete!")
    
    return {
        'rolling_results': rolling_results,
        'report_df': report_df,
        'robustness_metrics': robustness_metrics
    }

if __name__ == "__main__":
    print("📊 Rolling Period Analysis Module")
    print("This module should be imported and used with existing strategy results.")
