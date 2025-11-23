'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-11-23 12:04:17
LastEditTime: 2025-11-23 15:36:15
FilePath: /Unsupervised-Learning-Homework/Homework 3/Code/Problem_1.py
Description: 
    Graphical Models.
'''
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np  
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Problem 1a: Download stock data and compute log returns
output_dir = 'Homework 3/Code/Data'
figure_dir = 'Homework 3/Latex/Figures'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

raw_data_file = os.path.join(output_dir, 'raw_stock_data.csv')
log_returns_file = os.path.join(output_dir, 'log_returns.csv')

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "BAC", "XOM", "CVX", "JNJ", "PFE", "WMT", "PG", "KO"]
# 苹果、微软、谷歌、亚马逊、Meta、英伟达、摩根大通、美国银行、埃克森美孚、雪佛龙、强生、辉瑞、沃尔玛、宝洁、可口可乐
start_date = "2021-01-01"

raw_data = yf.download(tickers, start=start_date, end=None)
raw_data.to_csv(raw_data_file)

# Extract closing prices and compute log returns
close_prices = raw_data['Close']
print(f"\nData shape: {close_prices.shape}")
print(f"Date range: {close_prices.index[0]} to {close_prices.index[-1]}")

log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
log_returns.to_csv(log_returns_file)
print(f"\nMissing values per stock:\n{log_returns.isnull().sum()}")

# Visual exploration
# Summary statistics
summary_stats = log_returns.describe()
summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
corr_matrix = log_returns.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            mask=mask, square=True, linewidths=0.5)
plt.title('Correlation Matrix of Log Returns', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, '1a_correlation_heatmap.png'), dpi=150)

# Plot cumulative returns
plt.figure(figsize=(14, 8))
cumulative_returns = (1 + log_returns).cumprod()
cumulative_returns.plot(alpha=0.8)
plt.title('Cumulative Returns (Jan 2021 - Present)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, '1a_cumulative_returns.png'), dpi=150)

# Relationships within different industries
sectors = {
    'Tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
    'Finance': ['JPM', 'BAC'],
    'Energy': ['XOM', 'CVX'],
    'Healthcare': ['JNJ', 'PFE'],
    'Consumer': ['WMT', 'PG', 'KO']
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (sector, stocks) in enumerate(sectors.items()):
    sector_corr = log_returns[stocks].corr()
    sns.heatmap(sector_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=axes[idx], square=True, vmin=-1, vmax=1)
    axes[idx].set_title(f'{sector} Sector Correlation')

axes[-1].axis('off')
plt.suptitle('Within-Sector Correlation Analysis', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, '1a_sector_correlations.png'), dpi=150)

# Problem 1b: Graphical Lasso

