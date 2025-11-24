'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-11-23 17:27:40
LastEditTime: 2025-11-23 20:33:30
FilePath: /Unsupervised-Learning-Homework/Homework 3/Code/hw3_utils.py
Description: 
    process_stock_data: 处理数据，根据 verbose 参数决定是否画图。

    fit_glasso_models: 拟合标准 Glasso 和非参数 Glasso，同样由 verbose 控制输出。
'''
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler

class Prob1Analysis:
    def __init__(self,
                 output_dir='Homework 3/Code/Data',
                 figure_dir='Homework 3/Latex/Figures',
                 start_date="2021-01-01"):
        self.output_dir = output_dir
        self.figure_dir = figure_dir
        self.start_date = start_date
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        
        self.raw_data_file = os.path.join(self.output_dir, 'raw_stock_data.csv')
        self.log_returns_file = os.path.join(self.output_dir, 'log_returns.csv')
        
        self.tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", 
                        "JPM", "BAC", "XOM", "CVX", "JNJ", "PFE", "WMT", "PG", "KO"]
        
        self.sectors = {
            'Tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
            'Finance': ['JPM', 'BAC'],
            'Energy': ['XOM', 'CVX'],
            'Healthcare': ['JNJ', 'PFE'],
            'Consumer': ['WMT', 'PG', 'KO']
        }
        
        self.log_returns = None
        self.glasso_results = None

    def process_stock_data(self, verbose=False):
        # Download stock data and compute log returns
        if os.path.exists(self.log_returns_file):
            log_returns = pd.read_csv(self.log_returns_file, index_col=0, parse_dates=True)
        
        else:
            print(f"Downloading data from yfinance...")
            raw_data = yf.download(self.tickers, start=self.start_date, end=None)
            raw_data.to_csv(self.raw_data_file)
            
            # Extract closing prices and compute log returns
            close_prices = raw_data['Close']
            print(f"\nData shape: {close_prices.shape}")
            print(f"Date range: {close_prices.index[0]} to {close_prices.index[-1]}")

            log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
            log_returns.to_csv(self.log_returns_file)
            # print(f"\nMissing values per stock:\n{log_returns.isnull().sum()}")
        
        self.log_returns = log_returns

        # Visual exploration
        if verbose:
            self._plot_visual_exploration()
            
        return log_returns
    
    def _plot_visual_exploration(self):
        # Summary statistics
        summary_stats = self.log_returns.describe()
        summary_stats.to_csv(os.path.join(self.output_dir, 'summary_statistics.csv'))

        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        corr_matrix = self.log_returns.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    mask=mask, square=True, linewidths=0.5)
        plt.title('Correlation Matrix of Log Returns', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '1a_correlation_heatmap.png'), dpi=150)
        plt.show()
        
        # Plot cumulative returns
        plt.figure(figsize=(14, 8))
        cumulative_returns = (1 + self.log_returns).cumprod()
        cumulative_returns.plot(alpha=0.8)
        plt.title('Cumulative Returns (Jan 2021 - Present)', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '1a_cumulative_returns.png'), dpi=150)
        plt.show()

        # Relationships within different industries
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (sector, stocks) in enumerate(self.sectors.items()):
            sector_corr = self.log_returns[stocks].corr()
            sns.heatmap(sector_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                        ax=axes[idx], square=True, vmin=-1, vmax=1)
            axes[idx].set_title(f'{sector} Sector Correlation')

        axes[-1].axis('off')
        plt.suptitle('Within-Sector Correlation Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '1a_sector_correlations.png'), dpi=150)
        plt.show()
        
    def fit_glasso_models(self, verbose=False):
        if self.log_returns is None:
            self.process_stock_data()
            
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(self.log_returns.values)
        
        # Grid search best alpha values
        alphas_grid = np.logspace(np.log10(0.01), np.log10(0.8), 30)
        
        # Standard Graphical Lasso
        if verbose:
            print("Fitting Standard Graphical LassoCV...")
        gl_cv = GraphicalLassoCV(alphas=alphas_grid,cv=5, n_jobs=-1, max_iter=3000,tol=1e-4)
        gl_cv.fit(X_standardized)
        # print(gl_cv.cv_results_)
        
        # Non-parametric (Rank-based) Graphical Lasso
        if verbose:
            print("Fitting Non-parametric (Rank) Graphical LassoCV...")
            
        # Transform data to ranks
        n = self.log_returns.shape[0]
        X_ranks = self.log_returns.rank() / (n + 1)
        X_normal_scores = norm.ppf(X_ranks)
        X_np_standardized = StandardScaler().fit_transform(X_normal_scores)
        
        gl_np = GraphicalLassoCV(alphas=alphas_grid,cv=5, n_jobs=-1, max_iter=3000,tol=1e-4)
        gl_np.fit(X_np_standardized)
        
        # Compute partial correlations
        prec = gl_cv.precision_
        d = np.sqrt(np.diag(prec))
        partial_corr = -prec / np.outer(d, d)
        np.fill_diagonal(partial_corr, 1.0)
        
        prec_np = gl_np.precision_
        d_np = np.sqrt(np.diag(prec_np))
        partial_corr_np = -prec_np / np.outer(d_np, d_np)
        np.fill_diagonal(partial_corr_np, 1.0)
        
        self.glasso_results = {
            'gl_cv': gl_cv,
            'gl_np': gl_np,
            'precision_gl': prec,
            'precision_np': prec_np,
            'partial_corr_gl': partial_corr,
            'partial_corr_np': partial_corr_np
        }
        
        # Verbose output and visualization
        if verbose:
            self._plot_glasso_results(gl_cv,gl_np,partial_corr,partial_corr_np)
            
        
        return self.glasso_results
        
    def _plot_glasso_results(self, gl_cv, gl_np, partial_corr, partial_corr_np):
        print("\n--- Graphical Lasso Results ---")
        print(f"Standard GL Best Alpha: {gl_cv.alpha_:.6f}")
        print(f"Non-Parametric GL Best Alpha: {gl_np.alpha_:.6f}")
        
        # Plot CV curves
        plt.figure(figsize=(10, 5))
        cv_results = gl_cv.cv_results_
        cv_means = cv_results['mean_test_score']
        cv_alphas = cv_results['alphas']
        
        plt.semilogx(cv_alphas, cv_means, 'o-', label='Standard GL CV Score')
        plt.axvline(gl_cv.alpha_, linestyle='--', color='r', label=f'Best Alpha: {gl_cv.alpha_:.4f}')
        
        plt.xlabel('Regularization Parameter (Alpha/Lambda)')
        plt.ylabel('Cross-Validation Score (Log Likelihood)')
        plt.title('Hyperparameter Tuning Path: Selecting Sparsity')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '1b_glasso_cv_curve.png'), dpi=150)
        plt.show()
        
        # Plot precision matrices
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Standard
        prec = gl_cv.precision_
        d = np.sqrt(np.diag(prec))
        partial_corr = -prec / np.outer(d, d)
        np.fill_diagonal(partial_corr, 1.0)
        
        sns.heatmap(partial_corr, ax=axes[0], cmap='coolwarm', 
                    xticklabels=self.tickers, yticklabels=self.tickers, vmin=-1, vmax=1)
        axes[0].set_title(f'Standard GL Precision Matrix\n(alpha={gl_cv.alpha_:.4f})')
        
        # Non-parametric
        prec_np = gl_np.precision_
        d_np = np.sqrt(np.diag(prec_np))
        partial_corr_np = -prec_np / np.outer(d_np, d_np)
        np.fill_diagonal(partial_corr_np, 1.0)
        
        sns.heatmap(partial_corr_np, ax=axes[1], cmap='coolwarm', 
                    xticklabels=self.tickers, yticklabels=self.tickers, vmin=-1, vmax=1)
        axes[1].set_title(f'Non-Parametric GL Precision Matrix\n(alpha={gl_np.alpha_:.4f})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '1b_glasso_precision_matrices.png'), dpi=150)
        plt.show()