'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-11-23 17:27:40
LastEditTime: 2025-11-26 11:22:20
FilePath: /Unsupervised-Learning-Homework/Homework 3/Code/prob1_utils.py
Description: 
    Utility functions and classes for Problem 1 analysis in Homework 3.
'''
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import networkx as nx
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from statsmodels.tsa.stattools import grangercausalitytests
from causallearn.search.ConstraintBased.PC import pc

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
        
    def calculate_bic_for_dag(self, data, G):
        n_samples, n_features = data.shape
        bic_total = 0

        nodes = G.get_nodes()

        for i, node in enumerate(nodes):
            neighbors = G.get_adjacent_nodes(node)
            neighbor_indices = [nodes.index(n) for n in neighbors]

            y = data.iloc[:, i].values

            if len(neighbor_indices) == 0:
                residuals = y - np.mean(y)
            else:
                X = data.iloc[:, neighbor_indices].values
                model = LinearRegression()
                model.fit(X, y)
                residuals = y - model.predict(X)

            variance = max(1e-9, np.var(residuals))

            # Log-likelihood term for this node
            log_likelihood = -0.5 * n_samples * (np.log(2 * np.pi * variance) + 1)

            # Number of parameters = number of parents + 1 (intercept/variance)
            k = len(neighbor_indices) + 1

            # BIC = k * ln(n) - 2 * ln(L)
            bic_node = k * np.log(n_samples) - 2 * log_likelihood
            bic_total += bic_node

        return bic_total
    
    def fit_pc_model(self, alphas=[0.001, 0.01, 0.05, 0.1, 0.2], verbose=False):
        if self.log_returns is None:
            self.process_stock_data()
        
        data_np = self.log_returns.values
        labels = self.log_returns.columns.tolist()
        
        results = []
        best_bic = float('inf')
        best_graph = None
        best_alpha = 0.05
        
        if verbose:
            print(f"Tuning PC Algorithm over alphas: {alphas}...")
            
        for alpha in alphas:
            cg = pc(data_np, alpha, "fisherz")
            n_edges = cg.G.get_num_edges()
            bic = self.calculate_bic_for_dag(self.log_returns, cg.G)

            results.append({
                'alpha': alpha,
                'n_edges': n_edges,
                'bic': bic,
                'graph': cg.G
            })

            if verbose:
                print(f"  Alpha: {alpha:<6} | Edges: {n_edges:<4} | BIC: {bic:.2f}")

            if bic < best_bic:
                best_bic = bic
                best_graph = cg.G
                best_alpha = alpha

        if verbose:
            self._plot_pc_results(results, best_alpha, best_graph, labels)

        return {
            'best_graph': best_graph,
            'best_alpha': best_alpha,
            'results': results
        }
        
    def _plot_pc_results(self, results, best_alpha, best_graph, labels):
        alphas_plot = [r['alpha'] for r in results]
        edges_plot = [r['n_edges'] for r in results]
        bics_plot = [r['bic'] for r in results]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Alpha (Significance Level)')
        ax1.set_ylabel('Number of Edges', color=color)
        ax1.plot(alphas_plot, edges_plot, marker='o', color=color, label='Sparsity (Edges)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('BIC Score (Lower is Better)', color=color)
        ax2.plot(alphas_plot, bics_plot, marker='x', linestyle='--', color=color, label='BIC Score')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'PC Algorithm Hyperparameter Tuning\nBest Alpha (min BIC): {best_alpha}')
        fig.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '1c_pc_tuning.png'), dpi=150)
        plt.show()

        print(f"\nBest Alpha selected by BIC: {best_alpha}")
        print(f"Number of edges in best graph: {best_graph.get_num_edges()}")

        G_nx = nx.DiGraph()
        nodes = best_graph.get_nodes()

        for i in range(len(nodes)):
            G_nx.add_node(i, label=labels[i])

        graph_edges = best_graph.get_graph_edges()
        directed_edges = []
        undirected_edges = []

        for edge in graph_edges:
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            idx1 = nodes.index(node1)
            idx2 = nodes.index(node2)

            endpoint1 = edge.get_endpoint1()
            endpoint2 = edge.get_endpoint2()

            ep1_name = endpoint1.name if hasattr(endpoint1, 'name') else str(endpoint1)
            ep2_name = endpoint2.name if hasattr(endpoint2, 'name') else str(endpoint2)
            if ep1_name == 'TAIL' and ep2_name == 'ARROW':
                directed_edges.append((idx1, idx2))
                G_nx.add_edge(idx1, idx2)
            elif ep1_name == 'ARROW' and ep2_name == 'TAIL':
                directed_edges.append((idx2, idx1))
                G_nx.add_edge(idx2, idx1)
            else:
                undirected_edges.append((idx1, idx2))
                G_nx.add_edge(idx1, idx2)
                G_nx.add_edge(idx2, idx1)

        plt.figure(figsize=(16, 16))
        pos = nx.spring_layout(G_nx, k=2.5, iterations=100, seed=42)

        nx.draw_networkx_nodes(G_nx, pos, node_size=2000, node_color='lightblue',
                              alpha=0.9, edgecolors='navy', linewidths=2)
        nx.draw_networkx_labels(G_nx, pos, {i: labels[i] for i in range(len(labels))},
                               font_size=11, font_weight='bold')

        if directed_edges:
            nx.draw_networkx_edges(G_nx, pos, edgelist=directed_edges,
                                  edge_color='darkblue', arrows=True, arrowsize=25,
                                  arrowstyle='-|>', connectionstyle='arc3,rad=0.15',
                                  width=2.5, alpha=0.7, node_size=2000)

        if undirected_edges:
            nx.draw_networkx_edges(G_nx, pos, edgelist=undirected_edges,
                                  edge_color='red', arrows=False,
                                  width=2.0, alpha=0.5, style='dashed')

        plt.title(f"Optimal Directed Graph (PC Algorithm, alpha={best_alpha})\n"
                 f"Directed edges: {len(directed_edges)}, Undirected edges: {len(undirected_edges)}",
                 fontsize=14, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '1c_pc_graph.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
    def fit_granger_model(self, max_lag=5, significance_level=0.05, tune_params=True, verbose=False):
        if self.log_returns is None:
            self.process_stock_data()

        # Hyperparameter tuning
        if tune_params and verbose:
            self._tune_granger_parameters(max_lag_range=[1, 3, 5, 7, 10],
                                         alpha_range=[0.01, 0.05, 0.1])

        n_features = len(self.tickers)
        granger_matrix = np.zeros((n_features, n_features))
        p_value_matrix = np.zeros((n_features, n_features))
        optimal_lags = np.zeros((n_features, n_features), dtype=int)

        if verbose:
            print(f"\nRunning Granger Causality Tests (max_lag={max_lag}, alpha={significance_level})...")

        for i, cause in enumerate(self.tickers):
            for j, effect in enumerate(self.tickers):
                if i == j:
                    continue

                data = self.log_returns[[effect, cause]].dropna()

                try:
                    test_result = grangercausalitytests(data, max_lag, verbose=False)

                    min_p_value = 1.0
                    best_lag = 0

                    for lag in range(1, max_lag + 1):
                        p_values = [test_result[lag][0][test][1]
                                   for test in ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']]
                        avg_p = np.mean(p_values)

                        if avg_p < min_p_value:
                            min_p_value = avg_p
                            best_lag = lag

                    if min_p_value < significance_level:
                        granger_matrix[i, j] = 1
                        p_value_matrix[i, j] = min_p_value
                        optimal_lags[i, j] = best_lag

                except Exception as e:
                    if verbose:
                        print(f"  Warning: Test failed for {cause} -> {effect}: {e}")
                    continue

        granger_df = pd.DataFrame(granger_matrix, index=self.tickers, columns=self.tickers)
        p_value_df = pd.DataFrame(p_value_matrix, index=self.tickers, columns=self.tickers)
        lags_df = pd.DataFrame(optimal_lags, index=self.tickers, columns=self.tickers)

        if verbose:
            self._plot_granger_results(granger_df, p_value_df, lags_df, significance_level)

        return {
            'granger_matrix': granger_df,
            'p_values': p_value_df,
            'optimal_lags': lags_df
        }

    def _tune_granger_parameters(self, max_lag_range, alpha_range):
        print("=" * 70)
        print("Hyperparameter Tuning for Granger Causality Test")
        print("=" * 70)

        results = []

        for max_lag in max_lag_range:
            for alpha in alpha_range:
                n_features = len(self.tickers)
                granger_matrix = np.zeros((n_features, n_features))

                for i, cause in enumerate(self.tickers):
                    for j, effect in enumerate(self.tickers):
                        if i == j:
                            continue

                        data = self.log_returns[[effect, cause]].dropna()

                        try:
                            test_result = grangercausalitytests(data, max_lag, verbose=False)

                            min_p_value = 1.0
                            for lag in range(1, max_lag + 1):
                                p_values = [test_result[lag][0][test][1]
                                           for test in ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']]
                                avg_p = np.mean(p_values)
                                if avg_p < min_p_value:
                                    min_p_value = avg_p

                            if min_p_value < alpha:
                                granger_matrix[i, j] = 1

                        except:
                            continue

                n_edges = int(granger_matrix.sum())
                density = n_edges / (n_features * (n_features - 1))

                results.append({
                    'max_lag': max_lag,
                    'alpha': alpha,
                    'n_edges': n_edges,
                    'density': density
                })

        results_df = pd.DataFrame(results)
        print("\nParameter Tuning Results:")
        print(results_df.to_string(index=False))

        # Visualization
        pivot_edges = results_df.pivot(index='max_lag', columns='alpha', values='n_edges')
        pivot_density = results_df.pivot(index='max_lag', columns='alpha', values='density')

        _, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(pivot_edges, annot=True, fmt='.0f', cmap='YlOrRd',
                   ax=axes[0], cbar_kws={'label': 'Number of Edges'})
        axes[0].set_title('Number of Causal Edges vs Hyperparameters')
        axes[0].set_xlabel('Significance Level (α)')
        axes[0].set_ylabel('Max Lag')

        sns.heatmap(pivot_density, annot=True, fmt='.3f', cmap='viridis',
                   ax=axes[1], cbar_kws={'label': 'Graph Density'})
        axes[1].set_title('Graph Density vs Hyperparameters')
        axes[1].set_xlabel('Significance Level (α)')
        axes[1].set_ylabel('Max Lag')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '1e_granger_tuning.png'), dpi=150)
        plt.show()

        print("\n" + "=" * 70)

    def _plot_granger_results(self, granger_df, p_value_df, lags_df, significance_level):
        n_edges = int(granger_df.sum().sum())

        print("\n--- Granger Causality Test Results ---")
        print(f"Significance level: {significance_level}")
        print(f"Number of significant Granger causal relationships: {n_edges}")
        print(f"Graph density: {n_edges / (len(self.tickers) * (len(self.tickers) - 1)):.3f}")

        # Plot 1: Granger causality adjacency matrix
        _, axes = plt.subplots(1, 2, figsize=(20, 8))

        sns.heatmap(granger_df, annot=True, fmt='.0f', cmap='Blues',
                   xticklabels=self.tickers, yticklabels=self.tickers,
                   ax=axes[0], cbar_kws={'label': 'Granger Causes (1=Yes, 0=No)'})
        axes[0].set_title(f'Granger Causality Matrix (α={significance_level})\nRow causes Column')
        axes[0].set_xlabel('Effect (Y)')
        axes[0].set_ylabel('Cause (X)')

        p_value_masked = p_value_df.copy()
        p_value_masked[granger_df == 0] = np.nan

        sns.heatmap(p_value_masked, annot=True, fmt='.3f', cmap='Reds_r',
                   xticklabels=self.tickers, yticklabels=self.tickers,
                   ax=axes[1], cbar_kws={'label': 'P-value'}, vmin=0, vmax=significance_level)
        axes[1].set_title(f'P-values for Significant Relationships')
        axes[1].set_xlabel('Effect (Y)')
        axes[1].set_ylabel('Cause (X)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '1e_granger_matrix.png'), dpi=150)
        plt.show()

        # Plot 2: Network graph
        G = nx.DiGraph()
        for ticker in self.tickers:
            G.add_node(ticker)

        edge_list = []
        for i, cause in enumerate(self.tickers):
            for j, effect in enumerate(self.tickers):
                if granger_df.iloc[i, j] == 1:
                    edge_list.append((cause, effect))
                    G.add_edge(cause, effect,
                              weight=1 - p_value_df.iloc[i, j],
                              lag=int(lags_df.iloc[i, j]))

        plt.figure(figsize=(14, 14))
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

        nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='lightblue',
                              alpha=0.9, edgecolors='navy', linewidths=2)
        nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')

        if edge_list:
            nx.draw_networkx_edges(G, pos, edgelist=edge_list,
                                  edge_color='darkblue', arrows=True, arrowsize=25,
                                  arrowstyle='-|>', connectionstyle='arc3,rad=0.1',
                                  width=2, alpha=0.7, node_size=2500)

        plt.title(f"Granger Causality Network (α={significance_level})\n"
                 f"{n_edges} significant causal relationships",
                 fontsize=14, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '1e_granger_network.png'),
                   dpi=150, bbox_inches='tight')
        plt.show()

        # Summary statistics
        print("\nTop 10 strongest Granger causal relationships:")
        relationships = []
        for i, cause in enumerate(self.tickers):
            for j, effect in enumerate(self.tickers):
                if granger_df.iloc[i, j] == 1:
                    relationships.append({
                        'Cause': cause,
                        'Effect': effect,
                        'P-value': p_value_df.iloc[i, j],
                        'Lag': int(lags_df.iloc[i, j])
                    })

        if relationships:
            relationships_df = pd.DataFrame(relationships).sort_values('P-value')
            print(relationships_df.head(10).to_string(index=False))