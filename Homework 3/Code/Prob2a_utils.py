'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-11-24 19:58:10
LastEditTime: 2025-11-25 09:43:24
FilePath: /Unsupervised-Learning-Homework/Homework 3/Code/Prob2a_utils.py
Description: 
    Utility functions and classes for Problem 2 of Homework 3.
    Kernel Density Estimation (KDE) for digit generation.
    - process_data: Load and preprocess sklearn digits dataset
    - fit_kde_models: Fit KDE in multiple spaces with hyperparameter tuning
    - generate_samples: Generate new samples from trained KDE
    - evaluate_samples: Evaluate quality of generated samples
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist


class Prob2Analysis:
    def __init__(self,
                 output_dir='Homework 3/Code/Data',
                 figure_dir='Homework 3/Latex/Figures'):
        self.output_dir = output_dir
        self.figure_dir = figure_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        
        self.data = None
        self.targets = None
        self.kde_results = None
        self.generated_samples = None
        
    def process_data(self, verbose=False):
        digits = load_digits()
        X = digits.data
        y = digits.target
        
        self.data = X
        self.targets = y
        
        if verbose:
            print(f"\nDataset shape: {X.shape}")
            print(f"Number of classes: {len(np.unique(y))}")
            print(f"Pixel value range: [{X.min():.1f}, {X.max():.1f}]")
            print(f"Mean pixel value: {X.mean():.2f} ± {X.std():.2f}")
            
        return X, y
    
    def fit_kde_models(self, 
                       spaces=['pca_20', 'pca_50', 'original'],
                       bandwidth_range=None,
                       kernel='gaussian',
                       cv_folds=5,
                       verbose=False):
        if self.data is None:
            self.process_data()
        
        if bandwidth_range is None:
            bandwidth_range = np.logspace(-1.5, 0.5, 20)
        
        results = {}
        
        if verbose:
            print("Hyperparameter Tuning: Bandwidth Selection")
        
        for space in spaces:
            # Prepare data in target space
            if space == 'original':
                X_space = StandardScaler().fit_transform(self.data)
                n_features = self.data.shape[1]
                
            else:
                n_comp = int(space.split('_')[1])
                pca = PCA(n_components=n_comp)
                X_space = pca.fit_transform(self.data)
                X_space = StandardScaler().fit_transform(X_space)
                n_features = n_comp
            
            # Hyperparameter tuning
            kde = KernelDensity(kernel=kernel, algorithm='ball_tree')
            best_bandwidth = None
            best_score = -np.inf
            cv_scores_dict = {'bandwidth': [], 'mean_score': [], 'std_score': []}
            
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=25)
            
            for bw in bandwidth_range:
                fold_scores = []
                for train_idx, test_idx in kf.split(X_space):
                    X_train, X_test = X_space[train_idx], X_space[test_idx]
                    kde = KernelDensity(bandwidth=bw)
                    kde.fit(X_train)
                    score = kde.score(X_test)
                    fold_scores.append(score)

                mean_score = np.mean(fold_scores)
                std_score = np.std(fold_scores)
                
                cv_scores_dict['bandwidth'].append(bw)
                cv_scores_dict['mean_score'].append(mean_score)
                cv_scores_dict['std_score'].append(std_score)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_bandwidth = bw
            
            if verbose:
                print(f"  Best bandwidth: {best_bandwidth:.6f}")
                print(f"  Cross-validation score: {best_score:.4f}")
            
            # Train final model
            kde_final = KernelDensity(
                kernel=kernel,
                bandwidth=best_bandwidth,
                algorithm='ball_tree'
            )
            kde_final.fit(X_space)
            
            results[space] = {
                'model': kde_final,
                'bandwidth': best_bandwidth,
                'cv_score': best_score,
                'X_train': X_space,
                'cv_results': cv_scores_dict,
                'n_features': n_features,
                'pca': PCA(n_components=int(space.split('_')[1]))
                        if space.startswith('pca_') else None
            }
        
        self.kde_results = results
        
        if verbose:
            self._plot_kde_tuning_results(results)
        
        return results
    
    def _plot_kde_tuning_results(self, results):
        """Visualize bandwidth tuning for each space."""
        n_spaces = len(results)
        fig, axes = plt.subplots(1, n_spaces, figsize=(5*n_spaces, 4))
        
        if n_spaces == 1:
            axes = [axes]
        
        for idx, (space, result) in enumerate(results.items()):
            cv_results = result['cv_results']
            bandwidths = cv_results['bandwidth']
            mean_scores = cv_results['mean_score']
            std_scores = cv_results['std_score']
            
            axes[idx].semilogx(bandwidths, mean_scores, 'o-', linewidth=2, markersize=6)
            axes[idx].fill_between(bandwidths, 
                                   np.array(mean_scores) - np.array(std_scores),
                                   np.array(mean_scores) + np.array(std_scores),
                                   alpha=0.2)
            axes[idx].axvline(result['bandwidth'], linestyle='--', color='r', 
                             label=f"Best: {result['bandwidth']:.4f}")
            axes[idx].set_xlabel('Bandwidth')
            axes[idx].set_ylabel('Mean Log-Likelihood (Higher is Better)')
            axes[idx].set_title(f'Bandwidth Tuning: {space}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '2a_kde_tuning.png'), dpi=150)
        plt.show()
    
    def generate_samples(self, space='pca_20', n_samples=100, verbose=False):
        result = self.kde_results[space]
        model = result['model']
        
        # Generate samples in latent space
        X_generated = model.sample(n_samples, random_state=42)
        
        # Transform back to original space
        if space == 'original':
            # Need to inverse standardization
            # For original space, we need the scaler that was used
            scaler = StandardScaler()
            scaler.fit(self.data)
            X_original = scaler.inverse_transform(X_generated)
        
        else:
            # Inverse PCA transformation
            pca = result['pca']
            pca.fit(self.data)
            X_pca_space = X_generated
            
            # Inverse standardization in PCA space
            scaler = StandardScaler()
            scaler.fit(pca.transform(self.data))
            X_pca_unscaled = scaler.inverse_transform(X_pca_space)
            
            # Inverse PCA
            X_original = pca.inverse_transform(X_pca_unscaled)
        
        # Clip to valid range [0, 16]
        X_original = np.clip(X_original, 0, 16)
        
        self.generated_samples = {
            'samples': X_original,
            'space': space,
            'n_samples': n_samples
        }
        
        if verbose:
            print(f"\nGenerated {n_samples} samples from space: {space}")
            print(f"Generated samples shape: {X_original.shape}")
            print(f"Generated samples range: [{X_original.min():.2f}, {X_original.max():.2f}]")
        
        return X_original
    
    def evaluate_samples(self, generated_samples=None, verbose=False):
        if generated_samples is None:
            generated_samples = self.generated_samples['samples']
        
        metrics = {}
        
        # 1. Sample statistics comparison
        metrics['original_mean'] = self.data.mean()
        metrics['original_std'] = self.data.std()
        metrics['generated_mean'] = generated_samples.mean()
        metrics['generated_std'] = generated_samples.std()
        
        # 2. Sparsity check (% of zero pixels)
        metrics['original_sparsity'] = (self.data == 0).mean()
        metrics['generated_sparsity'] = (generated_samples == 0).mean()
        
        # 3. Quality check: Can we classify generated samples?
        try:
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.model_selection import cross_val_score
            
            knn = KNeighborsClassifier(n_neighbors=5)
            scores = cross_val_score(
                knn, self.data, self.targets, cv=5, scoring='accuracy'
            )
            metrics['knn_accuracy_original'] = scores.mean()
            
        except Exception as e:
            metrics['knn_accuracy_original'] = None
        
        if verbose:
            self._print_evaluation_metrics(metrics)
        
        return metrics
    
    def _print_evaluation_metrics(self, metrics):
        print(f"Mean (Original):     {metrics['original_mean']:.4f}")
        print(f"Mean (Generated):    {metrics['generated_mean']:.4f}")
        print(f"Std  (Original):     {metrics['original_std']:.4f}")
        print(f"Std  (Generated):    {metrics['generated_std']:.4f}")
        print(f"Sparsity (Original): {metrics['original_sparsity']:.4f}")
        print(f"Sparsity (Generated):{metrics['generated_sparsity']:.4f}")
        if metrics['knn_accuracy_original'] is not None:
            print(f"KNN Accuracy (Original): {metrics['knn_accuracy_original']:.4f}")
    
    def visualize_generated_samples(self, generated_samples=None, n_display=25):
        if generated_samples is None:
            generated_samples = self.generated_samples['samples']
            space = self.generated_samples['space']
        else:
            space = 'provided'
        
        grid_size = int(np.sqrt(n_display))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()
        
        for i in range(min(n_display, len(generated_samples))):
            axes[i].imshow(generated_samples[i].reshape(8, 8), cmap='gray')
            axes[i].axis('off')
        
        plt.suptitle(f'Generated Digits from KDE ({space})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, f'2a_generated_samples_{space}.png'), dpi=150)
        plt.show()
    
    def compare_distributions(self, generated_samples=None):
        if generated_samples is None:
            generated_samples = self.generated_samples['samples']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pixel intensity distribution
        axes[0].hist(self.data.flatten(), bins=50, alpha=0.6, label='Original', 
                    color='blue', density=True)
        axes[0].hist(generated_samples.flatten(), bins=50, alpha=0.6, label='Generated',
                    color='red', density=True)
        axes[0].set_xlabel('Pixel Intensity')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Pixel Intensity Distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Sparsity comparison
        sparsity_orig = (self.data == 0).sum(axis=1).mean()
        sparsity_gen = (generated_samples == 0).sum(axis=1).mean()
        
        categories = ['Original', 'Generated']
        sparsities = [sparsity_orig, sparsity_gen]
        axes[1].bar(categories, sparsities, color=['blue', 'red'], alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Average # of Zero Pixels')
        axes[1].set_title('Sparsity Comparison')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '2a_distribution_comparison.png'), dpi=150)
        plt.show()


