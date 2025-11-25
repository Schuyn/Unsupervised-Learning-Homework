'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-11-25 10:49:19
LastEditTime: 2025-11-25 12:44:23
FilePath: /Unsupervised-Learning-Homework/Homework 3/Code/Prob2b_utils.py
Description: 
    Utility functions and classes for Problem 2b of Homework 3.
    Generative Adversarial Network (GAN) for digit generation.
    - process_data: Load and preprocess sklearn digits dataset
    - build_generator: Build generator network
    - build_discriminator: Build discriminator network
    - fit_gan: Train GAN with hyperparameter tuning
    - generate_samples: Generate new samples from trained GAN
    - evaluate_samples: Evaluate quality of generated samples
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    def __init__(self, latent_dim=64, hidden_dims=[128, 256], output_dim=64, 
                 activation='relu', use_batchnorm=True):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Build layers
        layers = []
        in_dim = latent_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            in_dim = h_dim
        
        # Output layer with Sigmoid for [0, 1] range
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[256, 128], 
                 activation='leaky_relu', dropout_rate=0.3):
        super(Discriminator, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            elif activation == 'relu':
                layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        
        # Output layer with Sigmoid for probability
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class Prob2bAnalysis:
    def __init__(self,
                 output_dir='Homework 3/Code/Data',
                 figure_dir='Homework 3/Latex/Figures',
                 device=None,
                 seed=25):
        self.output_dir = output_dir
        self.figure_dir = figure_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
        else:
            self.device = device
        
        self.data = None
        self.targets = None
        self.scaler = None
        self.gan_results = None
        self.generated_samples = None
        self.training_history = None
        
        self.seed = seed
        
    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def process_data(self, verbose=False):
        # Different form 2a process_data, so cannot share code
        digits = load_digits()
        X = digits.data
        y = digits.target
        
        # Scale to [0, 1] for GAN training
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = self.scaler.fit_transform(X)
        
        self.data = X
        self.data_scaled = X_scaled
        self.targets = y
        
        if verbose:
            print(f"\nDataset shape: {X.shape}")
            print(f"Number of classes: {len(np.unique(y))}")
            print(f"Original pixel range: [{X.min():.1f}, {X.max():.1f}]")
            print(f"Scaled pixel range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
            print(f"Device: {self.device}")
            
        return X, y
    
    def fit_gan(self,
                latent_dim=64,
                g_hidden_dims=[128, 256],
                d_hidden_dims=[256, 128],
                lr_g=0.0002,
                lr_d=0.0002,
                batch_size=64,
                n_epochs=300,
                n_critic=1,
                beta1=0.5,
                beta2=0.999,
                label_smoothing=0.1,
                verbose=False):
        self._set_seed()
        if self.data is None:
            self.process_data()
        
        output_dim = self.data.shape[1]  # 64 for 8x8 images
        
        # Build networks
        generator = Generator(
            latent_dim=latent_dim,
            hidden_dims=g_hidden_dims,
            output_dim=output_dim,
            activation='relu',
            use_batchnorm=True
        ).to(self.device)
        
        discriminator = Discriminator(
            input_dim=output_dim,
            hidden_dims=d_hidden_dims,
            activation='leaky_relu',
            dropout_rate=0.3
        ).to(self.device)
        
        # Optimizers
        optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
        optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Prepare data
        X_tensor = torch.FloatTensor(self.data_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Training history
        history = {
            'epoch': [],
            'd_loss': [],
            'g_loss': [],
            'd_real_acc': [],
            'd_fake_acc': []
        }
        
        if verbose:
            print(f"\nTraining GAN...")
            print(f"  Latent dim: {latent_dim}")
            print(f"  Generator: {g_hidden_dims}")
            print(f"  Discriminator: {d_hidden_dims}")
            print(f"  Epochs: {n_epochs}, Batch size: {batch_size}")
            print("-" * 60)
        
        # Training loop
        for epoch in range(n_epochs):
            d_losses = []
            g_losses = []
            d_real_accs = []
            d_fake_accs = []
            
            for batch_data in dataloader:
                real_data = batch_data[0]
                current_batch_size = real_data.size(0)
                
                # Labels with smoothing
                real_labels = torch.ones(current_batch_size, 1).to(self.device) * (1 - label_smoothing)
                fake_labels = torch.zeros(current_batch_size, 1).to(self.device)
                
                # ---------------------
                # Train Discriminator
                # ---------------------
                for _ in range(n_critic):
                    optimizer_d.zero_grad()
                    
                    # Real data
                    d_real_output = discriminator(real_data)
                    d_real_loss = criterion(d_real_output, real_labels)
                    
                    # Fake data
                    z = torch.randn(current_batch_size, latent_dim).to(self.device)
                    fake_data = generator(z)
                    d_fake_output = discriminator(fake_data.detach())
                    d_fake_loss = criterion(d_fake_output, fake_labels)
                    
                    # Total discriminator loss
                    d_loss = d_real_loss + d_fake_loss
                    d_loss.backward()
                    optimizer_d.step()
                
                d_losses.append(d_loss.item())
                d_real_accs.append((d_real_output > 0.5).float().mean().item())
                d_fake_accs.append((d_fake_output < 0.5).float().mean().item())
                
                # ---------------------
                # Train Generator
                # ---------------------
                optimizer_g.zero_grad()
                
                z = torch.randn(current_batch_size, latent_dim).to(self.device)
                fake_data = generator(z)
                g_output = discriminator(fake_data)
                
                # Generator wants discriminator to think fake is real
                g_loss = criterion(g_output, torch.ones(current_batch_size, 1).to(self.device))
                g_loss.backward()
                optimizer_g.step()
                
                g_losses.append(g_loss.item())
            
            # Record epoch metrics
            history['epoch'].append(epoch + 1)
            history['d_loss'].append(np.mean(d_losses))
            history['g_loss'].append(np.mean(g_losses))
            history['d_real_acc'].append(np.mean(d_real_accs))
            history['d_fake_acc'].append(np.mean(d_fake_accs))
            
            # Print progress
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1:4d}/{n_epochs}] | "
                      f"D Loss: {history['d_loss'][-1]:.4f} | "
                      f"G Loss: {history['g_loss'][-1]:.4f} | "
                      f"D(real): {history['d_real_acc'][-1]:.3f} | "
                      f"D(fake): {history['d_fake_acc'][-1]:.3f}")
        
        # Store results
        self.gan_results = {
            'generator': generator,
            'discriminator': discriminator,
            'latent_dim': latent_dim,
            'history': history,
            'config': {
                'latent_dim': latent_dim,
                'g_hidden_dims': g_hidden_dims,
                'd_hidden_dims': d_hidden_dims,
                'lr_g': lr_g,
                'lr_d': lr_d,
                'batch_size': batch_size,
                'n_epochs': n_epochs
            }
        }
        
        self.training_history = history
        
        if verbose:
            print("Training completed!")
            self._plot_training_history(history)
        
        return self.gan_results
    
    def tune_hyperparameters(self,
                             latent_dims=[32, 64, 128],
                             lr_g_values=[0.00005, 0.0001, 0.0002],
                             lr_d=0.0002,   
                             hidden_configs=[
                                 ([128, 256], [256, 128]),
                                 ([256, 512], [512, 256]),
                                 ([64, 128, 256], [256, 128, 64])
                             ],
                             n_epochs=200,
                             n_eval_samples=500,
                             verbose=False):
        if self.data is None:
            self.process_data()
        
        results = []
        
        for latent_dim in latent_dims:
            for lr_g in lr_g_values:
                for g_hidden, d_hidden in hidden_configs:
                    # Train model
                    self.fit_gan(
                        latent_dim=latent_dim,
                        g_hidden_dims=g_hidden,
                        d_hidden_dims=d_hidden,
                        lr_g=lr_g,
                        lr_d=lr_d,
                        n_epochs=n_epochs,
                        verbose=False
                    )
                        
                    # Generate and evaluate samples
                    samples = self.generate_samples(n_samples=n_eval_samples, verbose=False)
                    metrics = self.evaluate_samples(samples, verbose=False)
                    
                    # Store results
                    result = {
                        'latent_dim': latent_dim,
                        'lr_g': lr_g,
                        'g_hidden': str(g_hidden),
                        'd_hidden': str(d_hidden),
                        'mean_diff': abs(metrics['generated_mean'] - metrics['original_mean']),
                        'std_diff': abs(metrics['generated_std'] - metrics['original_std']),
                        'diversity_ratio': metrics.get('diversity_ratio', 0)
                    }
                    results.append(result)
                    
                    if verbose:
                        print(f"  Mean Diff: {result['mean_diff']:.4f} | "
                              f"Std Diff: {result['std_diff']:.4f} | "
                              f"Diversity: {result['diversity_ratio']:.3f}")
        
        results_df = pd.DataFrame(results)
        
        if verbose:
            print("Tuning Results Summary:")
            print(results_df.to_string(index=False))
            self._plot_tuning_results(results_df)
        
        return results_df
    
    def _plot_tuning_results(self, results_df):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Mean diff by lr_g, grouped by latent_dim
        for latent in results_df['latent_dim'].unique():
            subset = results_df[results_df['latent_dim'] == latent]
            avg_by_lr = subset.groupby('lr_g')['mean_diff'].mean()
            axes[0].plot(avg_by_lr.index, avg_by_lr.values, 'o-', label=f'latent={latent}', markersize=8)
        axes[0].set_title('Mean Diff vs lr_g (Lower is Better)')
        axes[0].set_xscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Mean difference
        pivot = results_df.pivot_table(
            values='mean_diff', 
            index='latent_dim', 
            columns='lr_g',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1])
        axes[1].set_title('Mean Difference (Lower is Better)')
        
        # Std difference
        pivot_std = results_df.pivot_table(
            values='std_diff',
            index='latent_dim',
            columns='lr_g',
            aggfunc='mean'
        )
        sns.heatmap(pivot_std, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[2])
        axes[2].set_title('Std Difference (Lower is Better)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '2b_gan_tuning.png'), dpi=150)
        plt.show()
    
    def _plot_training_history(self, history):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = history['epoch']
        
        # Losses
        axes[0].plot(epochs, history['d_loss'], label='Discriminator Loss', alpha=0.8)
        axes[0].plot(epochs, history['g_loss'], label='Generator Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('GAN Training Losses')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Discriminator accuracy
        axes[1].plot(epochs, history['d_real_acc'], label='D(real) accuracy', alpha=0.8)
        axes[1].plot(epochs, history['d_fake_acc'], label='D(fake) accuracy', alpha=0.8)
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random guess')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Discriminator Performance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '2b_gan_training.png'), dpi=150)
        plt.show()
    
    def generate_samples(self, n_samples=100, verbose=False):
        generator = self.gan_results['generator']
        latent_dim = self.gan_results['latent_dim']
        
        generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, latent_dim).to(self.device)
            generated_scaled = generator(z).cpu().numpy()
        
        # Inverse transform to original scale
        generated = self.scaler.inverse_transform(generated_scaled)
        
        # Clip to valid range [0, 16]
        generated = np.clip(generated, 0, 16)
        
        self.generated_samples = {
            'samples': generated,
            'samples_scaled': generated_scaled,
            'n_samples': n_samples
        }
        
        if verbose:
            print(f"\nGenerated {n_samples} samples from GAN")
            print(f"Generated samples shape: {generated.shape}")
            print(f"Generated samples range: [{generated.min():.2f}, {generated.max():.2f}]")
        
        return generated
    
    def evaluate_samples(self, generated_samples=None, verbose=False):
        if generated_samples is None:
            generated_samples = self.generated_samples['samples']
        
        metrics = {}
        
        # 1. Sample statistics comparison
        metrics['original_mean'] = self.data.mean()
        metrics['original_std'] = self.data.std()
        metrics['generated_mean'] = generated_samples.mean()
        metrics['generated_std'] = generated_samples.std()
        
        # 2. Sparsity check (% of near-zero pixels)
        threshold = 0.5
        metrics['original_sparsity'] = (self.data < threshold).mean()
        metrics['generated_sparsity'] = (generated_samples < threshold).mean()
        
        # 3. Coverage: fraction of original data modes covered
        # Using PCA + clustering approximation
        try:
            pca = PCA(n_components=10)
            orig_pca = pca.fit_transform(self.data)
            gen_pca = pca.transform(generated_samples)
            
            # Simple coverage: check if generated samples are near original samples
            from scipy.spatial.distance import cdist
            distances = cdist(gen_pca, orig_pca, metric='euclidean')
            min_distances = distances.min(axis=1)
            coverage_threshold = np.percentile(
                cdist(orig_pca, orig_pca).flatten(), 50
            )
            metrics['coverage'] = (min_distances < coverage_threshold).mean()
        except Exception as e:
            metrics['coverage'] = None
        
        # 4. Mode collapse check: diversity of generated samples
        try:
            gen_pca = PCA(n_components=10).fit_transform(generated_samples)
            pairwise_dist = cdist(gen_pca, gen_pca, metric='euclidean')
            np.fill_diagonal(pairwise_dist, np.inf)
            metrics['avg_nn_distance'] = pairwise_dist.min(axis=1).mean()
            
            orig_pca = PCA(n_components=10).fit_transform(self.data)
            orig_pairwise = cdist(orig_pca, orig_pca, metric='euclidean')
            np.fill_diagonal(orig_pairwise, np.inf)
            metrics['orig_avg_nn_distance'] = orig_pairwise.min(axis=1).mean()
            
            # Diversity ratio (higher is better, 1.0 means same diversity as original)
            metrics['diversity_ratio'] = metrics['avg_nn_distance'] / metrics['orig_avg_nn_distance']
        except Exception as e:
            metrics['diversity_ratio'] = None
        
        if verbose:
            self._print_evaluation_metrics(metrics)
        
        return metrics
    
    def _print_evaluation_metrics(self, metrics):
        """Print evaluation metrics."""
        print("\n--- GAN Evaluation Metrics ---")
        print(f"Mean (Original):     {metrics['original_mean']:.4f}")
        print(f"Mean (Generated):    {metrics['generated_mean']:.4f}")
        print(f"Std  (Original):     {metrics['original_std']:.4f}")
        print(f"Std  (Generated):    {metrics['generated_std']:.4f}")
        print(f"Sparsity (Original): {metrics['original_sparsity']:.4f}")
        print(f"Sparsity (Generated):{metrics['generated_sparsity']:.4f}")
        if metrics.get('coverage') is not None:
            print(f"Coverage:            {metrics['coverage']:.4f}")
        if metrics.get('diversity_ratio') is not None:
            print(f"Diversity Ratio:     {metrics['diversity_ratio']:.4f}")
    
    def visualize_generated_samples(self, generated_samples=None, n_display=25):
        """Visualize grid of generated samples."""
        if generated_samples is None:
            generated_samples = self.generated_samples['samples']
        
        grid_size = int(np.sqrt(n_display))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()
        
        for i in range(min(n_display, len(generated_samples))):
            axes[i].imshow(generated_samples[i].reshape(8, 8), cmap='gray')
            axes[i].axis('off')
        
        plt.suptitle('Generated Digits from GAN', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '2b_generated_samples_gan.png'), dpi=150)
        plt.show()
    
    def compare_distributions(self, generated_samples=None):
        if generated_samples is None:
            generated_samples = self.generated_samples['samples']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
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
        sparsity_orig = (self.data < 0.5).sum(axis=1).mean()
        sparsity_gen = (generated_samples < 0.5).sum(axis=1).mean()
        
        categories = ['Original', 'Generated']
        sparsities = [sparsity_orig, sparsity_gen]
        axes[1].bar(categories, sparsities, color=['blue', 'red'], alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Average # of Near-Zero Pixels')
        axes[1].set_title('Sparsity Comparison')
        axes[1].grid(axis='y', alpha=0.3)
        
        # PCA projection comparison
        pca = PCA(n_components=2)
        orig_pca = pca.fit_transform(self.data)
        gen_pca = pca.transform(generated_samples)
        
        axes[2].scatter(orig_pca[:, 0], orig_pca[:, 1], alpha=0.3, label='Original', s=10)
        axes[2].scatter(gen_pca[:, 0], gen_pca[:, 1], alpha=0.5, label='Generated', s=10)
        axes[2].set_xlabel('PC1')
        axes[2].set_ylabel('PC2')
        axes[2].set_title('PCA Projection Comparison')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '2b_distribution_comparison_gan.png'), dpi=150)
        plt.show()
    
    def visualize_latent_interpolation(self, n_steps=10):
        generator = self.gan_results['generator']
        latent_dim = self.gan_results['latent_dim']
        
        generator.eval()
        
        # Generate two random latent vectors
        z1 = torch.randn(1, latent_dim).to(self.device)
        z2 = torch.randn(1, latent_dim).to(self.device)
        
        # Interpolate
        interpolations = []
        for alpha in np.linspace(0, 1, n_steps):
            z = (1 - alpha) * z1 + alpha * z2
            with torch.no_grad():
                img = generator(z).cpu().numpy()
            img = self.scaler.inverse_transform(img)
            img = np.clip(img, 0, 16)
            interpolations.append(img.reshape(8, 8))
        
        # Plot
        fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 2))
        for i, img in enumerate(interpolations):
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('Start')
            elif i == n_steps - 1:
                axes[i].set_title('End')
        
        plt.suptitle('Latent Space Interpolation', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '2b_latent_interpolation.png'), dpi=150)
        plt.show()
    
    def compare_with_real_samples(self, n_display=10):
        generated = self.generated_samples['samples'][:n_display]
        real_indices = np.random.choice(len(self.data), n_display, replace=False)
        real = self.data[real_indices]
        
        fig, axes = plt.subplots(2, n_display, figsize=(2 * n_display, 4))
        
        for i in range(n_display):
            axes[0, i].imshow(real[i].reshape(8, 8), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Real', fontsize=12)
            
            axes[1, i].imshow(generated[i].reshape(8, 8), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Generated', fontsize=12)
        
        plt.suptitle('Real vs Generated Samples', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '2b_real_vs_generated.png'), dpi=150)
        plt.show()


