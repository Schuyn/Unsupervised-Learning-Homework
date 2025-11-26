'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-11-25 15:08:49
LastEditTime: 2025-11-25 20:32:03
FilePath: /Unsupervised-Learning-Homework/Homework 3/Code/Prob2c_utils.py
Description: 
    Utility functions and classes for Problem 2c of Homework 3.
    Denoising Diffusion Probabilistic Model (DDPM) for digit generation.
    - process_data: Load and preprocess sklearn digits dataset
    - build_denoiser: Build denoising network
    - fit_diffusion: Train diffusion model with hyperparameter tuning
    - generate_samples: Generate new samples via reverse diffusion
    - evaluate_samples: Evaluate quality of generated samples
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # Use both sin and cos to capture more information
        return embeddings


class DenoisingMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[256, 512, 256], time_emb_dim=64):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(), # Instead of ReLU, GELU(Gaussian Error Linear Unit) as a smooth version often performs better
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        
        # Time embedding projection for each layer
        self.time_projections = nn.ModuleList([
            nn.Linear(time_emb_dim, h_dim) for h_dim in hidden_dims
        ])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.GroupNorm(8, hidden_dims[i + 1]),
                    nn.GELU()
                )
            )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], input_dim)
        
        self.act = nn.GELU()
        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(8, h_dim) for h_dim in hidden_dims
        ])
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Input
        h = self.input_layer(x)
        h = self.norm_layers[0](h)
        h = self.act(h)
        h = h + self.time_projections[0](t_emb)
        
        # Hidden layers
        for i, layer in enumerate(self.hidden_layers):
            h = layer(h)
            h = h + self.time_projections[i + 1](t_emb)
        
        # Output
        return self.output_layer(h)


class DiffusionScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device), 
            self.alphas_cumprod[:-1]
        ])
        
        # Pre-compute values for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Pre-compute values for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
    
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t):
        """Reverse diffusion: p(x_{t-1} | x_t)"""
        betas_t = self.betas[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None]
        
        # Predict noise
        predicted_noise = model(x_t, t.float())
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        # Add noise (except for t=0)
        if (t > 0).any():
            posterior_variance_t = self.posterior_variance[t][:, None]
            noise = torch.randn_like(x_t)
            # Only add noise where t > 0
            mask = (t > 0).float()[:, None]
            return model_mean + mask * torch.sqrt(posterior_variance_t) * noise
        else:
            return model_mean


class Prob2cAnalysis:
    def __init__(self,
                 output_dir='Homework 3/Code/Data',
                 figure_dir='Homework 3/Latex/Figures',
                 device=None,
                 seed=25):
        self.output_dir = output_dir
        self.figure_dir = figure_dir
        self.seed = seed
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.data = None
        self.targets = None
        self.scaler = None
        self.diffusion_results = None
        self.generated_samples = None
        self.training_history = None
    
    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def process_data(self, verbose=False):
        digits = load_digits()
        X = digits.data
        y = digits.target
        
        # Scale to [0, 1] then to [-1, 1] for diffusion
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
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
    
    def fit_diffusion(self,
                      num_timesteps=500,
                      hidden_dims=[256, 512, 256],
                      time_emb_dim=64,
                      lr=1e-3,
                      batch_size=128,
                      n_epochs=200,
                      beta_start=1e-4,
                      beta_end=0.02,
                      verbose=False):
        self._set_seed()
        
        if self.data is None:
            self.process_data()
        
        input_dim = self.data.shape[1]  # 64 for 8x8 images
        
        # Build model
        model = DenoisingMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            time_emb_dim=time_emb_dim
        ).to(self.device)
        
        # Build scheduler
        scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=self.device
        )
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Prepare data
        X_tensor = torch.FloatTensor(self.data_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Training history
        history = {
            'epoch': [],
            'loss': []
        }
        
        if verbose:
            print(f"\nTraining Diffusion Model...")
            print(f"  Timesteps: {num_timesteps}")
            print(f"  Hidden dims: {hidden_dims}")
            print(f"  Learning rate: {lr}")
            print(f"  Epochs: {n_epochs}, Batch size: {batch_size}")
        
        # Training loop
        model.train()
        for epoch in range(n_epochs):
            epoch_losses = []
            
            for batch_data in dataloader:
                x_0 = batch_data[0]
                current_batch_size = x_0.size(0)
                
                # Sample random timesteps
                t = torch.randint(0, num_timesteps, (current_batch_size,), device=self.device)
                
                # Sample noise
                noise = torch.randn_like(x_0)
                
                # Forward diffusion
                x_t = scheduler.q_sample(x_0, t, noise)
                
                # Predict noise
                optimizer.zero_grad()
                predicted_noise = model(x_t, t.float())
                
                # Loss
                loss = criterion(predicted_noise, noise)
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            history['epoch'].append(epoch + 1)
            history['loss'].append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1:4d}/{n_epochs}] | Loss: {avg_loss:.6f}")
        
        # Store results
        self.diffusion_results = {
            'model': model,
            'scheduler': scheduler,
            'history': history,
            'config': {
                'num_timesteps': num_timesteps,
                'hidden_dims': hidden_dims,
                'time_emb_dim': time_emb_dim,
                'lr': lr,
                'batch_size': batch_size,
                'n_epochs': n_epochs
            }
        }
        
        self.training_history = history
        
        if verbose:
            self._plot_training_history(history)
        
        return self.diffusion_results
    
    def tune_hyperparameters(self,
                             num_timesteps_values=[200, 500, 1000],
                             lr_values=[1e-4, 5e-4, 1e-3],
                             hidden_configs=[
                                 [128, 256, 128],
                                 [256, 512, 256],
                                 [256, 512, 512, 256]
                             ],
                             n_epochs=150,
                             n_eval_samples=500,
                             verbose=False):
        if self.data is None:
            self.process_data()
        
        results = []
        total_configs = len(num_timesteps_values) * len(lr_values) * len(hidden_configs)
        current_config = 0
        
        if verbose:
            print("Hyperparameter Tuning for Diffusion Model")
            print(f"Total configurations: {total_configs}")
        
        for num_timesteps in num_timesteps_values:
            for lr in lr_values:
                for hidden_dims in hidden_configs:
                    current_config += 1
                    if verbose:
                        print(f"\n[{current_config}/{total_configs}] "
                              f"T={num_timesteps}, lr={lr}, hidden={hidden_dims}")
                    
                    # Train model
                    self.fit_diffusion(
                        num_timesteps=num_timesteps,
                        hidden_dims=hidden_dims,
                        lr=lr,
                        n_epochs=n_epochs,
                        verbose=False
                    )
                    
                    # Generate and evaluate samples
                    samples = self.generate_samples(n_samples=n_eval_samples, verbose=False)
                    metrics = self.evaluate_samples(samples, verbose=False)
                    
                    # Store results
                    result = {
                        'num_timesteps': num_timesteps,
                        'lr': lr,
                        'hidden_dims': str(hidden_dims),
                        'final_loss': self.training_history['loss'][-1],
                        'mean_diff': abs(metrics['generated_mean'] - metrics['original_mean']),
                        'std_diff': abs(metrics['generated_std'] - metrics['original_std']),
                        'diversity_ratio': metrics.get('diversity_ratio', 0)
                    }
                    results.append(result)
                    
                    if verbose:
                        print(f"  Loss: {result['final_loss']:.6f} | "
                              f"Mean Diff: {result['mean_diff']:.4f} | "
                              f"Std Diff: {result['std_diff']:.4f}")
        
        results_df = pd.DataFrame(results)
        
        if verbose:
            print("Tuning Results Summary:")
            print(results_df.to_string(index=False))
            
            # Find best configuration
            best_idx = results_df['mean_diff'].idxmin()
            best_config = results_df.iloc[best_idx]
            print("Best Configuration (by Mean Diff):")
            print(f"  Timesteps: {best_config['num_timesteps']}")
            print(f"  Learning Rate: {best_config['lr']}")
            print(f"  Hidden Dims: {best_config['hidden_dims']}")
            print(f"  Mean Diff: {best_config['mean_diff']:.4f}")
            print(f"  Std Diff: {best_config['std_diff']:.4f}")
            
            self._plot_tuning_results(results_df)
        
        return results_df
    
    def _plot_tuning_results(self, results_df):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Mean difference heatmap (timesteps x lr)
        pivot_mean = results_df.pivot_table(
            values='mean_diff',
            index='num_timesteps',
            columns='lr',
            aggfunc='mean'
        )
        sns.heatmap(pivot_mean, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0])
        axes[0].set_title('Mean Difference (Lower is Better)')
        axes[0].set_xlabel('Learning Rate')
        axes[0].set_ylabel('Timesteps')
        
        # Std difference heatmap
        pivot_std = results_df.pivot_table(
            values='std_diff',
            index='num_timesteps',
            columns='lr',
            aggfunc='mean'
        )
        sns.heatmap(pivot_std, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1])
        axes[1].set_title('Std Difference (Lower is Better)')
        axes[1].set_xlabel('Learning Rate')
        axes[1].set_ylabel('Timesteps')
        
        # Final loss by architecture
        arch_loss = results_df.groupby('hidden_dims')['final_loss'].mean()
        colors = plt.cm.Set2(np.linspace(0, 1, len(arch_loss)))
        axes[2].bar(range(len(arch_loss)), arch_loss.values,
                      color=colors, edgecolor='black')
        axes[2].set_xticks(range(len(arch_loss)))
        axes[2].set_xticklabels([s.replace(', ', ',\n') for s in arch_loss.index],
                                   fontsize=8, rotation=0)
        axes[2].set_ylabel('Final Loss')
        axes[2].set_title('Final Loss by Architecture (Lower is Better)')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '2c_diffusion_tuning.png'), dpi=150)
        plt.show()
    
    def _plot_training_history(self, history):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        epochs = history['epoch']
        losses = history['loss']
        
        ax.plot(epochs, losses, 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Diffusion Model Training Loss')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '2c_diffusion_training.png'), dpi=150)
        plt.show()
    
    def generate_samples(self, n_samples=100, verbose=False):
        """Generate new samples via reverse diffusion."""
        if self.diffusion_results is None:
            raise ValueError("Diffusion model not trained. Call fit_diffusion() first.")
        
        model = self.diffusion_results['model']
        scheduler = self.diffusion_results['scheduler']
        num_timesteps = self.diffusion_results['config']['num_timesteps']
        input_dim = self.data.shape[1]
        
        model.eval()
        
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(n_samples, input_dim, device=self.device)
            
            # Reverse diffusion
            for t in reversed(range(num_timesteps)):
                t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
                x = scheduler.p_sample(model, x, t_batch)
            
            generated_scaled = x.cpu().numpy()
        
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
            print(f"\nGenerated {n_samples} samples from Diffusion Model")
            print(f"Generated samples shape: {generated.shape}")
            print(f"Generated samples range: [{generated.min():.2f}, {generated.max():.2f}]")
        
        return generated
    
    def evaluate_samples(self, generated_samples=None, verbose=False):
        """Evaluate quality of generated samples."""
        if generated_samples is None:
            generated_samples = self.generated_samples['samples']
        
        metrics = {}
        
        # 1. Sample statistics comparison
        metrics['original_mean'] = self.data.mean()
        metrics['original_std'] = self.data.std()
        metrics['generated_mean'] = generated_samples.mean()
        metrics['generated_std'] = generated_samples.std()
        
        # 2. Sparsity check
        threshold = 0.5
        metrics['original_sparsity'] = (self.data < threshold).mean()
        metrics['generated_sparsity'] = (generated_samples < threshold).mean()
        
        # 3. Diversity check
        try:
            gen_pca = PCA(n_components=10).fit_transform(generated_samples)
            pairwise_dist = cdist(gen_pca, gen_pca, metric='euclidean')
            np.fill_diagonal(pairwise_dist, np.inf)
            metrics['avg_nn_distance'] = pairwise_dist.min(axis=1).mean()
            
            orig_pca = PCA(n_components=10).fit_transform(self.data)
            orig_pairwise = cdist(orig_pca, orig_pca, metric='euclidean')
            np.fill_diagonal(orig_pairwise, np.inf)
            metrics['orig_avg_nn_distance'] = orig_pairwise.min(axis=1).mean()
            
            metrics['diversity_ratio'] = metrics['avg_nn_distance'] / metrics['orig_avg_nn_distance']
        except Exception as e:
            metrics['diversity_ratio'] = None
        
        if verbose:
            self._print_evaluation_metrics(metrics)
        
        return metrics
    
    def _print_evaluation_metrics(self, metrics):
        """Print evaluation metrics."""
        print("\n--- Diffusion Model Evaluation Metrics ---")
        print(f"Mean (Original):     {metrics['original_mean']:.4f}")
        print(f"Mean (Generated):    {metrics['generated_mean']:.4f}")
        print(f"Std  (Original):     {metrics['original_std']:.4f}")
        print(f"Std  (Generated):    {metrics['generated_std']:.4f}")
        print(f"Sparsity (Original): {metrics['original_sparsity']:.4f}")
        print(f"Sparsity (Generated):{metrics['generated_sparsity']:.4f}")
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
        
        plt.suptitle('Generated Digits from Diffusion Model', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '2c_generated_samples_diffusion.png'), dpi=150)
        plt.show()
    
    def compare_distributions(self, generated_samples=None):
        """Compare original vs generated distributions."""
        if generated_samples is None:
            generated_samples = self.generated_samples['samples']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Pixel intensity distribution
        axes[0].hist(self.data.flatten(), bins=50, alpha=0.6, label='Original',
                    color='blue', density=True)
        axes[0].hist(generated_samples.flatten(), bins=50, alpha=0.6, label='Generated',
                    color='red', density=True)
        axes[0].set_xlabel('Pixel Intensity')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Pixel Intensity Distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Sparsity comparison
        sparsity_orig = (self.data < 0.5).sum(axis=1).mean()
        sparsity_gen = (generated_samples < 0.5).sum(axis=1).mean()
        
        categories = ['Original', 'Generated']
        sparsities = [sparsity_orig, sparsity_gen]
        axes[1].bar(categories, sparsities, color=['blue', 'red'], alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('Average # of Near-Zero Pixels')
        axes[1].set_title('Sparsity Comparison')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Plot 3: PCA projection comparison
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
        plt.savefig(os.path.join(self.figure_dir, '2c_distribution_comparison_diffusion.png'), dpi=150)
        plt.show()
    
    def visualize_diffusion_process(self, n_steps=10):
        """Visualize the reverse diffusion process."""
        if self.diffusion_results is None:
            raise ValueError("Diffusion model not trained. Call fit_diffusion() first.")
        
        model = self.diffusion_results['model']
        scheduler = self.diffusion_results['scheduler']
        num_timesteps = self.diffusion_results['config']['num_timesteps']
        input_dim = self.data.shape[1]
        
        model.eval()
        
        # Choose timesteps to visualize
        vis_timesteps = np.linspace(num_timesteps - 1, 0, n_steps, dtype=int)
        
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(1, input_dim, device=self.device)
            
            samples_at_timesteps = []
            
            # Reverse diffusion
            for t in reversed(range(num_timesteps)):
                t_batch = torch.full((1,), t, device=self.device, dtype=torch.long)
                x = scheduler.p_sample(model, x, t_batch)
                
                if t in vis_timesteps:
                    sample = x.cpu().numpy()
                    sample = self.scaler.inverse_transform(sample)
                    sample = np.clip(sample, 0, 16)
                    samples_at_timesteps.append((t, sample.reshape(8, 8)))
        
        # Sort by timestep (descending)
        samples_at_timesteps.sort(key=lambda x: x[0], reverse=True)
        
        # Plot
        fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 2.5))
        for i, (t, img) in enumerate(samples_at_timesteps):
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f't={t}', fontsize=10)
        
        plt.suptitle('Reverse Diffusion Process (Noise → Image)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '2c_diffusion_process.png'), dpi=150)
        plt.show()
    
    def compare_with_real_samples(self, n_display=10):
        """Side-by-side comparison of real vs generated samples."""
        if self.generated_samples is None:
            self.generate_samples(n_samples=n_display)
        
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
        
        plt.suptitle('Real vs Generated Samples (Diffusion)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, '2c_real_vs_generated_diffusion.png'), dpi=150)
        plt.show()


