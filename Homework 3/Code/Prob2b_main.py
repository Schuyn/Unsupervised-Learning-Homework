'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-11-25 10:49:06
LastEditTime: 2025-11-25 13:28:54
FilePath: /Unsupervised-Learning-Homework/Homework 3/Code/Prob2b_main.py
Description: 
    Main script to run GAN model fitting for Homework 3 Problem 2b.
    Optimized flow:
    1. process_data() -> X, y
    2. tune_hyperparameters() (optional) -> best config
    3. fit_gan() -> trained GAN
    4. generate_samples() -> generated data
    5. evaluate_samples() -> metrics
    6. visualize_generated_samples() + compare_distributions()
'''
from Prob2b_utils import Prob2bAnalysis


def main():
    # Initialize analysis
    p2b = Prob2bAnalysis(
        output_dir='Homework 3/Code/Data',
        figure_dir='Homework 3/Latex/Figures'
    )
    
    # Load and preprocess data
    X, y = p2b.process_data(verbose=False)
    
    # Hyperparameter tuning
    tuning_results = p2b.tune_hyperparameters(
        latent_dims=[32, 64, 128],
        hidden_configs=[
            ([128, 256], [256, 128]),
            ([256, 512], [512, 256]),
        ],
        n_epochs=200,
        verbose=False
    )
    # Best parameters from tuning: latent_dim=32, lr_g=0.0001, G=[128, 256]
    
    gan_results = p2b.fit_gan(
        latent_dim=32,
        g_hidden_dims=[128, 256],
        d_hidden_dims=[256, 128],
        lr_g=0.0001,
        lr_d=0.0002,
        batch_size=64,
        n_epochs=300,
        n_critic=1,
        beta1=0.5,
        label_smoothing=0.1,
        verbose=False
    )
    
    # Generate samples
    generated_samples = p2b.generate_samples(
        n_samples=100,
        verbose=False
    )
    
    # Evaluate generated samples
    metrics = p2b.evaluate_samples(verbose=True)
    
    # Visualization
    print("\nVisualizing generated samples...")
    p2b.visualize_generated_samples(n_display=25)
    
    print("\nComparing original vs generated distributions...")
    p2b.compare_distributions()
    
    print("\nVisualizing latent space interpolation...")
    p2b.visualize_latent_interpolation(n_steps=10)
    
    print("\nComparing real vs generated samples side-by-side...")
    p2b.compare_with_real_samples(n_display=10)


if __name__ == "__main__":
    main()


