'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-11-25 15:08:40
LastEditTime: 2025-11-25 20:54:04
FilePath: /Unsupervised-Learning-Homework/Homework 3/Code/Prob2c_main.py
Description: 
    Main script to run Diffusion Model fitting for Homework 3 Problem 2c.
    Optimized flow:
    1. process_data() -> X, y
    2. tune_hyperparameters() (optional) -> best config
    3. fit_diffusion() -> trained model
    4. generate_samples() -> generated data
    5. evaluate_samples() -> metrics
    6. visualize_generated_samples() + compare_distributions()
'''
from Prob2c_utils import Prob2cAnalysis


def main():
    # Initialize analysis (with seed for reproducibility)
    p2c = Prob2cAnalysis(
        output_dir='Homework 3/Code/Data',
        figure_dir='Homework 3/Latex/Figures',
        seed=25
    )
    
    # Load and preprocess data
    X, y = p2c.process_data(verbose=False)
    
    # Hyperparameter tuning
    # tuning_results = p2c.tune_hyperparameters(
    #     num_timesteps_values=[200, 500, 1000],
    #     lr_values=[1e-4, 5e-4, 1e-3],
    #     hidden_configs=[
    #         [128, 256, 128],
    #         [256, 512, 256],
    #     ],
    #     n_epochs=150,
    #     verbose=False
    # ) # Best results: lr=0.0005, timesteps=500, architecture=[256, 512, 256]
    
    # Train Diffusion Model with selected hyperparameters
    diffusion_results = p2c.fit_diffusion(
        num_timesteps=500,
        hidden_dims=[256, 512, 256],
        time_emb_dim=64,
        lr=5e-4,
        batch_size=128,
        n_epochs=200,
        beta_start=1e-4,
        beta_end=0.02,
        verbose=False
    )
    
    # Generate samples
    generated_samples = p2c.generate_samples(
        n_samples=100,
        verbose=True
    )
    
    # Evaluate generated samples
    metrics = p2c.evaluate_samples(verbose=True)
    
    # Visualization
    print("\nVisualizing generated samples...")
    p2c.visualize_generated_samples(n_display=25)
    
    print("\nComparing original vs generated distributions...")
    p2c.compare_distributions()
    
    print("\nVisualizing reverse diffusion process...")
    p2c.visualize_diffusion_process(n_steps=10)
    
    print("\nComparing real vs generated samples side-by-side...")
    p2c.compare_with_real_samples(n_display=10)


if __name__ == "__main__":
    main()


