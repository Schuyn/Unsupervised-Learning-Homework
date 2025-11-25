'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-11-24 19:58:17
LastEditTime: 2025-11-25 09:37:20
FilePath: /Unsupervised-Learning-Homework/Homework 3/Code/Prob2_main.py
Description: 
    Main script to run data processing and clustering model fitting for Homework 3.
    Optimized flow:
    1. process_data() -> X, y
    2. fit_kde_models() -> trained models
    3. generate_samples() -> generated data
    4. evaluate_samples() -> metrics
    5. visualize_generated_samples() + compare_distributions()
'''
from Prob2a_utils import Prob2Analysis


def main():
    # Initialize analysis
    p2 = Prob2Analysis(
        output_dir='Homework 3/Code/Data',
        figure_dir='Homework 3/Latex/Figures'
    )
    X, y = p2.process_data(verbose=False)
    
    kde_results = p2.fit_kde_models(
        spaces=['pca_20', 'pca_50', 'original'],
        bandwidth_range=None,  # Auto-generate
        kernel='gaussian',
        cv_folds=5,
        verbose=False
    )
    
    # Generate from PCA-20 space (fastest and often best quality)
    generated_pca20 = p2.generate_samples(
        space='pca_20',
        n_samples=100,
        verbose=True
    )
    
    metrics = p2.evaluate_samples(verbose=True)
    
    # Visualization and comparison
    print("\nVisualizing generated samples...")
    p2.visualize_generated_samples(n_display=25)
    
    print("\nComparing original vs generated distributions...")
    p2.compare_distributions()
    

if __name__ == "__main__":
    main()


