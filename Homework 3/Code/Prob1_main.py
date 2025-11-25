'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-11-23 17:41:45
LastEditTime: 2025-11-24 12:27:59
FilePath: /Unsupervised-Learning-Homework/Homework 3/Code/main.py
Description: 
    Main script to run data processing and graphical model fitting for Homework 3.
'''
from Prob1_utils import Prob1Analysis

def main():
    # Problem 1a
    p1=Prob1Analysis()
    log_returns = p1.process_stock_data(verbose=False)

    # Problem 1b
    glasso_results = p1.fit_glasso_models(verbose=False)

    # Problem 1c
    pc_results = p1.fit_pc_model(verbose=False)
    
    # Problem 1e
    granger_results = p1.fit_granger_model(verbose=True)
    
if __name__ == "__main__":
    main()