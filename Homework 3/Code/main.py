'''
Author: Chuyang Su cs4570@columbia.edu
Date: 2025-11-23 17:41:45
LastEditTime: 2025-11-23 17:41:51
FilePath: /Unsupervised-Learning-Homework/Homework 3/Code/main.py
Description: 
    Main script to run data processing and graphical model fitting for Homework 3.
'''
from hw3_utils import process_stock_data

def main():
    FILENAME = 'log_returns.csv'
    log_returns = process_stock_data(filepath=FILENAME, verbose=True)


if __name__ == "__main__":
    main()