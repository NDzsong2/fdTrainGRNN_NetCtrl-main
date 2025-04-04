"""
This method is to plot out from the data that is loaded from the filename.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

def plot_training_testing_results(file_path_training, file_path_testing, ci_percent=.75):
    # Load data from the generated data files
    training_data = pd.read_csv(file_path_training)
    testing_data = pd.read_csv(file_path_testing)
    
    # Determine number of plots (nodes) for each case
    num_train_plots = len(training_data.columns) - 1  # Exclude 'epoch'
    num_test_plots = len(testing_data.columns) - 1
    
    ### First Figure: Training Loss
    plt.figure(figsize=(8, 6))
    for i in range(num_train_plots):
        plt.semilogy(training_data.index, 
                     training_data[f'training_loss{i+1}'], 
                     alpha=0.8, 
                     label=f'Training Loss {i+1}')
        # color = p_train[0].get_color()
        # lower_ci = np.percentile(training_data[f'training_loss{i+1}'], 100*(1-ci_percent), axis=0)  # Lower quantile
        # upper_ci = np.percentile(training_data[f'training_loss{i+1}'], 100*(ci_percent), axis=0)    # Upper quantile
        # plt.fill_between(training_data.index, lower_ci, upper_ci, color=color, alpha=0.3)
        
    
    plt.title('Training Loss vs. Epochs', fontsize = 16)
    plt.xlabel('Epoch', fontsize = 14)
    plt.ylabel('Training Loss', fontsize = 14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(training_data.index.min(), training_data.index.max())
    plt.legend(fontsize = 14)
    plt.grid(True)
    # plt.show()  # Show training loss figure


    ### Second Figure: Testing Loss
    plt.figure(figsize=(8, 6))
    for i in range(num_test_plots):
        plt.semilogy(testing_data.index, 
                     testing_data[f'testing_loss{i+1}'], 
                     alpha=0.8, 
                     label=f'Testing Loss {i+1}')
    
    plt.title('Testing Loss vs. Epochs', fontsize = 16)
    plt.xlabel('Epoch', fontsize = 14)
    plt.ylabel('Testing Loss', fontsize = 14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(testing_data.index.min(), testing_data.index.max())
    plt.legend(fontsize = 14)
    plt.grid(True)

    plt.draw()  # Draw all figures
    plt.pause(0.1)  # Pause for a short time (to display both)

    plt.show()  # Show testing loss figure















