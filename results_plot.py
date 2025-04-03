"""
This method is to plot out from the data that is loaded from the filename.
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

def plot_training_results(filename, log_interval, ci_percent=.75):
    # Load data
    with open(filename) as f:
        data = json.load(f)
    print('Sparse edges:', np.mean(data['num_sparse_edges']))
    print('Env edges:', np.mean(data['num_env_edges']))

    # Plot the loss decrease
    cost_table = np.array(data['cost_table'])
    print('Average Costs:', np.median(cost_table[:,:,-1], 0))    # The last cost (for all models) in the cost_table
    print('Autonomous Costs:', np.mean(data['auto_costs']))
    names = ['GRNN-Fixed', 'GRNN', 'GRNN-Full', 'GCNN', 'GRNN-Sparse']
    # name_index = [0,1,2,4,3]
    name_index = [3]
    for i in name_index:
        if i == 0:
            continue
        cost = cost_table[:,i,:]
        median_cost = np.median(cost, axis=0)                # Median of the cost
        lower_ci = np.quantile(cost, 1-ci_percent, axis=0)   # Lower quantile 25%
        upper_ci = np.quantile(cost, ci_percent, axis=0)     # Upper quantile 75%
        ind = np.arange(len(median_cost)) * log_interval
        p = plt.semilogy(ind, median_cost, label=names[i])   # Make a plot with log scaling on the y-axis.
        color = p[0].get_color()
        plt.fill_between(ind, lower_ci, upper_ci, color=color, alpha=0.3)   # The curves are defined by the points (x, y1) and (x, y2). This creates one or multiple polygons describing the filled area.

    plt.title(r'Performance vs. Epoches ($\|A\|_2={}$)'.format(data['Anorm']))   # 'r' means raw string. When a 'R' or 'r' is present before a string, a character following a backslash included in the string without any change.
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Cost')
    plt.ylim([1,4.5])
    plt.legend()   # Automatic detection of elements to be shown in the legend.
    plt.show()     # Display all open figures.













