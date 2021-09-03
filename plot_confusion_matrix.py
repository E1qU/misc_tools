# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere Univerity
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

"""


import numpy as np
import matplotlib.pyplot as plt
import itertools

from matplotlib import rc
    
    
    

def plot_confusion_matrix(conf_mat,
                          target_names,
                          title='Confusion matrix',
                          titlesize=50,
                          colormap='Reds',
                          normalize=True,
                          num_decimals=4,
                          use_latex_font=True,
                          use_colorbar=True,
                          colorbar_labelsize=25,
                          boxtext_labelsize=50,
                          xlabel='Predicted label',
                          xlabelsize=40,
                          ylabel='True label',
                          ylabelsize=40,
                          xticks_rotation=0,
                          xticks_fontsize=25,
                          yticks_rotation=0,
                          yticks_fontsize=25,
                          figure_size=(8, 6)):
    
    """
    #####################################################################################
    Description: Plots a confusion matrix.
    #####################################################################################
    
    Input parameters:
    
    
    conf_mat: Confusion matrix in the format that sklearn.metrics.confusion_matrix produces.
    
    target_names: The names of the classes, i.e. the tick labels for the x-axis and y-axis. These classes
                  should be given in a list format, e.g. ['dog', 'cat'] if you have two different classes
                  with labels 'dog' and 'cat'.
                  
    title: The title of the confusion matrix in string format.
    
    titlesize: The font size for the title, needs to be a positive integer or a positive float.
    
    colormap: The color map of the confusion matrix from the Matplotlib library. Options: 'viridis', 'plasma',
              'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr',
              'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn' and 'YlGn'.
              
    normalize: If set to True, then the confusion matrix will be normalized. If set to False, then plain number
               of samples for each class will be displayed.
               
    num_decimals: The number of decimals that will be displayed (only for normalized confusion matrices), needs to
                  be a non-negative integer.
    
    use_latex_font: If set to True, the text in the confusion matrix will be displayed using a LaTeX format. If
                    set to False, then the traditional font of Matplotlib will be used. Note that you need to
                    have a LaTeX interpreter installed if you want to use this feature.
                    
    use_colorbar: If set to True, a colorbar will be displayed.
    
    colorbar_labelsize: Defines the label size of the colorbar, needs to be a positive integer or a positive float.
    
    boxtext_labelsize: Defines the label size of the text inside the boxes of the confusion matrix, needs to be
                       a positive integer or a positive float.
                       
    xlabel: The label for x-axis, needs to be a string.
    
    xlabelsize: Defines the label size of xlabel, needs to be a positive integer or a positive float.
    
    ylabel: The label for y-axis, needs to be a string.
    
    ylabelsize: Defines the label size of ylabel, needs to be a positive integer or a positive float.
    
    xticks_rotation: Sets the rotation angle of the tick labels for the x-axis, can be a float or an integer.
    
    xticks_fontsize: Sets the font size of the tick labels for the x-axis, needs to be a positive integer or a 
                     positive float.
                     
    yticks_rotation: Sets the rotation angle of the tick labels for the y-axis, can be a float or an integer.
    
    yticks_fontsize: Sets the font size of the tick labels for the y-axis, needs to be a positive integer or a 
                     positive float.
    
    figure_size: The width and height of the figure in inches as a tuple, e.g. (4, 2).
    
    """

    
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    if use_latex_font:
        rc('text', usetex=True)
    else:
        rc('text', usetex=False)
    
    colormap = plt.get_cmap(colormap)
        
    if normalize:
        conf_mat = np.round(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals=num_decimals)

    plt.figure(figsize=figure_size)
    plt.imshow(conf_mat, interpolation='nearest', cmap=colormap)
    plt.title(title, fontsize=titlesize)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=xticks_rotation, fontsize=xticks_fontsize)
        plt.yticks(tick_marks, target_names, rotation=yticks_rotation, fontsize=yticks_fontsize)
        
    if use_colorbar:
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=colorbar_labelsize)
    

    thresh = conf_mat.max() / 1.5 if normalize else conf_mat.max() / 2
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        if normalize:
            plt.text(j, i, f"{conf_mat[i, j]:.{num_decimals}f}",
                     horizontalalignment="center",
                     color="white" if conf_mat[i, j] > thresh else "black", fontsize=boxtext_labelsize)
        else:
            plt.text(j, i, "{:,}".format(conf_mat[i, j]),
                     horizontalalignment="center",
                     color="white" if conf_mat[i, j] > thresh else "black", fontsize=boxtext_labelsize)


    plt.tight_layout()
    plt.ylabel(ylabel, fontsize=ylabelsize)
    plt.xlabel(xlabel, fontsize=xlabelsize)
    plt.show()


    
