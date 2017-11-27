# -*- coding: utf-8 -*-
"""
Created on Sun Jan 03 17:56:19 2016

@author: Gabi
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_mnist_digit(image):
    """ Plot a single MNIST image."""
    image = np.reshape(image, [28,28])    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()