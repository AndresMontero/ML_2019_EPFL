import matplotlib.pyplot as plt
import numpy as np

def multiHistPlots(x,figsize = (15,15)):
    """Plot multiple histograms from the columns of a matrix.
        Args:
            x (numpy.ndarray): a matrix
    """
    n = x.shape[1]
    n_rows = np.ceil(np.sqrt(n)).astype(np.int64)
    n_cols = np.floor(np.sqrt(n)).astype(np.int64)

    if n_rows * n_cols < n:
        n_cols = np.ceil(np.sqrt(n)).astype(np.int64)

    fig, axes = plt.subplots(nrows = n_rows, ncols = n_cols, figsize = figsize)

    c = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if n > 1:
                ax = axes[row][col]
            else:
                ax = axes
            if c < x.shape[1]:
                ax.hist(x[:,c], label = 'feature_{:d}'.format(c), density = True)
                ax.legend(loc = 'upper left')
                ax.set_xlabel('Value')
                ax.set_ylabel('Probability')
            c = c + 1
    plt.show()

def plot_figures(x_dim, y_dim1, y_dim2, title, legend, y_label,x_label):
    fig = plt.figure(1, figsize=(18,12))
    plt.title(title, fontsize=18)
    plt.plot(x_dim, y_dim1)
    plt.plot(x_dim, y_dim2, 'r')
    plt.grid(True)
    plt.legend(legend)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.show()
    return fig