import numpy as np
from implementations import *
from proj1_utils import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""

    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_lambda_ridge_reg(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression."""

    train_indices = np.delete(k_indices,k,axis = 0).ravel()
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[k_indices[k]]
    y_test = y[k_indices[k]]
     
    w_star, loss = ridge_regression(y_train,x_train,lambda_)
    
    loss_tr = loss
    loss_te = compute_mse(y_test,x_test,w_star)
    return loss_tr, loss_te