import numpy as np
from implementations import *
from proj1_utils import *

def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold.
    
        Args:
            y       (numpy.ndarray): the ground truth labels
            k_fold  (float)        : the number of folds
            seed    (int)          : the seed for the random number generator
        Returns:
            numpy.ndarray: the indices for each fold
    """  
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_lambda_ridge_reg(y, x, k_indices, k, lambda_):
    """Return the training and test loss of ridge regression for lambda 
       crossvalidation.
    
        Args:
            y          (numpy.ndarray): the ground truth labels
            x          (numpy.ndarray): the features
            k_indices  (numpy.ndarray): the indices for the k fold
            k          (int)          : the current fold
            lambda_    (float)        : the regularization coefficient
        Returns:
            numpy.ndarray: the training losses
            numpy.ndarray: the test losses
    """

    train_indices = np.delete(k_indices,k,axis = 0).ravel()
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[k_indices[k]]
    y_test = y[k_indices[k]]
     
    w_star, loss = ridge_regression(y_train,x_train,lambda_)
    
    loss_tr = loss
    loss_te = compute_mse(y_test,x_test,w_star)
    return loss_tr, loss_te

def cross_validation_degree_lambda_ridge_reg(y, x, k_indices, k, degree, lambda_):
    """Return the training and test loss of ridge regression for degree and lambda 
       crossvalidation.

       This function assumes that the feature matrix x has a one hot encoded 
       categorical feature at its 3 last columns
       
       Args:
            y          (numpy.ndarray): the ground truth labels
            x          (numpy.ndarray): the features
            k_indices  (numpy.ndarray): the indices for the k fold
            k          (int)          : the current fold
            degree     (int)          : the degree of the polynomial features
            lambda_    (float)        : the regularization coefficient
        Returns:
            numpy.ndarray: training losses
            numpy.ndarray: test losses
    """

    train_indices = np.delete(k_indices,k,axis = 0).ravel()
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[k_indices[k]]
    y_test = y[k_indices[k]]

    # Remove categorical cols
    # Categorical columns are always the last ones by convention
    # There are 3 categorical columns due to one hot encondig of PRI_jet_num = {0,1,2,3}
    N_CAT_COLS = 3
    cat_cols = x_train.shape[1] - np.asarray(range(1,N_CAT_COLS+1))
    x_train_num, x_train_cat = split_numerical_categorical(x_train,cat_cols)
    x_test_num, x_test_cat = split_numerical_categorical(x_test,cat_cols)

    ext_x_train = build_poly(x_train_num, degree)
    ext_x_test = build_poly(x_test_num, degree)

    ext_x_train = np.hstack((ext_x_train,x_train_cat))
    ext_x_test = np.hstack((ext_x_test,x_test_cat))
     
    w_star, loss = ridge_regression(y_train,ext_x_train,lambda_)
    
    loss_tr = loss
    loss_te = compute_mse(y_test,ext_x_test,w_star)
    return loss_tr, loss_te

def cross_validation_ridge_reg_best_lambda(y, x, lambdas, k_fold, seed = 1):    
    """Return the best lambda, the training and test rmse for ridge regression 
        crossvalidation.
    
    Args:
        y          (numpy.ndarray): the ground truth labels
        x          (numpy.ndarray): the features
        k_indices  (numpy.ndarray): the indices for the k fold
        k          (int)          : the current fold
        degree     (int)          : the degree of the polynomial features
        lambda_    (float)        : the regularization coefficient
    Returns:
        numpy.ndarray: training losses
        numpy.ndarray: test losses
    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
   
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
      
    for lambda_ in lambdas:
        fold_rmse_tr = []
        fold_rmse_te = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation_lambda_ridge_reg(y,x,k_indices,k,lambda_)
            fold_rmse_tr.append(np.sqrt(2*loss_tr))
            fold_rmse_te.append(np.sqrt(2*loss_te))
        rmse_tr.append(np.mean(fold_rmse_tr))
        rmse_te.append(np.mean(fold_rmse_te))
    return lambdas[np.argmin(rmse_te)], rmse_tr, rmse_te

def cross_validation_ridge_reg_best_degree_best_lambda(y, x, degrees, lambdas, k_fold, seed = 1):    
    """Return the best degree, the best lambda, the training and test rmse for ridge regression 
        crossvalidation.
       
       Args:
            y          (numpy.ndarray): the ground truth labels
            x          (numpy.ndarray): the features
            k_indices  (numpy.ndarray): the indices for the k fold
            k          (int)          : the current fold
            degree     (int)          : the degree of the polynomial features
            lambda_    (float)        : the regularization coefficient
        Returns:
            numpy.ndarray: training losses
            numpy.ndarray: test losses
    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
   
    # define lists to store the loss of training data and test data
    rmse_tr = np.zeros((len(degrees),len(lambdas))) 
    rmse_te = np.zeros((len(degrees),len(lambdas)))
    
    for (i,degree) in enumerate(degrees):
        for (j,lambda_) in enumerate(lambdas):
            fold_rmse_tr = []
            fold_rmse_te = []
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation_degree_lambda_ridge_reg(y,x,k_indices,k, degree,lambda_)
                fold_rmse_tr.append(np.sqrt(2*loss_tr))
                fold_rmse_te.append(np.sqrt(2*loss_te))
            rmse_tr[i,j] = np.mean(fold_rmse_tr)
            rmse_te[i,j] = np.mean(fold_rmse_te)
    min_val = np.min(rmse_te)
    min_val_idxs = np.where(rmse_te == min_val)
    best_degree = degrees[min_val_idxs[0][0]]
    best_lambda = lambdas[min_val_idxs[1][0]]
    return best_degree,best_lambda, rmse_tr, rmse_te