import numpy as np
from implementations import *
from proj1_utils import *
from proj1_helpers import *

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

def cross_validation_log(x, y, lambda_=0, gamma=0.001, max_iters=1000, k_fold=int(5), seed=28):
    """Train the model and evaluate loss based on cross validation
    
    Args:
        x (numpy.array): Transposed x values
        y (numpy.array): y values
        lambda_ (float): float for the regularization
        gamma (float): learning rate
        max_iters(int) : max number of iterations
        k_fold(int): number of k_folds
        seed: seed for random definitions
        
    Returns:
        loss_tr (float): loss of the training set
        loss_val (float): loss of the validation set
        mean_accuracy_tr (float): mean accuracy of the model training set
        mean_accuracy_val (float): mean accuracy of the model validation set
    
    """
    mses_tr = []
    mses_val = []
    accuracy_tr = []
    accuracy_val = []
    
    k_indices = build_k_indices(y, k_fold, seed);
    for i in range(k_fold):
        newk_index = np.delete(k_indices, i, 0)
        indices_train = newk_index.ravel()
        indices_val = k_indices[i]

        # Train data at each iteration "i" of the loop
        x_train = x[indices_train]
        y_train = y[indices_train]

        # Validate the data at each iteration "i" of the loop
        x_val = x[indices_val]
        y_val = y[indices_val]
        
        # create initial w for methods using it
        w_initial = np.zeros((x_train.shape[1], 1))

        # Use regularized logistic regression/ change for logistic_regression 
        weights,_ = reg_logistic_regression(y_train, x_train, w_initial, max_iters, gamma, lambda_)
            
        loss_tr = np.sqrt(2 * calculate_loss_log(y_train, x_train, weights))
        loss_val = np.sqrt(2 * calculate_loss_log(y_val, x_val, weights))

        # Append loss of this round to list
        mses_tr.append(loss_tr)
        mses_val.append(loss_val)
        
        # calculate accuracy and add it to list
        y_pred_tr = predict_labels(weights, x_train)
        y_pred_val = predict_labels(weights, x_val)
        y_train = relabel_y_negative(y_train)
        y_val = relabel_y_negative(y_val)

        accuracy_tr.append(np.sum(y_pred_tr == y_train)/len(y_train))
        accuracy_val.append(np.sum(y_pred_val == y_val)/len(y_val))


    mean_accuracy_tr = np.mean(accuracy_tr)
    mean_accuracy_val = np.mean(accuracy_val)
    loss_tr = np.mean(mses_tr)
    loss_val = np.mean(mses_val)
    return loss_tr, loss_val, mean_accuracy_tr, mean_accuracy_val