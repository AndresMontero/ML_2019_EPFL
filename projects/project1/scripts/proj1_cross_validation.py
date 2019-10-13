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
        y_val = relabel_y_negative(y_val)
        
        # create initial w for methods using it
        w_initial = np.zeros((x_train.shape[1], 1))

        # Use regularized logistic regression/ change for logistic_regression 
        weights,_,_ = reg_logistic_regression(y_train, x_train, w_initial, max_iters, gamma, 0.1)
            
        loss_tr = np.sqrt(2 * calculate_loss_log(y_train, x_train, weights))
        loss_val = np.sqrt(2 * calculate_loss_log(y_val, x_val, weights))

        # Append loss of this round to list
        mses_tr.append(loss_tr)
        mses_val.append(loss_val)
        
        # calculate accuracy and add it to list
        y_pred_tr = predict_labels(weights, x_train)
        y_pred_val = predict_labels(weights, x_val)
        accuracy_tr.append(np.sum(y_pred_tr == y_train)/len(y_train))
        accuracy_val.append(np.sum(y_pred_val == y_val)/len(y_val))


    mean_accuracy_tr = np.mean(accuracy_tr)
    mean_accuracy_val = np.mean(accuracy_val)
    loss_tr = np.mean(mses_tr)
    loss_val = np.mean(mses_val)
    return loss_tr, loss_val, mean_accuracy_tr, mean_accuracy_val