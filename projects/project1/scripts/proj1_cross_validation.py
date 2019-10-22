import numpy as np
from implementations import *
from proj1_utils import *
from proj1_helpers import *

def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold.

       Args: 
            y      (numpy.ndarray): ground truth labels
            k_fold (int)          : number of current fold 
            seed   (int)          : seed for random methods
        Returns:
            numpy.ndarray         : array of indices for the k-fold
    """

    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def degree_lambda_grid_search(y, x, cat_cols, ratio_train, method_flag, degrees, lambdas, gamma = 1e-6, max_iters = 2000, seed = 10):
    """Return the best degree and lambda for a specified method

        It is similar to crossvalidationo, however there is 

        Args:
            y           (numpy.array)   : ground truth labels
            x           (numpy.ndarray) : clean data
            cat_cols    (list)          : categorical columns
            ratio_train (float)         : proportion of data to use for training
            method_flag (int)           : flag indicating the method to use
            degrees     (numpy.ndarray) : degrees to evaluate
            lambdas     (numpy.ndarray) : lambdas to evaluate
            seed        (int)           : seed for random methods
        Returns
            int           : best degree
            float         : best lambda
            float         : accuracy score
            numpy.ndarray : accuracy scores grid
    """
    x_train, y_train, x_val, y_val = split_data(x, y, ratio_train, seed)

    x_train_num, x_train_cat = split_numerical_categorical(x_train, cat_cols)
    x_val_num, x_val_cat = split_numerical_categorical(x_val, cat_cols)

    x_train_num, mean, std = preprocess_num_features_fit(x_train_num)
    x_val_num = preprocess_num_features_transform(x_val_num, mean, std)

    x_train_ohe_cat = one_hot_encode(x_train_cat)
    x_val_ohe_cat = one_hot_encode(x_val_cat)
      
    accuracy_scores_grid = np.zeros((len(degrees),len(lambdas)))
    w_initial = np.zeros((x_train.shape[1]))
    for (j,lambda_ )in enumerate(lambdas):
        for (i,degree) in enumerate(degrees):
            ext_x_train_num = build_poly(x_train_num, degree)
            ext_x_val_num = build_poly(x_val_num, degree)
            x_train = np.hstack((ext_x_train_num,x_train_ohe_cat))
            x_val = np.hstack((ext_x_val_num, x_val_ohe_cat))
            
            if method_flag == 1:
                # Least squares GD
                w_star, loss = least_squares_GD(y_train,x_train,w_initial,max_iters,gamma)
                
            elif method_flag == 2:
                # Least squares SGD
                w_star, loss = least_squares_SGD(y_train,x_train,w_initial,max_iters,gamma)
        
            elif method_flag == 3:
                # Least squares
                w_star, loss = least_squares(y_train,x_train)

            elif method_flag == 4:
                # Ridge regression
                w_star, loss = ridge_regression(y_train,x_train,lambda_)

            elif method_flag == 5:
                # Logistic regression
                y_train = relabel_y_non_negative(y_train)
                w_star, loss = logistic_regression(y_train,x_train,w_initial,max_iters,gamma)

            elif method_flag == 6:
                # Regularized logistic regression
                y_train = relabel_y_non_negative(y_train)
                w_star, loss = reg_logistic_regression(y_train,x_train,w_initial,max_iters,gamma,lambda_)
            if loss == np.nan:
                break
            y_pred = predict_labels(w_star,x_val)
            if method_flag == 5 or method_flag == 6:
                y_pred = relabel_y_negative(y_pred)
            accuracy_scores_grid[i,j] = get_accuracy_score(y_pred,y_val)
            degree_idx, lambda_idx = np.unravel_index(np.argmax(accuracy_scores_grid),accuracy_scores_grid.shape)
            accuracy_score = accuracy_scores_grid[degree_idx, lambda_idx]
            best_degree = degrees[degree_idx]
            best_lambda = lambdas[lambda_idx]
    return best_degree, best_lambda, accuracy_score, accuracy_scores_grid


def cross_validation(y, x, cat_cols, method_flag, k_indices, k, degree = None, lambda_ = None,  gamma = None, max_iters = None):
    """Return the train losses, test losses and accuracy score of the selected method for the current fold

        Args: 
            y           (numpy.ndarray): ground truth labels
            x           (numpy.ndarray): preprocessed features
            cat_cols    (list)         : list of categorical columns
            method_flag (int)          : flag indicating the method to use
            k_indices   (numpy.ndarray): indices to use to build the validation set
            k           (int)          : current fold
            lambda_     (float)        : regularization coefficient
            degree      (int)          : degree of the polynomial
            gamma       (float)        : learning rate
            max_iters   (int)          : maximum number of iterations
        Returns:
            float: training loss
            float: validation loss
            float: accuracy score
    """
    train_indices = np.delete(k_indices,k,axis = 0).ravel()
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_val = x[k_indices[k]]
    y_val = y[k_indices[k]]

    x_train_num, x_train_cat = split_numerical_categorical(x_train, cat_cols)
    x_val_num, x_val_cat = split_numerical_categorical(x_val, cat_cols)

    x_train_num, mean, std = preprocess_num_features_fit(x_train_num)
    x_val_num = preprocess_num_features_transform(x_val_num, mean, std)

    x_train_ohe_cat = one_hot_encode(x_train_cat)
    x_val_ohe_cat = one_hot_encode(x_val_cat)

    ext_x_train_num = build_poly(x_train_num, degree)
    ext_x_val_num = build_poly(x_val_num, degree)

    x_train = np.hstack((ext_x_train_num,x_train_ohe_cat))
    x_val = np.hstack((ext_x_val_num, x_val_ohe_cat))

    w_initial = np.zeros((x_train.shape[1]))
    if method_flag == 1:
        # Least squares GD
        w_star, loss_tr = least_squares_GD(y_train,x_train,w_initial,max_iters,gamma)
    elif method_flag == 2:
        # least squares SGD
        w_star, loss_tr = least_squares_SGD(y_train,x_train,w_initial,max_iters,gamma)
    elif method_flag == 3:
        # least squares 
        w_star, loss_tr = least_squares(y_train,x_train)
    elif method_flag == 4:
        # Ridge regression
        w_star, loss_tr = ridge_regression(y_train,x_train,lambda_)      
    elif method_flag == 5:
        # Logistic regression
        y_train = relabel_y_non_negative(y_train)
        w_star, loss_tr = logistic_regression(y_train,x_train,w_initial,max_iters,gamma)
    elif method_flag == 6:
        # Regularized logistic regression
        y_train = relabel_y_non_negative(y_train)
        w_star, loss_tr = reg_logistic_regression(y_train,x_train,w_initial,max_iters,gamma,lambda_)
    y_pred = predict_labels(w_star,x_val)
    if method_flag == 5 or method_flag == 6:
        loss_va = calculate_loss_log(y_val, x_val, w_star)
        y_pred = relabel_y_negative(y_pred)
    else:
        loss_va = np.sqrt(2*compute_mse(y_val,x_val,w_star))
    accuracy_score = get_accuracy_score(y_pred,y_val)
    return loss_tr, loss_va, accuracy_score

def k_fold_cross_validation(y, x, cat_cols, method_flag, k_fold, degree = None, lambda_ = None, gamma = None, max_iters = None, seed = 1):
    """Return the train and validation losses and the accuracy score
    
        Args:
            y           (numpy.ndarray): ground truth labels
            x           (numpy.ndarray): preprocessed features
            cat_cols    (list)         : list of categorical columns
            method_flag (int)          : flag indicating the method to use
            k_fold      (int)          : number of folds
            degree      (int)          : degree of the polynomial
            lambda_     (float)        : regularization coefficient
            gamma       (float)        : learning rate
            max_iters   (int)          : maximum number of iterations
            seed        (float)        : seed for random methods
        Returns:
            numpy.ndarray: training losses
            numpy.ndarray: validation losses
            numpy.ndarray: accuracy scores   
    """
    k_indices = build_k_indices(y,k_fold,seed)
    losses_tr = np.zeros(k_fold)
    losses_va = np.zeros(k_fold)
    accuracy_scores = np.zeros(k_fold)
    for k in range(k_fold):
        loss_tr, loss_va, accuracy_score = cross_validation(y,x,cat_cols,method_flag,k_indices,k,degree,lambda_,gamma,max_iters)
        losses_tr[k] = loss_tr
        losses_va[k] = loss_va
        accuracy_scores[k] = accuracy_score
    return losses_tr, losses_va, accuracy_scores


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