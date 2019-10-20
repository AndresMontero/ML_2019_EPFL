import numpy as np
from implementations_utils import *
from proj1_helpers import *

############################### Linear Regression - iterative models

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Calculate the loss and weights with gradient descent
       linear regression
    Args:
        y  (numpy.ndarray): the ground truth labels
        tx (numpy.ndarray): the features
        initial_w  (numpy.ndarray): the initial weights
        max_iters (int): number of iterations
        gamma  (float): learning rate
    
    Returns: 
        w numpy.ndarray: the optimum weights
        loss float: the MSE   
    """
    loss = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y,tx,w)
        w = w-gamma*gradient
        if (n_iter % 50 == 0):
            loss = compute_mse(y,tx,w)
            print("GD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter+1, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w,loss
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Calculate the loss and weights with stochastic gradient descent
       linear regression
    Args:
        y  (numpy.ndarray): the ground truth labels
        tx (numpy.ndarray): the features
        initial_w  (numpy.ndarray): the initial weights
        max_iters (int): number of iterations
        batch_size (int): batch size - number of columns
        gamma  (float): learning rate
    
    Returns: 
        w numpy.ndarray: the optimum weights
        loss float: the MSE   
    """
    batch_size = 1
    loss = []
    w = initial_w
    for n_iter in range(1,max_iters+1):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient_mse(minibatch_y,minibatch_tx,w)
            w = w-gamma*gradient
            if (n_iter % 50 == 0):
                loss = compute_mse(minibatch_y,minibatch_tx,w)
                print("SGD({bi}/{ti}): loss={l}, w(0)={w0}, w1={w1}".format(
                      bi=n_iter, ti=max_iters, l=loss, w0=w[0], w1=w[1]))
    return w,loss

################################### Least Squares

def least_squares(y, tx):
    """Calculate the least squares solution.
    Args:
        y  (numpy.ndarray): the ground truth labels
        tx (numpy.ndarray): the features
    Returns: 
        numpy.ndarray: the optimum weights
        numpy.float64: the MSE   
    """
    w_star = np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y))
    loss = compute_mse(y,tx,w_star)
    return w_star, loss

def ridge_regression(y, tx, lambda_):
    """Calculate the ridge regression solution.
    
    Args:
        y       (numpy.ndarray): the ground truth labels
        tx      (numpy.ndarray): the features
        lambda_ (float)        : the regularization coefficient
    Returns:
        numpy.ndarray: the optimum weights
        numpy.float64: the MSE         
    """
    lambda_prime = 2 * lambda_ * y.shape[0]
    w_star = np.linalg.solve(np.dot(tx.T,tx) + np.eye(tx.T.shape[0])*lambda_prime,np.dot(tx.T,y))
    loss = compute_mse(y,tx,w_star)
    return w_star, loss
    
################################### Logistic Regression
def logistic_regression(y, tx, w_initial, max_iters, gamma):
    """Implement logistic regression using gradient descent
    
    Args: 
      y =>(numpy.array): Target values
      tx => (numpy.array): Transposed features
      w_initial => (numpy.array): Initial Weigths 
      max_iters => (int): number of iterations.
      gamma=> (float): the gamma to use.
          
    Returns: 
        w =>(numpy.array): Calculated Weights
        loss => (numpy.array): Calculated Loss
    """

    assert max_iters > 0, "max_iters should be a positive number"
    assert y.shape[0] == tx.shape[0], "y and tx should have the same number of entries (rows)"
    assert tx.shape[1] == w_initial.shape[0], "initial_w should be the same degree as tx"
    
    print_every = 250
    w = w_initial
    for n_iter in range(max_iters+1):
        loss, w = learning_by_gradient_descent_log(y, tx, w, gamma)
        if (n_iter % print_every == 0):
            # print average loss for the last print_every iterations
            print(f"#Iteration: {n_iter}, Loss: {loss}")
      
    loss = learning_by_gradient_descent_log(y, tx, w, gamma)
    
    return w, loss

################################### Regularized Logistic Regression

def reg_logistic_regression(y, tx, w_initial, max_iters, gamma,lambda_):
    """Implement logistic regression using gradient descent
    
    Args: 
        y =>(numpy.array): Target values
        tx => (numpy.array): Transposed features
        w_initial => (numpy.array): Initial Weigths 
        max_iters => (int): number of iterations.
        gamma=> (float): the gamma to use.
          
    Returns: 
        w =>(numpy.array): Calculated Weights
        loss => (numpy.array): Calculated Loss
    """

    assert max_iters > 0, "max_iters should be a positive number"
    assert y.shape[0] == tx.shape[0], "y and tx should have the same number of entries (rows)"
    assert tx.shape[1] == w_initial.shape[0], "initial_w should be the same degree as tx"
    
    print_every = 250
    w = w_initial

    for n_iter in range(max_iters+1):
        loss, w = learning_by_reg_gradient_descent_log(y, tx, w, gamma,lambda_)
        if (n_iter % print_every == 0):
            # print average loss for the last print_every iterations
            print(f"#Iteration: {n_iter}, Loss: {loss}")
               
    loss = learning_by_reg_gradient_descent_log(y, tx, w, gamma,lambda_)
    
    return w, loss