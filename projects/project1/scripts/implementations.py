import numpy as np
from implementations_utils import *



def compute_mse(y, tx, w):
    """Calculate the mse
    """
    e=y-np.dot(tx,w)
    return (1/(2*y.shape[0]))*np.dot(e.T,e)

def least_squares(y, tx):
    """Calculate the least squares solution."""
    w_star = np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y))
    loss = compute_mse(y,tx,w_star)
    return w_star, loss

def ridge_regression(y, tx, lambda_):
    """Implement ridge regression."""
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
    losses =[]
    for n_iter in range(max_iters+1):
        loss, w = learning_by_gradient_descent_log(y, tx, w, gamma)
        if (n_iter % print_every == 0):
            # print average loss for the last print_every iterations
            print(f"#Iteration: {n_iter}, Loss: {loss}")
            losses.append(loss)

            
    loss = learning_by_gradient_descent_log(y, tx, w, gamma)
    
    return w, loss,losses

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
    losses =[]
    for n_iter in range(max_iters+1):
        loss, w = learning_by_reg_gradient_descent_log(y, tx, w, gamma,lambda_)
        if (n_iter % print_every == 0):
            # print average loss for the last print_every iterations
            print(f"#Iteration: {n_iter}, Loss: {loss}")
            losses.append(loss)

            
    loss = learning_by_reg_gradient_descent_log(y, tx, w, gamma,lambda_)
    
    return w, loss,losses