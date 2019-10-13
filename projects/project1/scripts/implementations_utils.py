import numpy as np


################################################## Utils for Logistic and Regularized Logistic Regression
def sigmoid(t):
    """Apply sigmoid function on t
    
    Args: 
        t=>(numpy.array): Values to apply sigmoid function
    
    Returns:
        => numpy.array: Calculated values of sigmoid
    """
    
    return 1.0 / (1.0 + np.exp(-t))


def calculate_loss_log(y, tx, w):
    """Compute the cost of log_regression
    
    Args: 
        y =>(numpy.array): Target values
        tx =>(numpy.array): Transposed features
        w => (numpy.array): Weigths 
          
    Returns:
        => numpy.array: Calculated loss
    """
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

    
def calculate_gradient_log(y, tx, w,):
    """Compute the gradient of loss for log_regression
    
    Args: 
        y =>(numpy.array): Target values
        tx => (numpy.array): Transposed features
        w => (numpy.array): Weigths 
          
    Returns:
        => numpy.array: Calculated logistic gradient
    """

    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


def learning_by_gradient_descent_log(y, tx, w, gamma):
    """Compute the gradient descen using logistic regression
    
    Args: 
        y =>(numpy.array): Target values
        tx => (numpy.array): Transposed features
        w => (numpy.array): Weigths 
        gamma=> (float): the gamma to use.
        
    Returns:
        w =>(numpy.array): Calculated Weights
        loss => (numpy.array): Calculated Loss
    """

    loss = calculate_loss_log(y, tx, w) 
    grad = calculate_gradient_log(y, tx, w)
    w -= gamma * grad
    return loss, w


def learning_by_reg_gradient_descent_log(y, tx, w, gamma, lambda_):
    """Compute the gradient descen using logistic regression
    
    Args: 
        y =>(numpy.array): Target values
        tx => (numpy.array): Transposed features
        w => (numpy.array): Weigths 
        gamma=> (float): the gamma to use.
        
    Returns: 
        w =>(numpy.array): Calculated Weights
        loss => (numpy.array): Calculated Loss
    """

    loss = calculate_loss_log(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    grad = calculate_gradient_log(y, tx, w) + 2 * lambda_ * w
    w -= gamma * grad
    return loss, w
