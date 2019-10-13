import numpy as np

def compute_mse(y, tx, w):
    """Compute the Mean Squared Error (MSE).

    Args:
        y  (numpy.ndarray): the ground truth labels
        tx (numpy.ndarray): the features
        w  (numpy.ndarray): the weights
    Returns:
        numpy.float64: the MSE 
    """
    e=y-np.dot(tx,w)
    return (1/(2*y.shape[0]))*np.dot(e.T,e)

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