import numpy as np
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