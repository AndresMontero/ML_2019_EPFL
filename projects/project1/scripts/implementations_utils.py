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

    loss = calculate_loss_log(y, tx, w) + (lambda_/2) * np.linalg.norm(w,2)
    grad = calculate_gradient_log(y, tx, w) + lambda_ * w
    w -= gamma * grad
    return loss, w

##########
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
            
def compute_gradient_mse(y, tx, w):
    """Compute the Gradient (MSE).
    Args:
        y  (numpy.ndarray): the ground truth labels
        tx (numpy.ndarray): the features
        w  (numpy.ndarray): the weights
    Returns:
        numpy.float64: the Gradient
    """
    error = y - np.dot(tx,w)
    return -1/len(y)*np.dot(tx.T,error)

##########