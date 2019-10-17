import numpy as np

def split_numerical_categorical(x,cat_cols):
    """Split a feature matrix in numerical and categorical 
       feature matrices.
       Args:
            x         (numpy.ndarray): the feature matrix
            cat_cols  (numpy.ndarray): the categorical features
        Returns:
            numpy.ndarray: the numerical feature matrix
            numpy.ndarray: the categorical feature matrix
    """

    x_num = np.delete(x,cat_cols,axis = 1)
    x_cat = x[:,cat_cols]
    return x_num, x_cat

def replace_undef_val_with_nan(x):
    """Replace the undefined values with nan.
       Args:
            x (numpy.ndarray): the feature matrix
       Returns:
            numpy.ndarray: the feature matrix with nan instead 
                           of undefined values
    """

    return np.where(x == -999.0, np.nan, x)

def nan_standardize_fit(x): 
    """Standardize and get the means and standard deviations.
       Args:
            x (numpy.ndarray): the feature matrix
       Returns:
            numpy.ndarray: the standardized feature matrix
            numpy.ndarray: the means of the features
            numpy.ndarray: the standard deviations of the features
    """      

    mean = np.nanmean(x, axis = 0)
    std = np.nanstd(x, axis = 0)
    return (x - mean)/std , mean, std

def nan_standardize_transform(x,mean,std):
    """Standardize with given means and standard deviations.
        Args:
            x (numpy.ndarray): the feature matrix
        Returns:
            numpy.ndarray: the standardized feature matrix
    """

    return (x - mean)/std

def relabel_y_non_negative(y):
    """Relabel -1 ground truth labels to 0.
        Args:
            y (numpy.ndarray): the ground truth labels
        Returns:
            numpy.ndarray: the relabeled ground truth labels
    """

    new_y = y.copy()
    new_y[new_y == -1] = 0
    return new_y
 
def relabel_y_negative(y):
    """Relabel 0 ground truth labels to -1.
       Args:
            y (numpy.ndarray): the ground truth labels
       Returns:
            numpy.ndarray: the relabeled ground truth labels
    """

    new_y = y.copy()
    new_y[new_y == 0] = -1
    return new_y
        
def replace_nan_val_with_mean(x):
    """Replace nan values with the mean of the column.
       Args:
            x (numpy.ndarray): the feature matrix
       Returns:
            numpy.ndarray: the relabeled ground truth labels
    """

    means = np.nanmean(x,axis = 0)
    n_cols = x.shape[1]
    new_x = x.copy()
    for i in range(n_cols):
        new_x[:,i] = np.where(np.isnan(new_x[:,i]), means[i], new_x[:,i])
    return new_x

def replace_nan_val_with_zero(x):
    """Replace nan values with 0.
        Args:
            x (numpy.ndarray): the feature matrix
        Returns:
            numpy.ndarray: the feature matrix with nan values replaced by 0
    """

    n_cols = x.shape[1]
    new_x = x.copy()
    for i in range(n_cols):
        new_x[:,i] = np.where(np.isnan(new_x[:,i]), 0, new_x[:,i])
    return new_x

def calculate_iqr(x):
    """Calculate the Interquartile range.
        Args:
            x (numpy.ndarray): the feature matrix
        Returns:
            numpy.ndarray: the interquartile range
            numpy.ndarray: the first quartile
            numpy.ndarray: the third quartile
    """

    q1 = np.quantile(x,0.25,axis = 0)
    q3 = np.quantile(x,0.75,axis = 0)
    return q3 - q1, q1, q3

def replace_iqr_outliers(x):
    """Replace points outside the bounds given by 
       first, third quartiles and the interquartile range.
        Args: 
            x (numpy.ndarray): the feature matrix
        Returns:
            numpy.ndarray: the outliers are replaced by the first or third quartile (the closest)
    """

    iqr, q1, q3= calculate_iqr(x)
    upper_bound = q3 + iqr * 1.5
    lower_bound = q1 - iqr * 1.5
    x_trunc_up = np.where(x > upper_bound,upper_bound,x)
    x_trunc_low = np.where(x_trunc_up < lower_bound,lower_bound,x_trunc_up)
    return x_trunc_low

def replace_nan_val_with_median(x):
    """Replace nan values with median.
        Args:
            x (numpy.ndarray): the feature matrix
        Returns:
            numpy.ndarray: the feature matrix with nan values replaced by medians
    """

    medians = np.nanmedian(x,axis = 0)
    n_cols = x.shape[1]
    new_x = x.copy()
    for i in range(n_cols):
        new_x[:,i] = np.where(np.isnan(new_x[:,i]), medians[i], new_x[:,i])
    return new_x

def one_hot_encode(x):
    """One hot encode features.
        Args:
            x (numpy.ndarray): a vector Nx1 containing categorical values
        Returns:
            numpy.ndarray: the one hot encoding of the vector
    """

    unique_vals = set(x.ravel())
    n_cols = len(unique_vals) - 1 
    ohe_x = np.zeros((x.shape[0],n_cols))
    for (row,col) in enumerate(x):
        if col < n_cols:
            ohe_x[int(row),int(col)] = 1
    return ohe_x

def add_bias(x):
    """Add bias column to feature matrix.
        Args:
            x (numpy.ndarray): the feature matrix
        Returns:
            numpy.ndarray: a feature matrix with the bias terms (1s) in the first column
    """

    return np.hstack((np.ones(x.shape[0]).reshape(-1,1),x))

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree.
    
        Args:
            x      (numpy.ndarray): the feature matrix
            degree (float)        : the degree of the polynomial features
        Returns:
            numpy.ndarray: the matrix with polynomial features
    """

    if len(x.shape) == 1:
        x_res = x.reshape(-1,1)
    else:
        x_res = x
    bias = np.ones(x_res.shape[0]).reshape(-1,1)
    ext_x = np.repeat(x_res,degree,axis = 1)
    powers = np.repeat(np.asarray(range(1,degree+1)).reshape(1,-1),x_res.shape[1],axis = 0).ravel()
    return np.hstack((bias,np.power(ext_x,powers))) 
     
def split_data(x, y, ratio, seed=1):
    """Split the dataset based on the split ratio. If ratio is 0.8 
        you will have 80% of your data set dedicated to training 
        and the rest dedicated to testing.
        Args:
            x:      (numpy.ndarray): the feature matrix
            y:      (numpy.ndarray): the ground turth labels
        Returns:
            numpy.ndarray: the train features
            numpy.ndarray: the train ground truth labels
            numpy.ndarray: the test features
            numpy.ndarray: the test ground truth labels
    """

    np.random.seed(seed)
    n_train = round(y.shape[0]*ratio)
    idx = np.random.permutation(range(y.shape[0]))
    x_shuffled = x[idx]
    y_shuffled = y[idx]
    return x_shuffled[:n_train],y_shuffled[:n_train],x_shuffled[n_train:],y_shuffled[n_train:]
    
def get_label_y_counts(y):
    """Get the count of each ground truth label. 
        Args:
            y (numpy.ndarray): the ground truth labels
        Returns:
            numpy.ndarray
    """
    
    return np.unique(y,return_counts=True)

def get_accurarcy_score(y_pred,y_val):
    """Get the accuracy score.
        Args:
            y_pred (numpy.array): the predicted labels
            y_val (numpy.array): the ground truth labels
        Returns:
            numpy.in64: the accuracy score. Range 0 to 1.

    """
    return np.sum(y_pred == y_val)/len(y_pred)