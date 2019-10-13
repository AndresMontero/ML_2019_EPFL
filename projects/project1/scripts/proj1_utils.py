import numpy as np
import matplotlib.pyplot as plt 

def split_numerical_categorical(x,cat_cols):
    x_num = np.delete(x,cat_cols,axis = 1)
    x_cat = x[:,cat_cols]
    return x_num, x_cat

def replace_undef_val_with_nan(x):
    return np.where(x == -999.0, np.nan, x)

def nan_standardize_fit(x):                                
    mean = np.nanmean(x, axis = 0)
    std = np.nanstd(x, axis = 0)
    return (x - mean)/std , mean, std

def nan_standardize_with_median_fit(x):                               
    median = np.nanmedian(x, axis = 0)
    iqr,_,_ = calculate_iqr(x)  
    return 2*(x - median)/iqr , median, iqr

def nan_standardize_with_median_transform(x,median,iqr):                             
    return 2*(x - median)/iqr 
    
def nan_standardize_transform(x,mean,std):
    return (x - mean)/std

def relabel_y_non_negative(y):
    new_y = y.copy()
    new_y[new_y == -1] = 0
    return new_y
 
def relabel_y_negative(y):
    new_y = y.copy()
    new_y[new_y == 0] = -1
    return new_y
        
def replace_nan_val_with_mean(x):
    means = np.nanmean(x,axis = 0)
    n_cols = x.shape[1]
    new_x = x.copy()
    for i in range(n_cols):
        new_x[:,i] = np.where(np.isnan(new_x[:,i]), means[i], new_x[:,i])
    return new_x

def replace_nan_val_with_zero(x):
    n_cols = x.shape[1]
    new_x = x.copy()
    for i in range(n_cols):
        new_x[:,i] = np.where(np.isnan(new_x[:,i]), 0, new_x[:,i])
    return new_x

def calculate_iqr(x):
    q1 = np.quantile(x,0.25,axis = 0)
    q3 = np.quantile(x,0.75,axis = 0)
    return q3 - q1, q1, q3

def replace_iqr_outliers(x):
    iqr, q1, q3= calculate_iqr(x)
    upper_bound = q3 + iqr * 1.5
    lower_bound = q1 - iqr * 1.5
    x_trunc_up = np.where(x > upper_bound,upper_bound,x)
    x_trunc_low = np.where(x_trunc_up < lower_bound,lower_bound,x_trunc_up)
    return x_trunc_low

def replace_nan_val_with_median(x):
    medians = np.nanmedian(x,axis = 0)
    n_cols = x.shape[1]
    new_x = x.copy()
    for i in range(n_cols):
        new_x[:,i] = np.where(np.isnan(new_x[:,i]), medians[i], new_x[:,i])
    return new_x

def one_hot_encode(x):
    unique_vals = set(x.ravel())
    # print(unique_vals)
    n_cols = len(unique_vals) - 1 
    ohe_x = np.zeros((x.shape[0],n_cols))
    for (row,col) in enumerate(x):
        if col < n_cols:
            ohe_x[int(row),int(col)] = 1
    return ohe_x

def add_bias(x):
    return np.hstack((np.ones(x.shape[0]).reshape(-1,1),x))
   
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    n_train = round(y.shape[0]*ratio)
    idx = np.random.permutation(range(y.shape[0]))
    x_shuffled = x[idx]
    y_shuffled = y[idx]
    return x_shuffled[:n_train],y_shuffled[:n_train],x_shuffled[n_train:],y_shuffled[n_train:]
    
    # def getFeatures(self, cols = None):
    #     if not cols:
    #         cols = self.cols
    #     features_list = self.getFeaturesList(cols)
    #     return np.hstack(features_list)
    
    
    
    # def getFeaturesList(self,cols):
    #     return [self.data_dict.get(c)[1].reshape(-1,1) for c in cols]
    
    # def replaceUndefVal(self,undef_orig = -999.0,undef_new = 0, cols = None):
    #     if not cols:
    #         cols = self.cols
    #     features = self.getFeatures(cols)    
    #     features = np.where(features == undef_orig, np.nan, features)
    #     self.updateData(cols,features)
    
    # def replaceNanVal(self,val = 0, cols = None):
    #     if not cols:
    #         cols = self.cols
    #     features = self.getFeatures(cols)
    #     features = np.where(np.isnan(features), val, features)
    #     self.updateData(cols,features)
        
    # def removeNan(self,cols = None):
    #     if not cols:
    #         cols = self.cols
    #     features_eval_nan = self.getFeatures(cols)
    #     features = self.getFeatures(self.cols)
    #     print(self.y.shape)
    #     print(features.shape)
    #     dataset = np.hstack((self.y,features))
    #     dataset = dataset[~np.isnan(features_eval_nan).any(axis = 1),:]
    #     features = dataset[:,1:]
    #     print(features.shape)
    #     y = dataset[:,1].reshape(-1,1)
    #     self.updateData(cols,features)
    #     self.y = y
        
    # def nanStandardizeFit(self, cols = None):
    #     if not cols:
    #         cols = [c for c in self.cols \
    #                 if (self.data_dict.get(c)[0] == self.FLOAT_TYPE\
    #                     or self.data_dict.get(c)[0] == self.FLOAT_ANGLE_TYPE)]
            
    #     # Check if columns are float or float angle
    #     for col in cols:
    #         if self.data_dict.get(col)[0] != self.FLOAT_TYPE \
    #         and self.data_dict.get(col)[0] != self.FLOAT_ANGLE_TYPE:
    #             raise ValueError('Cannot standardize categorical columns')
        
    #     features = self.getFeatures(cols)
    #     mean = np.nanmean(features, axis = 0)
    #     std = np.nanstd(features, axis = 0)
    #     features = (features - mean)/std
        
    #     self.updateData(cols,features)
        
    #     return cols,mean,std
    
    # def nanStandardizeTransform(self,cols,mean,std):
        
    #     # Check if columns are float or float angle
    #     for col in cols:
    #         if self.data_dict.get(col)[0] != self.FLOAT_TYPE \
    #         and self.data_dict.get(col)[0] != self.FLOAT_ANGLE_TYPE:
    #             raise ValueError('Cannot standardize categorical columns')
    #     features = self.getFeatures(cols)        
    #     features = (features - mean)/std
    #     self.updateData(cols,features)
   
    # def updateData(self,cols,features,dtypes_list = None):
        
    #     features_list = list(features.T)
    #     if not dtypes_list:
    #         dtypes_list = [self.data_dict.get(c)[0] for c in cols]
        
    #     dtypes_features = [list(x) for x in zip(dtypes_list, features_list)]
    #     update_dict = dict(zip(cols,dtypes_features))
    #     self.data_dict.update(update_dict)
        
    # def oneHotEncode(self, cols = None):
    #     if not cols:
    #         cols = [c for c in self.cols if self.data_dict.get(c)[0] == self.CATEGORICAL_TYPE]
        
    #     # Check if columns are categorical
    #     for col in cols:
    #         if self.data_dict.get(col)[0] != self.CATEGORICAL_TYPE:
    #             raise ValueError('Cannot one hot encode non-categorical columns')
    #     features = self.getFeatures(cols)
    #     n = features.shape[0]
    #     idx_n = np.asarray(range(n),dtype = np.int64)
    #     features_list = list(features.T.astype(np.int64))
    #     ohe_features_list = [np.zeros((n,len(set(x))-1)) for x in features_list]
    #     print(set(features_list[0]))
    #     for (idx,x) in enumerate(ohe_features_list): 
    #         no_dummy_var_idx = (features_list[idx] < (x.shape[1] - 1)).astype(np.int64)
    #         ndv_idx_n = idx_n[no_dummy_var_idx]
    #         ndv_one_hot_idx = features_list[idx][no_dummy_var_idx]
    #         print(x.shape)
    #         x[ndv_idx_n,ndv_one_hot_idx] = 1 
    #     features = np.hstack(ohe_features_list)
    #     self.dropFeatures(cols)
        
    #     cols_sizes_list = [(c,len(set(x))) for (c,x) in zip(cols,features_list)]
    #     ohe_cols = [[c+"_"+str(x) for x in range(s-1)] for (c,s) in cols_sizes_list]
    #     ohe_cols = [ohe_col for ohe_col_sublist in ohe_cols for ohe_col in ohe_col_sublist]
    #     dtypes_list = len(ohe_cols) * [self.CATEGORICAL_TYPE]
        
    #     self.cols = self.cols + ohe_cols
    #     self.dtypes = self.dtypes + dtypes_list
    #     self.updateData(ohe_cols,features,dtypes_list)
    
    # def nanStatistics():
    #     # TODO
    #     return
    
    # def outlierRemoval():
    #     # TODO
    #     return

    
def multiHistPlots(x,figsize = (15,15)):
    n = x.shape[1]
    n_rows = np.ceil(np.sqrt(n)).astype(np.int64)
    n_cols = np.floor(np.sqrt(n)).astype(np.int64)

    if n_rows * n_cols < n:
        n_cols = np.ceil(np.sqrt(n)).astype(np.int64)

    fig, axes = plt.subplots(nrows = n_rows, ncols = n_cols, figsize = figsize)

    c = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if n > 1:
                ax = axes[row][col]
            else:
                ax = axes
            if c < x.shape[1]:
                ax.hist(x[:,c], label = 'feature_{:d}'.format(c),density = True)
                ax.legend(loc = 'upper left')
                ax.set_ylabel('Probability')
                ax.set_xlabel('Value')
            c += 1
    plt.show()    