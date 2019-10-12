# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

class DataPreprocessor:
    def __init__(self,y,tX,cols,dtypes):
        assert tX.shape[1] == len(cols), "The number of columns of data does not match the number of column names"
        assert len(cols) == len(dtypes), "The number f column names must match the number of dtypes"
        self.CATEGORICAL_TYPE = 'categorical'
        self.FLOAT_TYPE = 'float'
        self.FLOAT_ANGLE_TYPE = 'float-angle'
        
        self.cols = cols
        self.dtypes = dtypes
        features = list(tX.T)
        dtypes_features = [list(x) for x in zip(dtypes, features)]
        self.data_dict = dict(zip(cols,dtypes_features))
        self.y = y
        
    def dropFeatures(self,cols):
        for c in cols:
            self.data_dict.pop(c) 
        self.cols = list(self.data_dict.keys())
        self.dtypes = [self.data_dict.get(c)[0] for c in self.cols]
        
    def getFeatures(self, cols = None):
        if not cols:
            cols = self.cols
        features_list = self.getFeaturesList(cols)
        return np.hstack(features_list)
    
    def relabelYNonNegative(self):
        self.y[self.y == -1] = 0        
    
    def relabelYNegative(self):
        self.y[self.y == 0] = -1
    
    def getYLabels(self):
        return self.y
    
    def getFeaturesList(self,cols):
        return [self.data_dict.get(c)[1].reshape(-1,1) for c in cols]
    
    def replaceUndefVal(self,undef_orig = -999.0,undef_new = 0, cols = None):
        if not cols:
            cols = self.cols
        features = self.getFeatures(cols)    
        features = np.where(features == undef_orig, np.nan, features)
        self.updateData(cols,features)
    
    def replaceNanVal(self,val = 0, cols = None):
        if not cols:
            cols = self.cols
        features = self.getFeatures(cols)
        features = np.where(np.isnan(features), val, features)
        self.updateData(cols,features)
        
    def nanStandardize(self, cols = None):
        if not cols:
            cols = [c for c in self.cols \
                    if (self.data_dict.get(c)[0] == self.FLOAT_TYPE\
                        or self.data_dict.get(c)[0] == self.FLOAT_ANGLE_TYPE)]
            
        # Check if columns are float or float angle
        for col in cols:
            if self.data_dict.get(col)[0] != self.FLOAT_TYPE \
            and self.data_dict.get(col)[0] != self.FLOAT_ANGLE_TYPE:
                raise ValueError('Cannot standardize categorical columns')
        
        features = self.getFeatures(cols)
        mean = np.nanmean(features, axis = 0)
        std = np.nanstd(features, axis = 0)
        features = (features - mean)/std
        
        self.updateData(cols,features)
   
    def updateData(self,cols,features,dtypes_list = None):
        
        features_list = list(features.T)
        if not dtypes_list:
            dtypes_list = [self.data_dict.get(c)[0] for c in cols]
        
        dtypes_features = [list(x) for x in zip(dtypes_list, features_list)]
        update_dict = dict(zip(cols,dtypes_features))
        self.data_dict.update(update_dict)
        
    def oneHotEncode(self, cols = None):
        if not cols:
            cols = [c for c in self.cols if self.data_dict.get(c)[0] == self.CATEGORICAL_TYPE]
        
        # Check if columns are categorical
        for col in cols:
            if self.data_dict.get(col)[0] != self.CATEGORICAL_TYPE:
                raise ValueError('Cannot one hot encode non-categorical columns')
        features = self.getFeatures(cols)
        n = features.shape[0]
        idx_n = np.asarray(range(n),dtype = np.int64)
        features_list = list(features.T.astype(np.int64))
        ohe_features_list = [np.zeros((n,len(set(x))-1)) for x in features_list]
        for (idx,x) in enumerate(ohe_features_list): 
            no_dummy_var_idx = (features_list[idx] < (x.shape[1] - 1)).astype(np.int64)
            ndv_idx_n = idx_n[no_dummy_var_idx]
            ndv_one_hot_idx = features_list[idx][no_dummy_var_idx]
            x[ndv_idx_n,ndv_one_hot_idx] = 1 
        features = np.hstack(ohe_features_list)
        self.dropFeatures(cols)
        
        cols_sizes_list = [(c,len(set(x))) for (c,x) in zip(cols,features_list)]
        ohe_cols = [[c+"_"+str(x) for x in range(s-1)] for (c,s) in cols_sizes_list]
        ohe_cols = [ohe_col for ohe_col_sublist in ohe_cols for ohe_col in ohe_col_sublist]
        dtypes_list = len(ohe_cols) * [self.CATEGORICAL_TYPE]
        
        self.cols = self.cols + ohe_cols
        self.dtypes = self.dtypes + dtypes_list
        self.updateData(ohe_cols,features,dtypes_list)
    
    def nanStatistics():
        
    
    def outlierRemoval():
        # TODO
        return

    
    def multiHistPlots(tX,cols,size):
    n = tX.shape[1]
    n_rows = np.ceil(np.sqrt(n)).astype(np.int64)
    n_cols = np.floor(np.sqrt(n)).astype(np.int64)
    
    fig, axes = plt.subplots(nrows = n_rows, ncols = n_cols, figsize = size)
    
    c = 0
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row][col]
            if c < tX.shape[1]:
                ax.hist(tX[:,c], label = '{}'.format(cols[c]),density = True)
                ax.legend(loc = 'upper left')
                ax.set_ylabel('Probability')
                ax.set_xlabel('Value')
            c += 1
    plt.show()    