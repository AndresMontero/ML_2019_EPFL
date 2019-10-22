""" Course: Machine Learning
    Projeect 1: The Higgs bosson machine learning challenge
    Authors:
        - Maraz Erick
        - Montero Andres
        - Villarroel Adrian
    This script runs the best ML model and generates the 
    submission file.
"""
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from proj1_utils import *
from implementations_utils import *
from implementations import *
from proj1_visualization import *
from proj1_cross_validation import *

print("Importing data...")
DATA_TRAIN_PATH = '../data/train.csv' 
y_train_raw, tX_train_raw, ids_train = load_csv_data(DATA_TRAIN_PATH)
DATA_TEST_PATH = '../data/test.csv' 
_, tX_test_raw, ids_test = load_csv_data(DATA_TEST_PATH)

print("Preprocessing data...")
cat_cols = [22]             # PRI_jet_num column number 
# Tuned hyperparameters
degree = 3     
gamma = 2.2e-6
lambda_ = 0.001

""" Preprocess training data """
x_train_num, x_train_cat = split_numerical_categorical(tX_train_raw,cat_cols)
# Preprocess numerical data
x_train_num_nan = replace_undef_val_with_nan(x_train_num)
x_train_num_stdz, train_mean, train_std = nan_standardize_fit(x_train_num_nan)
x_train_num_valid = replace_nan_val_with_median(x_train_num_stdz)
x_train_num_iqr = replace_iqr_outliers(x_train_num_valid)
x_train_poly = build_poly(x_train_num_iqr,degree)
# Preprocess categorical data
x_train_ohe_cat = one_hot_encode(x_train_cat)
# Merge preprocessed numerical and categorical data
x_train = np.hstack((x_train_poly,x_train_ohe_cat))
# Preprocess labels
y_train = relabel_y_non_negative(y_train_raw).reshape(-1,1) # Binary logistic regression accepts labels 0 or 1

""" Split in training and validation sets"""
x_train,  y_train, x_val, y_val = split_data(x_train,y_train,0.8)

""" Preprocess test data """

x_test_num, x_test_cat = split_numerical_categorical(tX_test_raw,cat_cols)

# Preprocess numerical data
x_test_num_nan = replace_undef_val_with_nan(x_test_num)
x_test_num_stdz = nan_standardize_transform(x_test_num_nan,train_mean,train_std)
x_test_num_valid = replace_nan_val_with_median(x_test_num_stdz)
x_test_num_iqr = replace_iqr_outliers(x_test_num_valid)
x_test_poly = build_poly(x_test_num_iqr,degree)
# Preprocess categorical data
x_test_ohe_cat = one_hot_encode(x_test_cat)
# Merge preprocessed numerical and categorical data
x_test = np.hstack((x_test_poly,x_test_ohe_cat))

print("Training model...")
max_iters = 1000
w_initial = np.zeros((x_train.shape[1], 1))
weights,_ = reg_logistic_regression(y_train, x_train, w_initial, max_iters, gamma, lambda_)

print("Validating model...")
y_pred_val = predict_labels_logistic(weights,x_val)
y_val = relabel_y_negative(y_val)
acc_val = get_accuracy_score(y_pred_val,y_val)
print("Accuracy on validation set: "+str(acc_val))

print("Predicting test labels...")
y_pred = predict_labels_logistic(weights, x_test)

print("Generating file for submission...")
create_csv_submission(ids_test, y_pred, "Reg_log_reg_submission.csv")
print("File generated")


  