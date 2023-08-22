#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 10:45:40 2023
@author: gmalov
"""

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, cross_val_score
from bayes_opt import BayesianOptimization, UtilityFunction
from lightgbm import LGBMClassifier
import numpy as np

# Load the data
data = pd.read_csv('/Users/gmalov/Downloads/icr-identify-age-related-conditions/train.csv', index_col=0)
greeks = pd.read_csv('/Users/gmalov/Downloads/icr-identify-age-related-conditions/greeks.csv', index_col=0)

# Cleaning column names
data.columns = data.columns.str.strip()

# Defining the target variable and the features matrix
y = data['Class']
X = data.drop('Class', axis=1)

# Encoding the 'EJ' column
le = LabelEncoder()
X['EJ'] = le.fit_transform(X['EJ'])

# Custom function to calculate balanced log loss
def balanced_log_loss(y_true, y_pred):
    # Calculate the number of occurrences of each class
    n0 = len(y_true[y_true == 0])
    n1 = len(y_true[y_true == 1])

    # Calculate the log loss for each class
    log_loss_0 = np.sum([np.log(1-p + 0.000000000000001) for y, p in zip(y_true, y_pred) if y == 0]) / n0 if n0 > 0 else 0
    log_loss_1 = np.sum([np.log(p + 0.000000000000001) for y, p in zip(y_true, y_pred) if y == 1]) / n1 if n1 > 0 else 0
    
    return (log_loss_0 + log_loss_1) / 2

# Define the hyperparameter space for LightGBM
param_space = {
    # ... [Your parameter space remains unchanged]
}

# Define the optimization function for Bayesian Optimization
def lgb_optimization(**args):
    # Convert some arguments to integer as required by LGBM
    args['num_leaves'] = int(args['num_leaves'])
    args['min_child_samples'] = int(args['min_child_samples'])
    # ... [Other arguments remain unchanged]

    # Initialize LGBM with the given arguments
    lgb = LGBMClassifier(**args, objective="binary", verbose=-1)
    skf = StratifiedKFold(n_splits=10)

    # Calculate the cross-validation score
    cv_score = cross_val_score(lgb, X, y, cv=skf, scoring=make_scorer(balanced_log_loss, needs_proba=True), n_jobs=-1).mean()
    
    return cv_score

# Create a UtilityFunction for acquisition
utility = UtilityFunction(kind='ei', kappa=2, xi=0.03)

# Initializing Bayesian Optimization
optimizer = BayesianOptimization(f=lgb_optimization, pbounds=param_space, random_state=42, verbose=2)
optimizer.set_gp_params(normalize_y=True)

# Maximizing the optimization function
optimizer.maximize(init_points=2000, n_iter=5000, acquisition_function=utility)

# Print the best parameters
print(optimizer.max['params'])
