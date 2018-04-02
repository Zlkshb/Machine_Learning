#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 11:39:45 2018

@author: kifli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LinearRegression


def backwardElimination(X,SL):
    pvalue = 1
    x_column = np.arange(len(X[0]))
    
    while(pvalue > SL):
        X_opt = X[:,x_column]
        regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
        pvalues = regressor_OLS.pvalues        
        indice = np.where(pvalues == pvalues.max())
        pvalue = pvalues[indice]
        if (pvalue > SL):
            x_column = np.delete(x_column,indice)
            
    return x_column
    #print(regressor_OLS.summary())

def backwardELiminationwithAdjR(X,SL):
    var = True
    x_column = np.arange(len(X[0]))
    
    while var:
        X_opt = X[:,x_column]
        regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
        pvalues = regressor_OLS.pvalues        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        indice = np.where(pvalues == pvalues.max())
        pvalue = pvalues[indice]
        if (pvalue > SL):
            x_column = np.delete(x_column,indice,1)
            X_opt = X[:,x_column]
            temp_regressor = sm.OLS(y,X_opt).fit()
            adjR_after = temp_regressor.rsquared_adj.astype(float)
            if adjR_before >= adjR_after:
                x_rollback = np.hstack((x, temp[:,[0,j]]))
                x_rollback = np.delete(x_rollback, j, 1)
                        
        else:
            var = False

def backwardEliminationWithR2(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x

dataset = pd.read_csv('50_Startups.csv')

# Setting independent and dependent variables into X and y
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Categorical variables
labelEncoder_x = LabelEncoder()
X[:,3] = labelEncoder_x.fit_transform(X[:,3])

onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:,1:]

# Splitting the data set into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(np.ones((len(X[:,0]),1)).astype(int),X, 1)
SL = 0.05
backwardElimination(X,SL)

# Select Significance Level (SL) : 0.05
#regressor_OLS.summary()
