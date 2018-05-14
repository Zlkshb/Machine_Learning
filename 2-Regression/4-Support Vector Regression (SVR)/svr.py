#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 14:49:34 2018

@author: kifli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR

def regression_func(regressor,X,y,X_test):
    y_pred = regressor.predict(X_test)

def show_plot(title,xlabel,ylabel):   
    plt.scatter(X,y,color='red')
    plt.plot(X,regressor.predict(X),color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

# Create the SVR results
regressor = SVR()