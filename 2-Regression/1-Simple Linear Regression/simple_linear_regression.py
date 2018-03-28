#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 20:08:21 2018

@author: kifli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
# Data Pre-processing
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Split the data into training model and test model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_predict = regressor.predict(X_test)

plt.scatter(X_train, y_train,color='red')
plt.plot(X_train, regressor.predict(X_train),'b')
plt.title('Salary vs Experiences (Training Set)')
plt.xlabel('Years of Experiences')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_test, y_test,color='red')
plt.plot(X_train, regressor.predict(X_train),'b')
plt.title('Salary vs Experiences (Test Set)')
plt.xlabel('Years of Experiences')
plt.ylabel('Salary')
plt.show()

print('Coefficients: \n',regressor.coef_)
print('Intercept: \n',regressor.intercept_)