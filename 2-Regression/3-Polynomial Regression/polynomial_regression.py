#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 00:32:29 2018

@author: kifli
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Fitting Linear Regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression to the dataset
# Add new x2 parameter for polynomial data
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualizing the Linear Regression results
y_lin_pred = lin_reg.predict(X)
plt.scatter(X,y,color = 'red')
plt.plot(X,y_lin_pred,color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# adding additional values for X points
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))

# Visualizing the Polynomial Regression results
y_poly_pred = lin_reg_2.predict(poly_reg.fit_transform(X_grid))

plt.scatter(X,y,color = 'red')
plt.plot(X_grid,y_poly_pred,color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

lin_reg.predict(6.5)
lin_reg_2.predict(poly_reg.fit_transform(6.5))