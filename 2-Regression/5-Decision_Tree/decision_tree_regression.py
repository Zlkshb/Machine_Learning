# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Decision Tree Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression_template import Regression

from sklearn.tree import DecisionTreeRegressor
# Importing dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, -1].values


# Fitting the Decision Tree Regression to the dataset

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)


# Predicting new result
y_pred = regressor.predict(6.5)

# To visualize the result given by the Decision Tree Regression process
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))

title = 'Truth or Bluff (Decision Tree Regression)'
xlabel = 'Position level'
ylabel = 'Salary'
regression = Regression(X,y,title,xlabel,ylabel)
regression.plot_regression(regressor,X_grid)