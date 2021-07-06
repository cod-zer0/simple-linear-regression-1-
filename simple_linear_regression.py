# Simple linear regression

## Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting simpler linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test.reshape(-1, 1))

# Visualizing the training set  results
plt.scatter(X_train, y_train , color = 'red')
plt.plot(X_train, regressor.predict(X_train.reshape(-1, 1)))