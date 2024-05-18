#This will contain my code for Multiple Linear Regression Lab from course material

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#pylab contains functions to create plots
import pylab as plab
#scikit-learn contains many useful functions for model evaluation and metrics
from sklearn import metrics
#scikit-learn contains a bunch of useful machine learning libraries. Linear model contains LinearRegression and multiple linear regression
from sklearn import linear_model


#create dataframe of our data
#init_df = pd.read_csv("Machine Learning with Python\Support Files\FuelConsumption.csv")
init_df = pd.read_csv("C:\Programming Projects\Machine Learning\IBM Ai Engineering\Machine Learning with Python\Support Files\FuelConsumption.csv")
#Preview data
print(init_df.head())

#Practise plotting - plot the CO2 emisisons against the engine size
plt.scatter(init_df.ENGINESIZE, init_df.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# ---- TRAIN TEST SPLIT ----
# get random number of rows, up to 80% of data which we can use for training and the other 20% for testing
msk = np.random.rand(len(init_df)) < 0.8

# create training and test sets
train = init_df[msk]
test = init_df[~msk]

#Visualise our training dataset and confirm linear relationship of independent variable with dependent variable
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='orange')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# ---- MULTIPLE LINEAR REGRESSION ----
#create linear regression object. Multiple linear regression is an extension of simple linear regression so there is a lot of cross-over of program logic
regr = linear_model.LinearRegression()

#Get x values with multiple coloumns and y values from our training set and convert them to a numpy array
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

#Train the model
regr.fit (train_x, train_y)

#print the cooefficients
print('Coefficients: ', regr.coef_)

"""
__Coefficient__ and __Intercept__  are the parameters of the fitted line. 

Given that it is a multiple linear regression model with 3 parameters and that the parameters are the intercept and coefficients of the hyperplane, sklearn can estimate them from our data. Scikit-learn uses plain Ordinary Least Squares method to solve this problem.

Ordinary Least Squares (OLS)
OLS is a method for estimating the unknown parameters in a linear regression model. 
OLS chooses the parameters of a linear function of a set of explanatory variables by minimizing the sum of the squares of the differences between the target dependent variable and those predicted by the linear function. In other words, it tries to minimizes the sum of squared errors (SSE) or mean squared error (MSE) between the target variable (y) and our predicted output ($\hat{y}$) over all samples in the dataset.

OLS can find the best parameters using of the following methods:
* Solving the model parameters analytically using closed-form equations
* Using an optimization algorithm (Gradient Descent, Stochastic Gradient Descent, Newton’s Method, etc.)
"""

# ----- PREDICTION -----
#Predict y values (AKA Y-hat) from our test set
test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

predicted_y = regr.predict(test_x)

#Create visualisation to compare the predicted values agains the actual values
plt.scatter(test_y, predicted_y)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()
# ---------- Interesting find looking at the above scatter plot ---------------
    # - The predicted values are pretty close to the actual values when the CO2 emissions is of lower values/amounts
    # - As the amount of CO2 emissions increases, so does the variance in the predicted values


#Mean Squared error
print("Mean Squared error: %.2f" % np.mean((predicted_y - test_y) ** 2))

#Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(test_x, test_y))