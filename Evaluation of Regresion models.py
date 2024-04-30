# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 07:39:09 2024

@author: shubh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###########################       Step 1       ################################
########################### Importing Data Set ################################

# Importing the dataset 
dataset = pd.read_csv('advertising.csv')
#x = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, -2].values

# Printing the data set
print(dataset.head())

# Printing all the column names 
column_headers = list(dataset.columns.values)
print("The Column Header :", column_headers)

###########################       Step 2       ################################
###########################  Cleaning Data Set ################################


# Checking if data set has null values 
print(dataset.isnull().sum())

# Collecting all the null values present
all_rows = dataset[dataset.isnull().any(axis=1)] 

#deleting duplicate values
# Duplicate values 
dupilcate = dataset[dataset.duplicated()]

#deleting duplicate values
dataset.drop_duplicates(inplace=True)


###########################       Step 3       ################################
###########################  Feature Selction  ################################

# Feature selection 
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


## as the number of featurres are less so we dont need feature selection

###########################       Step 4            ###########################
###########################  Categorical variables  ###########################




###########################       Step 5            ###########################
###########################    Data splitting       ###########################

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


###########################       Step 6            ###########################
###########################  Model Implimentation   ###########################

# XGBoost classfifer
from xgboost import XGBRegressor
RegressorXGB = XGBRegressor()
RegressorXGB.fit(x_train,y_train)
y_predictXGB = RegressorXGB.predict(x_test)



# linear regresion 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred_multi = regressor.predict(x_test)

# Polynomial Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y_train)
y_pred_poly = lin_reg_2.predict(poly_reg.transform(x_test))


# SVR
y_train_svr = y_train.reshape(len(y_train),1)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train_Tsvr = sc_x.fit_transform(x_train) 
y_train_Tsvr = sc_y.fit_transform(y_train_svr)


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train_Tsvr,y_train_Tsvr)
y_pred_svr = sc_y.inverse_transform(regressor.predict(sc_x.transform(x_test)))



# Decision tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x_train,y_train)
y_pred_tree = regressor.predict(x_test)


# Rrandom Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x_train,y_train)
y_pred_randtree = regressor.predict(x_test)


############# evaluation of model with R2 model##############
from sklearn.metrics import r2_score
print('R2 score by XGBregressor :',r2_score(y_test,y_predictXGB))
print('R2 score by linear_regresion :',r2_score(y_test,y_pred_multi))
print('R2 score by Polynomial regresion:',r2_score(y_test,y_pred_poly))
print('R2 score by SVR:',r2_score(y_test,y_pred_svr))
print('R2 score by Decision tree:',r2_score(y_test,y_pred_tree))
print('R2 score by Randomforest:',r2_score(y_test,y_pred_randtree))

# K- validation
from sklearn.model_selection import cross_val_score
accuracy_k = cross_val_score(estimator= RegressorXGB,X = x_train, y = y_train, cv = 10, n_jobs = -1) # use n_job =-1 to use cpus
print(' Average acuracy by K fold cross validation: ', accuracy_k.mean())
print('Std acuracy by K fold cross validation:',accuracy_k.std())
