import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

# Read the dataset and store in a Pandas DataFrame
df = pd.read_csv("C:\\repos\\dataset\\connekt\\regressao_2.csv")
x = df.iloc[:, 0:-1] # Input Variables (Regressors)
y = df.iloc[:, -1] # Target Variable
x_train = x
y_train = y
x_test = x.iloc[:]
y_test = y.iloc[:]
# Fit the Model
reg = ElasticNet(random_state=42).fit(x_train, y_train)
# K-fold Cross Validation
scores = cross_val_score(reg, x_train, y_train, cv=10)
print("CV Scores: ", scores)
# Mean from all CVs
print("CV Scores Mean: ", np.mean(np.array(scores)))
# Calculate R^2 of prediction
print("Prediction R^2: ", reg.score(x_train, y_train))
# Estimated coefficients
# print(reg.coef_)
# Independent term in the linear model
# print("Independent Term: ", reg.intercept_)
y_predicted = reg.predict(x_test)
print("Predicted (Test): ", y_predicted)
print("MSE (Test): ", mean_squared_error(y_test, y_predicted))