import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

# Read the dataset and store in a Pandas DataFrame
df = pd.read_csv("C:\\repos\\dataset\\connekt\\classificacao_1.csv")
x = df.iloc[:, 0:-1] # Input Variables (Regressors)
y = df.iloc[:, -1] # Target Variable
x_train = x
y_train = y
x_test = x.iloc[:]
y_test = y.iloc[:]
# Fit the Model
clf = LogisticRegression(random_state=42, solver='lbfgs', multi_class='auto').fit(x_train, y_train)
# K-fold Cross Validation
scores = cross_val_score(clf, x_train, y_train, cv=10)
print("CV Scores: ", scores)
# Mean from all CVs
print("CV Scores Mean: ", np.mean(np.array(scores)))
# Accuracy Mean
print("Accuracy Mean: ", clf.score(x_train, y_train))
# Probability estimates
print("Probabiliy (Train):")
print(clf.predict_proba(x_train))
# Independent term in the linear model
# print("Independent Term: ", reg.intercept_)
y_predicted = clf.predict(x_test)
print("Predicted (Test): ", y_predicted)
# Probability estimates
print("Probabiliy (Test): ", clf.predict_proba(x_test))