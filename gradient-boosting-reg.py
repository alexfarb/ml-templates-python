import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Read the dataset and store in a Pandas DataFrame
df = pd.read_csv("C:\\repos\\dataset\\connekt\\regressao_1.csv")
x = df.iloc[:, 0:-1] # Input Variables
y = df.iloc[:, -1] # Target Variable
x_train = x
y_train = y
x_test = x.iloc[:]
y_test = y.iloc[:]
# Fit the Model
#Create Gradient Boosting Classifier object
reg = GradientBoostingRegressor(n_estimators = 100, learning_rate= 1.0,
                                 max_depth = 1, random_state = 42).fit(x_train, y_train)
# K-fold Cross Validation
scores = cross_val_score(reg, x_train, y_train, cv = 10)
print("CV Scores: ", scores)
# Mean from all CVs
print("CV Scores Mean: ", np.mean(np.array(scores)))
# Accuracy Mean
# print("Accuracy Mean: ", clf.score(x_train, y_train))
# Probability estimates
# print("Probabiliy (Train):")
# print(clf.predict_proba(x_train))
y_predicted = reg.predict(x_test)
print("Predicted (Test): ", y_predicted)
# # Probability estimates
# print("Probabiliy (Test): ", clf.predict_proba(x_test))
# Scores
# print("Accuracy: ", accuracy_score(y_test, y_predicted))
# print("Precision: ", precision_score(y_test, y_predicted))
# print("Recall: ", recall_score(y_test, y_predicted))
# print("F1-Score: ", f1_score(y_test, y_predicted))
# print("ROC (AUC): ", roc_auc_score(y_test, y_predicted))