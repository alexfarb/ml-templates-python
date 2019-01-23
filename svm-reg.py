import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# Read the dataset and store in a Pandas DataFrame
df = pd.read_csv("C:\\repos\\dataset\\connekt\\regressao_2.csv")
x = df.iloc[:, 0:-1] # Input Variables
y = df.iloc[:, -1] # Target Variable
x_train = x
y_train = y
x_test = x.iloc[:]
y_test = y.iloc[:]
# Fit the Model
reg = svm.SVR(gamma='scale').fit(x_train, y_train)
# K-fold Cross Validation
scores = cross_val_score(reg, x_train, y_train, cv=10)
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
# get support vectors
# print("Support Vector: ", clf.support_vectors_)
# get indices of support vectors
# print("Support Vector Indices: ", clf.support_)
# get number of support vectors for each class
# print("Support Vector for each class: ", reg.n_support_)
# Scores
# print("Accuracy: ", accuracy_score(y_test, y_predicted))
# print("Precision: ", precision_score(y_test, y_predicted))
# print("Recall: ", recall_score(y_test, y_predicted))
# print("F1-Score: ", f1_score(y_test, y_predicted))
# print("ROC (AUC): ", roc_auc_score(y_test, y_predicted))