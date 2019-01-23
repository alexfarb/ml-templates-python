import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# Read the dataset and store in a Pandas DataFrame
df = pd.read_csv("C:\\repos\\dataset\\connekt\\classificacao_2.csv")
x = df.iloc[:, 0:-1] # Input Variables
y = df.iloc[:, -1] # Target Variable
x_train = x
y_train = y
x_test = x.iloc[:]
y_test = y.iloc[:]
# Fit the Model
neigh = KNeighborsClassifier(n_neighbors = 3).fit(x_train, y_train)
# K-fold Cross Validation
scores = cross_val_score(neigh, x_train, y_train, cv=10)
print("CV Scores: ", scores)
# Mean from all CVs
print("CV Scores Mean: ", np.mean(np.array(scores)))
# Accuracy Mean
# print("Accuracy Mean: ", clf.score(x_train, y_train))
# Probability estimates
print("Probabiliy (Train):")
print(neigh.predict_proba(x_train))
y_predicted = neigh.predict(x_test)
print("Predicted (Test): ", y_predicted)
# Probability estimates
print("Probabiliy (Test): ", neigh.predict_proba(x_test))
print("Accuracy: ", accuracy_score(y_test, y_predicted))
print("Precision: ", precision_score(y_test, y_predicted))
print("Recall: ", recall_score(y_test, y_predicted))
print("F1-Score: ", f1_score(y_test, y_predicted))
print("ROC (AUC): ", roc_auc_score(y_test, y_predicted))