import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
import numpy as np
import graphviz

# Read the dataset and store in a Pandas DataFrame
df = pd.read_csv(datapath)
x = df.iloc[:, 0:-1] # Input Variables (Regressors)
y = df.iloc[:, -1] # Target Variable
x_train = x
y_train = y
x_test = x.iloc[:]
y_test = y.iloc[:]
# Fit the Model
reg = tree.DecisionTreeRegressor().fit(x_train, y_train)
# K-fold Cross Validation
scores = cross_val_score(reg, x_train, y_train, cv=10)
print("CV Scores: ", scores)
# Mean from all CVs
print("CV Scores Mean: ", np.mean(np.array(scores)))
# Accuracy Mean
# print("Accuracy Mean: ", clf.score(x_train, y_train))
# Probability estimates
# print("Probabiliy (Train):")
# print(reg.predict_proba(x_train))
# Independent term in the linear model
# print("Independent Term: ", reg.intercept_)
y_predicted = reg.predict(x_test)
print("Predicted (Test): ", y_predicted)
# Probability estimates
# print("Probabiliy (Test): ", reg.predict_proba(x_test))
# Plot the Tree
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("Data")
# dot_data = tree.export_graphviz(clf, out_file=None,
#                      filled=True, rounded=True,
#                      special_characters=True)
# graph = graphviz.Source(dot_data)
# graph