import pandas as pd
from sklearn.cluster import KMeans

# Read the dataset and store in a Pandas DataFrame
df = pd.read_csv("C:\\repos\\dataset\\connekt\\agrupamento.csv")
# Read Centroids database
dfc = pd.read_csv("C:\\repos\\dataset\\connekt\\centroides_iniciais.csv")
nclusters = 7
init_centroids = dfc.iloc[0:nclusters, :]
# Fit the Model
kmeans = KMeans(n_clusters = nclusters, init = init_centroids, random_state = 42).fit(df)
# print(kmeans.labels_)
print("Predicted: ", kmeans.predict(df.iloc[:]))
print("Clusters Centers:")
print(kmeans.cluster_centers_)
