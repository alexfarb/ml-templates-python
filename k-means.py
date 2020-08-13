import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import operator
import warnings
warnings.filterwarnings("ignore")

# Read the dataset and store in a Pandas DataFrame
df = pd.read_csv(datapath)
# Read Centroids database
dfc = pd.read_csv(datapath)
nclusters = 16
init_centroids = dfc.iloc[0:nclusters, :]
# Fit the Model
sse = {}
sil_coeff_scores = {}
for k in range(1, nclusters+1):
    kmeans = KMeans(n_clusters = k, init = init_centroids.iloc[0:k, :], random_state = 42).fit(df)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
for i in range(2, nclusters+1):
    kmeansb = KMeans(n_clusters = i, init = init_centroids.iloc[0:i, :], random_state = 42).fit(df)
    label = kmeansb.labels_ # For Silhoutte Score
    sil_coeff = silhouette_score(df, label, metric='euclidean')
    sil_coeff_scores[i-1] = silhouette_score(df, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(i, sil_coeff))
# print(kmeans.labels_)
# print("Clusters Centers:")
# print(kmeans.cluster_centers_)
# Samples Prediction
print("Predicted: ", kmeans.predict(df.iloc[:]))
# Max Value of Silhoutte Coefficient
max_sil_score = max(sil_coeff_scores.items(), key=operator.itemgetter(1))[0]
print("Max Silhoutte Coefficient: ", max_sil_score+1)
# Plot Elbow Method
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.title("Elbow Method")
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

