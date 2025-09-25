# What is Clustering?

# Clustering = grouping similar data points together.
# Example:
# Suppose we have patients with features (Age, Cholesterol, Blood Pressure).
# K-Means might find 2 groups:
# Cluster 1 → younger patients, lower cholesterol
# Cluster 2 → older patients, higher cholesterol
# 👉 Useful when we don’t know labels (unsupervised).

# How does K-Means Work?

# Choose k (number of clusters).
# Randomly place k points (centroids).
# Assign each data point to the nearest centroid.
# Recalculate centroids (mean of each cluster).
# Repeat until stable.

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset
data = {
    "age": [25, 30, 35, 40, 50, 55, 60, 65],
    "cholesterol": [200, 210, 220, 230, 250, 260, 270, 280]
}

df = pd.DataFrame(data)


# Train K-Means
X = df[["age", "cholesterol"]]

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)

df["cluster"] = kmeans.labels_
print(df)

# Visualize Clusters
plt.figure()

plt.scatter(df["age"], df["cholesterol"], c=df["cluster"],  cmap="viridis", s=100)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c="red", marker="X", s=200, label="Centroid")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.legend()
# plt.show()

# Choosing k (Elbow Method)
# Select k from where it bends like an elbow

plt.figure()

inertia = []
for k in range(1,6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(range(1,6), inertia, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
