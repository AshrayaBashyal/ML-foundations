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