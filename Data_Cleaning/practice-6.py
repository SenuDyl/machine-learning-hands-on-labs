# Clustering with K-means
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("../Datasets/FE Course Data/ames.csv")
X = df.copy()
y = X.pop('SalePrice')
features = ["LotArea", "TotalBsmtSF", "FirstFlrSF", "SecondFlrSF","GrLivArea"]

X_scaled = X.loc[:, features]
X_scaled = (X_scaled - X_scaled.mean(axis=0))/X_scaled.std(axis=0)

kmeans = KMeans(n_clusters=10, random_state=0)
X["Cluster"] = kmeans.fit_predict(X_scaled)
print(X.head())

# Add centroid coordinates as features
centroids = kmeans.cluster_centers_  # shape = (n_clusters, n_features)
# Get cluster labels
labels = X["Cluster"].values  # 1D array of cluster assignments

# For each row, get the coordinates of its cluster centroid
X_cd = pd.DataFrame(centroids[labels], columns=[f"Centroid_{i}" for i in range(X_scaled.shape[1])])

# Add these centroid features to X
X = X.join(X_cd)
print(X.head())