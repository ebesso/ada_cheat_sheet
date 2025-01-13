import pandas as pd
import numpy as np

X = pd.DataFrame()
y = pd.DataFrame()

#################################################################################################################
# K-MEANS and DBSCAN
from sklearn.cluster import KMeans, DBSCAN
# https://scikit-learn.org/stable/api/sklearn.cluster.html

# K-Means
kmean = KMeans(n_clusters=10, random_state=42).fit(X) # Initializes with k-means++ per automatic
kmean.labels_ # Returns the predicted labels for X
kmean.cluster_centers_ # Returns the coordinates of mean of the clusters
kmean.inertia_ # Returns the sum of squares to nearest cluster mean

# DBSCAN
# Core point = a point with atleast min_samples amount of neighnors within an epsilon radius. Explores out from
# core point, but no border points (within epsilon of a core point but not atleast min_samples many)

labels = DBSCAN(eps=0.5, min_samples= 5).fit_predict(X) # DBSCAN returns the labels of the data


#################################################################################################################
# PCA AND T-SNE

# We can reduce dimensions in our data in order to visualize it
from sklearn.decomposition import PCA
# https://scikit-learn.org/stable/api/sklearn.decomposition.html
X_reduced_pca = PCA(n_components=2).fit(X).transform(X)

from sklearn.manifold import TSNE
# https://scikit-learn.org/stable/api/sklearn.manifold.html
X_reduced_tsne = TSNE(n_components=2, init='random', learning_rate='auto', random_state=0).fit_transform(X)






