#CLUSTERING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pandas import datetime
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

input_file = 'https://raw.githubusercontent.com/lcphy/Digital-Innovation-Lab/master/total_reviews.csv'
df = pd.read_csv(input_file, sep=';', header = 0)

df = df[['User Id', 'Religious', 'Sports','Theatre', 'Picnic', 'Nature','Shopping']]
df.head()

# (Nature - Picnic) Analysis 2 Clusters

X_s = df.iloc[:, [4,5]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_s)
    wcss.append(kmeans.inertia_)

plt.clf()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
display(plt.show())

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_s)

# Visualising the clusters
plt.clf()
plt.scatter(X_s[y_kmeans == 0, 0], X_s[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_s[y_kmeans == 1, 0], X_s[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
#plt.scatter(X_s[y_kmeans == 2, 0], X_s[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.legend()
display(plt.show())

#Proof

X = df.iloc[:, [4,5]].values

# Using the dendrogram to find the optimal number of clusters
plt.clf()
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Users')
plt.ylabel('Euclidean distances')
display(plt.show())

# (Nature - Shopping) Analysis 3 Clusters

X_s = df.iloc[:, [5,6]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_s)
    wcss.append(kmeans.inertia_)

plt.clf()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
display(plt.show())

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_s)

# Visualising the clusters
plt.clf()
plt.scatter(X_s[y_kmeans == 0, 0], X_s[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_s[y_kmeans == 1, 0], X_s[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_s[y_kmeans == 2, 0], X_s[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.legend()
display(plt.show())

#Proof

X = df.iloc[:, [5,6]].values

# Using the dendrogram to find the optimal number of clusters
plt.clf()
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Users')
plt.ylabel('Euclidean distances')
display(plt.show())

#Using Pearson Correlation

import seaborn as sns

input_file = "/content/gdrive/My Drive/Colab Notebooks/total_reviews.csv"
df = pd.read_csv(input_file, sep=';', header = 0) 

plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()