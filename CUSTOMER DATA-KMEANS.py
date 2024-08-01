IMPORTING THE DEPENDENCIES

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

DATA COLLECTION AND ANALYSIS

from google.colab import files
files.upload()

customer_data=pd.read_csv('Mall_Customers (2).csv')
customer_data

customer_data.head()

customer_data.shape

customer_data.info()

customer_data.isnull().sum()

CHOOSING THE ANNUAL INCOME COLUMN & PENDING SCORE COLUMN

x=customer_data.iloc[:,[3,4]].values

print(x)

CHOOSING NUMBERS OF CLUSTERS


wcss

# finding wcss value for number of clusters

wcss=[] #empty list wcss that will be used to store the WCSS values calculated for diff no.s of clusters.

for i in range(1,11):
  kmeans=KMeans(n_clusters=i, init='k-means++',random_state=42)
  kmeans.fit(x)

  wcss.append(kmeans.inertia_) # after fitting the model, the within_cister sum of squares (WCSS) is calculated using the inertia_attribute of the KMeans object


# plot an elbow graph

sns.set()
plt.plot(range(1,11),wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

OPTIMUM NUMBER OF CLUSTER = 5

training the K-means clustering model

kmeans=KMeans(n_clusters=5, init='k-means++',random_state=0)
#initializing the cluster centers. 'k-means++' is a smart initialization method that speeds up

#return a label for each data point based on their cluster
#fit the KMeans model and predict cluster labels
y= kmeans.fit_predict(x)

print(y)

5 CLUSTERS - 0,1,2,3,4

Visualizing all the clusters

plt.figure(figsize=(8,8))
plt.scatter(x[y==0,0], x[y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(x[y==1,0], x[y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(x[y==2,0], x[y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(x[y==3,0], x[y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(x[y==4,0], x[y==4,1], s=50, c='blue', label='Cluster 5')

#plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=100, c='cyan',label='Centroids') # this extracts x & y from the cluster

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()