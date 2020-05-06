#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Read the CSV file and look at the first five rows of the data
data=pd.read_csv("Wholesale customers data.csv")
print(data.head())

#See that there is a lot of variation in the magnitude of the data. Variables like Channel and Region have low magnitude whereas variables like Fresh, Milk, Grocery, etc.have a higher magnitude.
print(data.describe())

#Since K-Means is a distance-based algorithm, this difference of magnitude can create a problem. So let’s first bring all the variables to the same magnitude
#Standardizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

#See statistics of scaled data
print(pd.DataFrame(data_scaled).describe())

#Defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=2, init='k-means++')

#Fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)

#The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion
print(kmeans.inertia_)

#We got an inertia value of almost 2600. Now, let’s see how we can use the elbow curve to determine the optimum number of clusters in Python.
#We will first fit multiple k-means models and in each successive model, we will increase the number of clusters.
#We will store the inertia value of each model and then plot it to visualize the result


# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


#Can you tell the optimum cluster value from this plot? Looking at the above elbow curve, we can choose any number of clusters between 5 to 8.
#Let’s set the number of clusters as 6 and fit the model

kmeans = KMeans(n_jobs = -1, n_clusters=6, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)

#let’s look at the value count of points in each of the above-formed clusters
frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
print(frame['cluster'].value_counts())
