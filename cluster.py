import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# load the dataset

df = pd.read_csv('data/mall_customers.csv')

#print(df.head())
#print(df.describe())
#print(df.info())

# select features for clustering

#X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters=5, random_state=42, max_iter= 500)

# fit the model
df['Cluster'] = kmeans.fit_predict(X)

print(df.head())

# plot the clusters
plt.figure(figsize=(10,5))

plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)

# plot the centroids
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker='X',
    s=200
)

plt.title("Customer segmentation using k-means")
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

