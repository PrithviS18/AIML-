import pandas as pd
data = pd.read_csv('/content/cluster_data (1).csv')
data.head()
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 2)
model.fit(data)
print(model.labels_)
print (model.cluster_centers_)
from matplotlib import pyplot as plt
plt.scatter(data.iloc[model.labels_ == 0,0], data.iloc[model.labels_ == 0, 1], c = 'red',label = 'Cluster 0')
plt.scatter(data.iloc[model.labels_ == 1,0], data.iloc[model.labels_ == 1, 1], c = 'blue', label = 'Cluster 1')
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c = 'yellow', label = 'Centroids' )
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

print(model.inertia_)
#How to find the optimal number of clusters
#Elbow Method

maxx=0
score=0

for k in range(2,9):
  model2 = KMeans(n_clusters = k)
  model2.fit(data)
  val= model2.inertia_;
  if(maxx<val):
    maxx=val
    score=k

print(k)
