import numpy as np
from copy import deepcopy
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')

data=pd.read_csv('xclara.csv')
print("Input data and shape")
print(data.shape)
data.head()


f1=data['V1'].values
f2=data['V2'].values
plt.scatter(f1,f2,c='black',s=7)
X = np.array(list(zip(f1, f2)))

def dist(a,b,ax=1):
    return np.linalg.norm(a-b,axis=ax)

k=3
C_x=np.random.randint(0, np.max(X)-10, size=k)
C_y = np.random.randint(0, np.max(X)-20, size=k)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
C=np.array(list(zip(C_x,C_y)),dtype=np.float32)
print("Initial Centroids")
print(C)

C_old=np.zeros(shape=C.shape)
#cluster list for labels
clusters=np.zeros(len(X))
error=dist(C,C_old,None)

while error!=0:
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)
colors = ['r', 'g', 'b']
fig, ax = plt.subplots()
print(clusters)
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='y')
plt.show()
  
'''
==========================================================
scikit-learn
==========================================================
'''
from sklearn.cluster import KMeans
km=KMeans(3)
km=km.fit(X)
lables=km.predict(X)
centroids=km.cluster_centers_
print("Centroid values")
print("Scratch")
print(C) # From Scratch
print("sklearn")
print(centroids)
