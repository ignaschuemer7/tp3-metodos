
"""
En el archivo dataset_clusters.csv se encuentra el dataset X. Este contiene un conjunto de n muestras
{x1, x2, . . . , xi, . . . , xn}
con xi ∈ Rp (X es por lo tanto una matriz de n×p dimensiones). Si bien el conjunto tiene, a priori, dimensión
alta, suponemos que las muestras no se distribuyen uniformemente, por lo que podremos encontrar grupos
de muestras (clusters) similares entre sí. La similaridad entre un par de muestras xi, xj se puede medir
utilizando una función no-lineal de su distancia euclidiana:
para algún valor de σ.
1. Determinar si existen clusters o grupos de alta similaridad entre muestras en el dataset.
2. Determinar a que cluster pertenece cada muestra xi
3. Encontrar en centroide de cada cluster y a partir de estos, armar una clasificador basado en la distancia
de una muestra a cada centroide.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Cargar el dataset
X = np.loadtxt('dataset_clusters.csv', delimiter=',')
print(X.shape)

# Visualizar el dataset	
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# determinar si existen clusters o grupos de alta similaridad entre muestras en el dataset.
# Determinar a que cluster pertenece cada muestra xi
# Encontrar en centroide de cada cluster y a partir de estos, armar una clasificador basado en la distancia
# de una muestra a cada centroide.

reduced_data = PCA(n_components=2).fit_transform(X)
kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
kmeans.fit(reduced_data)
h = .02
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired, aspect='auto', origin='lower')
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3, color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\nCentroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()



