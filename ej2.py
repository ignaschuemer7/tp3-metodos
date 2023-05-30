
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
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Calcular la matriz de distancias
def calcular_matriz_distancias(X):
    """
    Calcula la matriz de distancias de un dataset X
    """
    # Calcular la matriz de distancias
    n = X.shape[0]
    matriz_distancias = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            matriz_distancias[i,j] = np.linalg.norm(X[i,:] - X[j,:])
            # print(f'Calculando distancia entre {i} y {j}')
            # print(f'Distancia: {matriz_distancias[i,j]}')
    return matriz_distancias

# Calcular la matriz de similaridades
def calcular_matriz_similaridades(X, sigma):
    """
    Calcula la matriz de similaridades de un dataset X
    """
    # Calcular la matriz de distancias
    matriz_distancias = calcular_matriz_distancias(X)
    # Calcular la matriz de similaridades
    matriz_similaridades = np.exp(-matriz_distancias**2/(2*sigma**2))
    return matriz_similaridades

# Calcular la matriz de grados
def calcular_matriz_grados(matriz_similaridades):
    """
    Calcula la matriz de grados de una matriz de similaridades
    """
    # Calcular la matriz de grados
    matriz_grados = np.diag(np.sum(matriz_similaridades, axis=1))
    return matriz_grados

#proyeccion de los datos a un espacio de menor dimension mediante el metodo de componentes principales
def PCA(X, n):
    """
    Calcula la matriz de proyeccion de los datos a un espacio de menor dimension mediante el metodo de componentes principales
    """
    # Calcular la matriz de covarianza
    covarianza = np.cov(X.T)
    # Calcular los autovalores y autovectores
    autovalores, autovectores = np.linalg.eig(covarianza)
    # Ordenar los autovectores segun los autovalores
    idx = autovalores.argsort()[::-1]   
    autovalores = autovalores[idx]
    autovectores = autovectores[:,idx]
    # Seleccionar los n autovectores correspondientes a los n autovalores mas grandes
    autovectores = autovectores[:,:n]
    # Proyectar los datos en el nuevo espacio
    X_proyectado = X @ autovectores
    return X_proyectado

def find_clusters(X, n_clusters):
    """
    Calcula los clusters a los que pertenece cada muestra de un dataset X
    """
    # Calcular la matriz de similaridades
    matriz_similaridades = calcular_matriz_similaridades(X, sigma=1)
    # Calcular la matriz de grados
    matriz_grados = calcular_matriz_grados(matriz_similaridades)
    # Calcular la matriz laplaciana
    matriz_laplaciana = matriz_grados - matriz_similaridades
    # Calcular los n autovectores correspondientes a los n autovalores mas grandes
    autovalores, autovectores = np.linalg.eig(matriz_laplaciana)
    idx = autovalores.argsort()[::-1]   
    autovalores = autovalores[idx]
    autovectores = autovectores[:,idx]
    autovectores = autovectores[:,:n_clusters]
    # Calcular la matriz de datos proyectada en el espacio de menor dimension
    X_proyectado = PCA(X, n_clusters)
    # Calcular los clusters a los que pertenece cada muestra
    clusters = np.zeros((X.shape[0], n_clusters))
    for i in range(X.shape[0]):
        for j in range(n_clusters):
            clusters[i,j] = np.linalg.norm(X_proyectado[i,:] - autovectores[j,:])
        clusters[i,:] = np.argsort(clusters[i,:])
    return clusters

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
#mostrar el dataset proyectado en el espacio de menor dimension, dimension 2

X_proyectado = PCA(X, 2)
print(X_proyectado.shape)
plt.scatter(X_proyectado[:, 0], X_proyectado[:, 1])
plt.show()
#mostrar los clusters en dimension 2(tarda)
clusters = find_clusters(X_proyectado, 2)
plt.scatter(X_proyectado[:, 0], X_proyectado[:, 1], c=clusters[:,0])    
plt.show()

#mostrar el dataset proyectado en el espacio de menor dimension, dimension 3
X_proyectado = PCA(X, 3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_proyectado[:, 0], X_proyectado[:, 1], X_proyectado[:, 2])
plt.show()


#%%

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

# determinar si existen clusters o grupos de alta similaridad entre muestras en el dataset.
# Determinar a que cluster pertenece cada muestra xi
# Encontrar en centroide de cada cluster y a partir de estos, armar una clasificador basado en la distancia
# de una muestra a cada centroide.
n=2
reduced_data = PCA(n_components=n).fit_transform(X)
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






# %%
