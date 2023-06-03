import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Calcular la matriz de distancias
def calcular_matriz_de_similaridades(X, sigma):
    """
    Calcula la matriz de similaridades de un dataset X
    """
    # Calcular la matriz de distancias
    n = X.shape[0]
    matriz_sim = np.zeros((n,n))
    #como la matriz es simetrica, solo calculamos la mitad
    for i in range(n):
        for j in range(i+1, n):
            matriz_sim[i,j] = np.exp(-np.linalg.norm(X[i,:]-X[j,:])**2/(2*sigma**2))
    #completamos la matriz
    matriz_sim = matriz_sim + matriz_sim.T
    return matriz_sim

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
    Calcula los clusters a los que pertenece cada muestra de un dataset X_proyectado en un espacio de menor dimension
    """
    # Calcular la matriz de similaridades
    matriz_similaridades = calcular_matriz_de_similaridades(X, 0.23)
    show_matrix(matriz_similaridades, 'Matriz de similaridades')
    # Calcular la matriz de grados
    matriz_grados = calcular_matriz_grados(matriz_similaridades)
    # Calcular la matriz laplaciana
    matriz_laplaciana = matriz_grados - matriz_similaridades    
    #normalizamos la matriz laplaciana
    matriz_laplaciana = matriz_laplaciana/np.linalg.norm(matriz_laplaciana)
    show_matrix(matriz_laplaciana, 'Matriz laplaciana normalizada')
    #descomposicion en valores singulares para obtener los autovectores de la matriz laplaciana normalizada
    U, S, V = np.linalg.svd(matriz_laplaciana)
    #seleccionamos los n autovectores correspondientes a los n autovalores mas grandes
    autovectores = U[:,:n_clusters]
    #normalizamos los autovectores
    autovectores = autovectores/np.linalg.norm(autovectores)

    clusters = np.zeros((X.shape[0], n_clusters))
    for i in range(X.shape[0]):
        for j in range(n_clusters):
            clusters[i,j] = np.linalg.norm(X[i,:] - autovectores[j,:])
        clusters[i,:] = np.argsort(clusters[i,:])
    return clusters

def show_matrix(A, title):
    """
    Muestra una matriz A con un estilo de color style
    """
    plt.imshow(A)
    #agregar leyendas a los ejes
    plt.xlabel('Muestras')
    plt.ylabel('Muestras')
    plt.colorbar()
    plt.title(title)
    plt.show()

def show_clustering_data(X, clusters, centroids, title='Clustering'):
    for i in range(1, int(np.max(clusters))):
        plt.scatter(X[clusters == i+1, 0], X[clusters == i+1, 1], label=f'Cluster {i}')
        plt.scatter(centroids[i-1, 0], centroids[i-1, 1], marker='X', color='black', label=f'Centroide {i}')
    plt.title(title)
    plt.legend(loc='best')
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 5)
    plt.show()

def find_centroids(cluster):
    """
    Calcula el centriode de un cluster
    """
    return np.mean(cluster, axis=0)

    
def main():
    # # Cargar el dataset
    X = np.loadtxt('dataset_clusters.csv', delimiter=',')
    print(X.shape)
    # # Visualizar el dataset	
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    # reduccion de dimensiones a 2 por medio de pca
    X_proyectado = PCA(X, 2)
    # print(X_proyectado.shape)
    plt.scatter(X_proyectado[:, 0], X_proyectado[:, 1])
    plt.show()

    #usar Vt de la descomposicion en valores singulares para reducir la dimension de los datos, para hacer PCA
    # U, S, Vt = np.linalg.svd(X)
    # X_proyectado = X @ Vt[:2,:].T
    # plt.scatter(X_proyectado[:, 0], X_proyectado[:, 1])
    # plt.show()
    #visualizar la matriz de similaridades 
    # matriz_similaridades = calcular_matriz_de_similaridades(X_proyectado, 0.23)
    # show_matrix(matriz_similaridades, 'Matriz de similaridades')

    #mostrar los clusters en dimension 2 por medio de kmeans
    clusters = find_clusters(X_proyectado, 2)
    #corregir los clusters para que los puntos mas cercanos al centroide sean los del cluster cprrrespondiente
    #y los mas lejanos los del otro cluster
    #teniendo en cuenta la distancia de cada punto a cada centroide

    plt.scatter(X_proyectado[:, 0], X_proyectado[:, 1], c=clusters[:,0])
    # # calcular el centroide de cada cluster
    centroide1 = find_centroids(X_proyectado[clusters[:,0]==0,:])
    centroide2 = find_centroids(X_proyectado[clusters[:,0]==1,:])
    # show_clustering_data(X_proyectado, clusters[:,0], np.array([centroide1, centroide2]), 'Clustering con k-means')
    print(centroide1)
    print(centroide2)
    plt.scatter(centroide1[0], centroide1[1], c='r', marker='x', s=100)
    plt.scatter(centroide2[0], centroide2[1], c='r', marker='x', s=100) 
    plt.show()

    #clasificar las muestras 
    # clasificar(X_proyectado, [centroide1,centroide2], 1)

    # #reduccion de dimensiones a 2 por medio de svd
    # U, S, V = np.linalg.svd(X)
    # X_reducido = U[:,:2]
    # print(X_reducido.shape)
    # plt.scatter(X_reducido[:, 0], X_reducido[:, 1])
    # plt.show()

    # #reduccion de dimensiones a 3 por medio de svd
    # X_reducido = U[:,:3]
    # print(X_reducido.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_reducido[:, 0], X_reducido[:, 1], X_reducido[:, 2])
    # plt.show()

    #mostrar el dataset proyectado en el espacio de menor dimension, dimension 3
    # X_proyectado = PCA(X, 3)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_proyectado[:, 0], X_proyectado[:, 1], X_proyectado[:, 2])
    # plt.show()

if __name__ == "__main__":
    main()

# #mostrar los clusters en dimension 2(tarda)
# #%%

# """
# En el archivo dataset_clusters.csv se encuentra el dataset X. Este contiene un conjunto de n muestras
# {x1, x2, . . . , xi, . . . , xn}
# con xi ∈ Rp (X es por lo tanto una matriz de n×p dimensiones). Si bien el conjunto tiene, a priori, dimensión
# alta, suponemos que las muestras no se distribuyen uniformemente, por lo que podremos encontrar grupos
# de muestras (clusters) similares entre sí. La similaridad entre un par de muestras xi, xj se puede medir
# utilizando una función no-lineal de su distancia euclidiana:
# para algún valor de σ.
# 1. Determinar si existen clusters o grupos de alta similaridad entre muestras en el dataset.
# 2. Determinar a que cluster pertenece cada muestra xi
# 3. Encontrar en centroide de cada cluster y a partir de estos, armar una clasificador basado en la distancia
# de una muestra a cada centroide.
# """
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import misc
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.metrics import pairwise_distances_argmin_min
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# # Cargar el dataset
# X = np.loadtxt('dataset_clusters.csv', delimiter=',')

# # determinar si existen clusters o grupos de alta similaridad entre muestras en el dataset.
# # Determinar a que cluster pertenece cada muestra xi
# # Encontrar en centroide de cada cluster y a partir de estos, armar una clasificador basado en la distancia
# # de una muestra a cada centroide.
# n=2
# reduced_data = PCA(n_components=n).fit_transform(X)
# kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
# kmeans.fit(reduced_data)
# h = .02
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired, aspect='auto', origin='lower')
# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3, color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\nCentroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()

# # %%
