import numpy as np
import matplotlib.pyplot as plt
from ej2 import *
from ej2Clasificador import *

# Calcular la matriz de distancias
def calcular_matriz_de_distancias(X):
    """
    Calcula la matriz de distancias de un dataset X
    """
    n = X.shape[0]
    matriz_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matriz_dist[i, j] = np.linalg.norm(X[i, :] - X[j, :])
            matriz_dist[j, i] = matriz_dist[i, j]
    return matriz_dist

# Encontrar los vecinos de un punto
def encontrar_vecinos(X, punto_indice, epsilon):
    vecinos = []
    for i in range(len(X)):

        if X[punto_indice, i] <= epsilon:
            vecinos.append(i)
    return vecinos

# Expandir el cluster
def expandir_cluster(X, punto_indice, vecinos, cluster, epsilon, min_muestras):
    cluster[punto_indice] = 1
    i = 0
    while i < len(vecinos):
        vecino_indice = vecinos[i]
        if cluster[vecino_indice] == 0:
            cluster[vecino_indice] = 1
            nuevos_vecinos = encontrar_vecinos(X, vecino_indice, epsilon)
            if len(nuevos_vecinos) >= min_muestras:
                vecinos += nuevos_vecinos
        i += 1



def dbscan(X, epsilon, min_muestras):
    # Calcular la matriz de distancias
    matriz_distancias = calcular_matriz_de_distancias(X)
    
    n = X.shape[0]
    cluster = np.zeros(n, dtype=int)  # 0 - unvisited, 1 - visited
    
    cluster_id = 1
    for i in range(n):
        if cluster[i] != 0:
            continue  # already visited
        vecinos = encontrar_vecinos(matriz_distancias, i, epsilon)
        if len(vecinos) < min_muestras:
            cluster[i] = -1  # mark as noise
        else:
            cluster_id += 1
            expandir_cluster(matriz_distancias, i, vecinos, cluster, epsilon, min_muestras)
            cluster[cluster == 1] = cluster_id
    return cluster



def main():
    # Cargar el dataset
    X_original = np.loadtxt('dataset_clusters.csv', delimiter=',')
    X = PCA(X_original,2)

    # Parámetros de DBSCAN
    epsilon = 0.67
    min_muestras = 10

    # Realizar el clustering con DBSCAN
    dbscan_data = dbscan(X, epsilon, min_muestras)
    # Graficar los clusters
    centroids = []
    for i in range(1, np.max(dbscan_data)):
        centroids.append(np.mean(X[dbscan_data == i+1], axis=0))
    centroids = np.array(centroids)

    show_clustering_data(X, dbscan_data, centroids, title='DBSCAN')
    distance_clasificator = find_subsets(X, centroids)
    show_clustering_data(X, distance_clasificator, centroids, title='Clasificador de datos por distancia a centroides')


    # Cargar el dataset (3 dimensiones)
    X_3d = PCA(X_original, 3)

    # Visualizar el dataset	(3 dimensiones)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2])
    plt.show()

    # Parámetros de DBSCAN para 3 dimensiones
    epsilon_3d = 1.04
    min_muestras_3d = 10

    # Realizar el clustering con DBSCAN para 3 dimensiones
    labels_3d = dbscan(X_3d, epsilon_3d, min_muestras_3d)

    # Graficar los clusters
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=labels_3d, cmap='viridis')
    plt.title('Clustering $\epsilon = 1.04$ y min_muestras = 10')
    plt.xlabel('Eje X') #IMPORTANTE VER QUE PONER EN LOS LABEL X E Y
    plt.ylabel('Eje Y')
    plt.show()
    plt.show()


if __name__ == "__main__":
    main()