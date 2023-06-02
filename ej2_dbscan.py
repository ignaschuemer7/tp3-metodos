import numpy as np
import matplotlib.pyplot as plt
from ej2 import *

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

    # Visualizar el dataset	
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    # Parámetros de DBSCAN
    epsilon = 0.67
    min_muestras = 10

    # Realizar el clustering con DBSCAN
    labels = dbscan(X, epsilon, min_muestras)

    # Graficar los clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title('Clustering')
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.show()

if __name__ == "__main__":
    main()