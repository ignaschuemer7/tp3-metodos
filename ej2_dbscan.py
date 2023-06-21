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

def Centroids_clasificator(data, centroids):
    subsets = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        distancias = np.zeros(centroids.shape[0])
        for j in range(centroids.shape[0]):
            distancias[j] = np.linalg.norm(data[i] - centroids[j])
        subsets[i] = np.argmin(distancias)+2
    return subsets


def main():
    # Cargar el dataset
    X_original = np.loadtxt('dataset_clusters.csv', delimiter=',')

    #Descompocision en valores singulares
    U,S,V = np.linalg.svd(X_original)

    # Graficar S
    plt.plot(range(1, len(S)+1), S)
    plt.xlabel('Dimensión', fontsize=15)
    plt.ylabel('Valor singular 'r'$\sigma_i$', fontsize=15)
    plt.title('Valores singulares de X', fontsize=18)
    plt.show()

    # mismo grafico pero s desde 1 a 10
    plt.plot(range(1, len(S[0:10])+1), S[0:10], 'o')
    plt.xticks(np.arange(0, 11, 1))
    plt.xlabel('Dimensión', fontsize=15)
    plt.ylabel('Valor singular 'r'$\sigma_i$', fontsize=15)
    plt.title('Valores singulares de X', fontsize=18)
    plt.show()

    #comparar que tan grande es el 1er valor con respecto al 3ro
    print("El 3er componente es ", ((S[2]/S[3])-1) * 100, "% mas grande que el 3ro")
    print(S[0], S[1], S[2], S[3]) 

    # Parámetros de DBSCAN
    epsilon = 0.67
    min_muestras = 10
    # Realizar el clustering con DBSCAN 
    X = PCA(X_original,2)
    dbscan_data = dbscan(X, epsilon, min_muestras)
    # Graficar los clusters
    centroids = []
    for i in range(1, np.max(dbscan_data)):
        centroids.append(np.mean(X[dbscan_data == i+1], axis=0))
    centroids = np.array(centroids)
    show_clustering_data(X, dbscan_data, centroids, title='DBSCAN')
    distance_clasificator = Centroids_clasificator(X, centroids)
    show_clustering_data(X, distance_clasificator, centroids, title='Clasificador de datos por distancia a centroides')

    # comparar dbscan con clasificador de datos por distancia a centroides, marcar con los puntos que clasifican distinto
    # restar ambos tags y ver cuales son distintos
    dif = (dbscan_data - distance_clasificator) 
    # si el valor es 0, clasificaron igual
    # si el valor es distinto de 0, clasificaron distinto
    # definir eso en el label
    dif[dif != 0] = 1
    plt.scatter(X[:,0], X[:,1], c=dif, cmap='Paired', label='Misma clasificacion')  
    plt.title('Clasificador de datos por distancia a centroides y dbscan',fontsize=15)
    plt.legend(loc='best', fontsize=14)
    plt.xlim(-3.5, 3.5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.ylim(-3.5, 5)
    plt.show()
    plt.show()
    #cuantificar el error de clasificacion
    #calcular la cantidad de puntos que clasifican distinto
    #dividir por la cantidad de puntos totales
    #multiplicar por 100 para tener el porcentaje de error
    cant_errores = np.count_nonzero(dif)
    error = (cant_errores/len(dif))*100
    print("El error de clasificacion es de: ", error, "%", 'con cantidad de errores: ', cant_errores, 'de un total de: ', len(dif), 'puntos')

    # Parámetros de DBSCAN para 4 dimensiones, luego hacemos un histograma para ver la distribución de los clusters	
    epsilon_4d = 1.04
    min_muestras_4d = 10
    X_4d = PCA(X_original, 4)
    # # Realizar el clustering con DBSCAN para 4 dimensiones
    labels_4d = dbscan(X_4d, epsilon_4d, min_muestras_4d)
    print(np.max(labels_4d))
    # Graficar los histogramas para poder visualisar aglomeramiento de datos, no podemos verlo en 4 dimensiones por eso hacemos un histograma
    #graficar los histogramas de los clusters en una misma imagen
    plt.figure()
    for i in range(1, np.max(labels_4d)):
        plt.hist(X_4d[labels_4d == i+1, 0], bins=20, alpha=0.5, label='Cluster {}'.format(i+1))  
    plt.legend()
    plt.title('Densidad de la primera componente de los datos agrupados por clusters')
    plt.xlabel('Componente 1')
    plt.ylabel('Densidad de puntos')
    plt.show()

    # hacer kmeans para 2, 4, 20, 106 dimensiones y ver como cambia la varianza intracluster con respecto a la cantidad de dimensiones
    # y ver como cambia el tamaño de los clusters con respecto a la cantidad de dimensiones
    X_2d = PCA(X_original, 2)
    clusters2, centroids2 = linear_kmeans(X_2d, 2)
    X_4d = PCA(X_original, 4)
    clusters4, centroids4 = linear_kmeans(X_4d, 2)
    X_20d = PCA(X_original, 20)
    clusters20, centroids20 = linear_kmeans(X_20d, 2)
    X_106d = PCA(X_original, 106)
    clusters106, centroids106 = linear_kmeans(X_106d, 2)
    #plotear las primeras 2 componentes de cada una de las proyecciones
    #quiero ver los puntos distintos con respecto a la 2da componente
    clusters4 = clusters4 - clusters2
    clusters20 = clusters20 - clusters2
    clusters106 = clusters106 - clusters2
    #en un mismo plot graficar los puntos en los que discrepan los clusters de la dim 20 y 106
    plt.figure()
    plt.scatter(X_20d[clusters20 != 0, 0], X_20d[clusters20 != 0, 1], c='r', label='Discrepancia entre 2D y 20D')
    plt.scatter(X_106d[clusters106 != 0, 0], X_106d[clusters106 != 0, 1], c='b', label='Discrepancia entre 2D y 106D')
    plt.legend(loc='best', fontsize=14)
    plt.title("Discrepancia de clusters a diferentes dimensiones", fontsize=16)
    plt.xlabel('Componente 1', fontsize=14)
    plt.ylabel('Componente 2', fontsize=14)
    plt.show()


    #hacer lo mismo pero mostrar en una figura las 4 componentes y DBSCAN
    epsilon_4d = 1.04
    min_muestras_4d = 10
    X_4d = PCA(X_original, 4)
    # # Realizar el clustering con DBSCAN para 4 dimensiones
    labels_4d = dbscan(X_4d, epsilon_4d, min_muestras_4d)
    fig, axs = plt.subplots(2, 2)
    max = np.max(labels_4d)
    fig.suptitle('Densidad de las muestras en cada componente del vector (4D)', fontsize=16)
    for i in range(2):
        for j in range(2):
            #componente 1, 2, 3 y 4
            for k in range(1, max):
                #tener en cuenga que hay que graficar segun 0,1,2,3 X_4d[labels_4d == i+1, 0] en esto
                if i==1 and j==0:
                    axs[i, j].hist(X_4d[labels_4d == k+1, 2], bins=20, alpha=0.5, label='Cluster {}'.format(k+1))
                    continue
                if i==1 and j==1:
                    axs[i, j].hist(X_4d[labels_4d == k+1, 3], bins=20, alpha=0.5, label='Cluster {}'.format(k+1))
                    continue

                axs[i, j].hist(X_4d[labels_4d == k+1, i+j], bins=20, alpha=0.5, label='Cluster {}'.format(k+1))
    axs[0, 0].set_title('Componente 1', fontsize=14)
    axs[0, 0].set_ylabel('Densidad de puntos', fontsize=14)
    axs[0, 1].set_title('Componente 2', fontsize=14)
    axs[1, 0].set_title('Componente 3', fontsize=14)
    axs[1, 0].set_ylabel('Densidad de puntos', fontsize=14)
    axs[1, 1].set_title('Componente 4', fontsize=14)
    #definir los labels de los ejes
    fig.tight_layout()
    plt.show()

    # Histogramas de las componentes 20 y 106
    epsilon_20d = 1.
    min_muestras_20d = 10
    X_20d = PCA(X_original, 20)
    # Realizar el clustering con DBSCAN para 20 dimensiones
    labels_20d = dbscan(X_20d, epsilon_20d, min_muestras_20d)
    #parametros para 106 dimensiones
    epsilon_106d = 1.
    min_muestras_106d = 10
    X_106d = PCA(X_original, 106)
    # Realizar el clustering con DBSCAN para 106 dimensiones
    labels_106d = dbscan(X_106d, epsilon_106d, min_muestras_106d)
    #ploteamos en una figura de 1x2, el componente 20 y el componente 106 para ver que hay ruido en ambos y no se idetifican clusters
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Densidad de las muestras (20D y 106D)', fontsize=16)
    axs[0].hist(X_20d[labels_20d == -1, 19], bins=20, alpha=0.5, label='Cluster {}'.format(1))
    axs[0].set_title('Componente 20 (20D)', fontsize=14)
    axs[1].hist(X_106d[labels_106d == -1, 105], bins=20, alpha=0.5, label='Cluster {}'.format(1))
    axs[1].set_title('Componente 106 (106D)', fontsize=14)
    axs[0].set_ylabel('Densidad de puntos', fontsize=14)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()