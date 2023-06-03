import numpy as np 
# Encontrar en centroide de cada cluster y a partir de estos, armar una clasificador basado en la distancia
# de una muestra a cada centroide.

# algoritmo que clasifica las muestras de un conjunto de datos. Los parametros que tenemos son:
# X: conjunto de datos en un espacio
# C: todos los centroides del conjunto en ese espacio, por lo tanto numero de clusters
def find_subsets(data, centroids):
    subsets = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        distancias = np.zeros(centroids.shape[0])
        for j in range(centroids.shape[0]):
            distancias[j] = np.linalg.norm(data[i] - centroids[j])
        subsets[i] = np.argmin(distancias)+2
    return subsets

