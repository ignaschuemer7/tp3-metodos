import numpy as np
# Encontrar en centroide de cada cluster y a partir de estos, armar una clasificador basado en la distancia
# de una muestra a cada centroide.

# algoritmo que clasifica las muestras de un conjunto de datos. Los parametros que tenemos son:
# X: conjunto de datos en un espacio
# C: todos los centroides del conjunto en ese espacio, por lo tanto numero de clusters
# r: definimos un radio de clasificacion, si la distancia de una muestra a un centroide es menor a r, la muestra pertenece a ese cluster
# devuelve un vector de etiquetas, donde cada etiqueta es el numero de cluster al que pertenece la muestra
def clasificar(X, C, r):
    #inicializar el vector de etiquetas
    etiquetas = np.zeros(X.shape[0])
    #para cada muestra
    for i in range(X.shape[0]):
        #calcular la distancia a cada centroide
        distancias = np.zeros(C.shape[0])
        for j in range(C.shape[0]):
            distancias[j] = np.linalg.norm(X[i,:]-C[j,:])
        #si la distancia minima es menor a r, la muestra pertenece a ese cluster
        if np.min(distancias) < r:
            etiquetas[i] = np.argmin(distancias)
        else:
            #si la distancia minima es mayor a r, la muestra no pertenece a ningun cluster
            etiquetas[i] = -1
    return etiquetas

