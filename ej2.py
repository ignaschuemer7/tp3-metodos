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

def PCA(X, n):
    """
    Calcula la matriz de proyeccion de los datos a un espacio de menor dimension mediante el metodo de componentes principales
    """
    U, S, Vt = np.linalg.svd(X)
    proyeccion = (Vt[:n,:]@X.T).T
    # invertimos el signo de la primera componente para que coincida con la de la matriz de similaridades
    proyeccion[:,0] = proyeccion[:,0]*-1
    return proyeccion

def linear_kmeans(X, n_clusters, max_iterations=100):
    # Inicialización de centroides
    centroids = X[:n_clusters, :]

    for _ in range(max_iterations):
        # Asignación de muestras a los clusters más cercanos
        distances = np.exp(-np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)**2)
        labels = np.argmax(distances, axis=1)
        
        # Actualización de los centroides
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])

        # Verificar convergencia
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids
    
    return labels, centroids

def show_clustering_data(X, clusters, centroids, title='Clustering'):
    for i in range(1, int(np.max(clusters))):
        plt.scatter(X[clusters == i+1, 0], X[clusters == i+1, 1], label=f'Cluster {i}')
        plt.scatter(centroids[i-1, 0], centroids[i-1, 1], marker='X', color='black', s=100)
    plt.title(title, fontsize=15)
    plt.legend(loc='best', fontsize=14)
    plt.xlim(-3.5, 3.5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.ylim(-3.5, 5)
    plt.savefig(f'{title}.svg', format='svg')
    plt.show()

def main():
    # Cargar el dataset
    X = np.loadtxt('dataset_clusters.csv', delimiter=',')
    U, S, Vt = np.linalg.svd(X)

    # PCA en 2D
    X_proyectado = PCA(X, 2)
    plt.scatter(X_proyectado[:, 0], X_proyectado[:, 1])
    plt.title('PCA 2D', fontsize=15)
    plt.xlabel('Componente principal 1', fontsize=12)
    plt.ylabel('Componente principal 2', fontsize=12)
    plt.savefig('PCA_2Dr.svg', format='svg')
    plt.show()

    # kmeans en 2D
    clusters, centroides = linear_kmeans(X_proyectado, 2)
    for i in range(0, 2):
        plt.scatter(X_proyectado[clusters == i, 0], X_proyectado[clusters == i, 1], label=f'Cluster {i+1}')
        plt.scatter(centroides[i, 0], centroides[i, 1], marker='X', color='black', s=100)
    plt.legend(loc='best', fontsize=14)
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Clustering con k-means', fontsize=15)
    plt.show()

if __name__ == "__main__":
    main()
