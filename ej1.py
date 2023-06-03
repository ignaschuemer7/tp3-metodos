from PIL import Image
import numpy as np
import os
from pylab import *
#printear las Imagenes de matplot en formato latex article
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

"""
En el archivo dataset_Imagenes.zip se encuentran varias imágenes. Cada Imagen es una matriz de p × p que
puede representarse como un vector x ∈ Rp∗p. A su vez, es posible armar un matriz de datos apilando los
vectores de cada Imagen. Se desea aprender una representación de baja dimensión de las imágenes mediante
una descomposición en valores singulares.
1. Visualizar en forma matricial p×p las primeras y las últimas dimensiones (autovectores) de la descomposición
obtenida. ¿Qué diferencias existen entre unas y otras? ¿Qué conclusiones pueden sacar?
"""
def create_matrix_data(directorio):
    # #queremos armar una matriz de datos apilando los vectores de cada Imagen, la matriz de datos va a tener p*p filas y tantas columnas como imagenes
    # #para eso vamos a recorrer las imagenes y apilar los vectores de cada Imagen como columnas de la matriz de datos

    matrices_Imagenes = []

    #queremos armar una matriz de datos apilando los vectores de cada Imagen, la matriz de datos va a tener p*p filas y tantas columnas como imagenes
    #para eso vamos a recorrer las imagenes y apilar los vectores de cada Imagen como columnas de la matriz de datos

    for nombre_archivo in os.listdir(directorio):
        if nombre_archivo.endswith('.jpeg'):
            ruta_Imagen = os.path.join(directorio, nombre_archivo)
            
            # Abrir la Imagen con PIL
            Imagen = Image.open(ruta_Imagen)
            
            # Convertir la Imagen a una matriz NumPy
            matriz_Imagen = np.array(Imagen)
            
            # Agregar la imagen como un vector con el flatten
            vector = matriz_Imagen.flatten()

            # Agregar el vector como columna en la matriz
            matrices_Imagenes.append(vector)

    return np.array(matrices_Imagenes).T

def show_matrix(matrices_Imagenes, title, cmap='gray', save = False, subplot = False): 
    plt.figure(figsize=(10,10))
    plt.imshow(matrices_Imagenes, cmap=cmap)
    plt.title(title)
    show()
    if save:
        plt.savefig(title+'.svg', format="svg")

def show_first_last_eigenvec(Mat, p, k, save = False):
    # Visualizar en forma matricial p×p las primeras y las últimas dimensiones (autovectores) de la descomposición

    #primeros autovectores
    first = Mat[:,k]
    #ultimos autovectores
    last = Mat[:,-k]

    #mostrar los primeros k autovectores y los ultimos k autovectores
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(first.reshape(p,p), cmap='gray')
    axs[0].set_title(f'primeros {str(k)} autovectores')
    axs[1].imshow(last.reshape(p,p), cmap='gray')
    axs[1].set_title(f'ultimos {str(k)} autovectores')
    plt.show()
    if save:
        plt.savefig('first_last_eigenvec.svg', format="svg")

def find_d(U, S, Vt, Imagen, error):
    norma_frobenius = np.linalg.norm(Imagen, ord='fro')
    for d in range(1, S.shape[0]):
        # Obtener la matriz de la Imagen reconstruida de 28x28
        Imagen_reconstruida = U[:, :d] @ np.diag(S[:d]) @ Vt[:d, :]
        # Calcular el error de la compresión
        error_actual = np.linalg.norm(Imagen - Imagen_reconstruida, ord='fro') / norma_frobenius
        
        # Imprimir el error
        if error_actual < error:
            break
    return d

def main():
    directorio = 'dataset_Imagenes'
    matrices_Imagenes = create_matrix_data(directorio)

    show_matrix(matrices_Imagenes, 'matriz de datos')
    # Imprimir el número de imágenes procesadas
    # Obtener el tamaño de la matriz de una Imagen con el primer vector 
    p = np.sqrt(matrices_Imagenes[:,0].shape[0]).astype(int)
    # Calcular la descomposición en valores singulares
    U, S, Vt = np.linalg.svd(matrices_Imagenes, full_matrices=True)
    # Visualizar en forma matricial p×p las primeras y las últimas dimensiones (autovectores)\
    #tomamos 1 autovector
    show_first_last_eigenvec(U, p, 1)


    """
    Dada una Imagen cualquiera del conjunto (por ejemplo la primera) encontrar d, el número mínimo
    de dimensiones a las que se puede reducir la dimensionalidad de su representación mediante valores
    singulares tal que el error entre la Imagen comprimida y la original no exceda el 5% bajo la norma de
    Frobenius. ¿Qué error obtienen si realizan la misma compresión (con el mismo d) para otra Imagen
    cualquiera del conjunto?
    """
    # Obtener la primera Imagen
    Imagen = matrices_Imagenes[:,6].reshape(p,p)
    # Calcular la descomposición en valores singulares
    U, S, Vt = np.linalg.svd(Imagen, full_matrices=False)

    # Calcular la norma de Frobenius de la matriz de la Imagen
    norma_frobenius = np.linalg.norm(Imagen, ord='fro')

    d = find_d(U, S, Vt, Imagen, 0.05)
    # Calcular el error de la compresión
    Imagen_reconstruida = U[:, :d] @ np.diag(S[:d]) @ Vt[:d, :]
    error = np.linalg.norm(Imagen - Imagen_reconstruida, ord='fro') / norma_frobenius
    print(f'Error con d = {d}: {error:.3f}')

    #printear la Imagen inicial y la reconstruida en una misma figura con subplots, agregarle una leyenda
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(Imagen, cmap='gray')
    axs[0].set_title('Imagen original')
    axs[1].imshow(Imagen_reconstruida, cmap='gray', label='Imagen reconstruida')
    axs[1].set_title('Imagen reconstruida con d = ' + str(d))
    #guardar la figura como svg
    plt.savefig('Imagenes_reconstruidas.svg', format="svg")
    plt.show()

    #con el d hallado, comprimir otra Imagen cualquiera del conjunto y ver el error, img es el nuemro de la imagen 
    img = 5
    Imagen2 = matrices_Imagenes[:,img].reshape(p,p)

    # Calcular la descomposición en valores singulares
    U, S, Vt = np.linalg.svd(Imagen2, full_matrices=False)

    # Calcular la norma de Frobenius de la matriz de la Imagen
    norma_frobenius = np.linalg.norm(Imagen2, ord='fro')

    # Obtener la matriz de la Imagen reconstruida de 28x28
    Imagen_reconstruida = U[:, :d] @ np.diag(S[:d]) @ Vt[:d, :]

    # Calcular el error de la compresión
    error = np.linalg.norm(Imagen2 - Imagen_reconstruida, ord='fro') / norma_frobenius

    # Imprimir el error
    print(f'Error con d = {d}: {error:.3f}')

    #printear la Imagen inicial y la reconstruida
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(Imagen2, cmap='gray')
    axs[0].set_title('Imagen original')
    axs[1].imshow(Imagen_reconstruida, cmap='gray')
    axs[1].set_title('Imagen reconstruida con d = ' + str(d))
    plt.savefig('ImagenesErrorConDanterior.svg', format="svg")
    plt.show()

    Imagen = matrices_Imagenes[:,5].reshape(p,p)
    d = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    #en una misma figura con subplots, agregarle una leyenda
    fig, axs = plt.subplots(1, len(d))
    for i in range(len(d)):
        # Calcular la descomposición en valores singulares
        U, S, Vt = np.linalg.svd(Imagen, full_matrices=False)
        # Obtener la matriz de la Imagen reconstruida de 28x28
        Imagen_reconstruida = U[:, :d[i]] @ np.diag(S[:d[i]]) @ Vt[:d[i], :]
        #printear la Imagen inicial y la reconstruida
        axs[i].imshow(Imagen_reconstruida, cmap='gray')
        axs[i].axis('off')
    plt.savefig('ImagenesReconstruidas.svg', format="svg")
    plt.show()





if __name__ == '__main__':
    main()




