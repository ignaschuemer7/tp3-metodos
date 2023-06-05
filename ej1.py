from PIL import Image
import numpy as np
import os
from pylab import *
import matplotlib.ticker as mticker

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

def show_first_eigenvec_and_s(U,S, p, save = False):
    # Visualizar en forma matricial p×p las primeras y las últimas dimensiones (autovectores) de la descomposición
    #visualizar en un grafico de 4 por 4 imagenes los primeros 16 autovectores y luego en otro grafico a S, para ver la importancia de cada autovector y su valor singular
    fig, axs = plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            axs[i,j].imshow(U[:,i*4+j].reshape(p,p), cmap='gray')
            axs[i,j].set_title(f'Ave {str(i*4+j+1)}')
            #sacar los ejes de la imagen
            axs[i,j].axis('off')
            fig.tight_layout()
    if save:
        plt.savefig('primeros_16_autovectores.svg', format="svg")
    plt.show()
    #mostrar S en un grafico de barras
    #que el tamaño de la imagen sea igual de grande que el grafico de los autovectores

    plt.bar(range(1, len(S)+1), S, color='gray')
    plt.xticks(range(0, len(S)+1))
    plt.xlabel('i') 
    plt.ylabel(r'$\sigma_i$')
    plt.title(r'Valores singulares distintos de cero $\sigma_i$')
    if save:
        plt.savefig('S.svg', format="svg")

    plt.show()
    
    

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

def punto_1(matrices_Imagenes, p): 
    p = np.sqrt(matrices_Imagenes[:,0].shape[0]).astype(int)
    U, S, Vt = np.linalg.svd(matrices_Imagenes, full_matrices=True)
    #plotear S en escala logaritmica
    show_first_eigenvec_and_s(U,S, p, save = True)


def punto_1_2(matrices_Imagenes, p, k):
    Imagen = matrices_Imagenes[:,k].reshape(p,p)
    U, S, Vt = np.linalg.svd(Imagen, full_matrices=False)
    # Calcular la norma de Frobenius de la matriz de la Imagen
    norma_frobenius = np.linalg.norm(Imagen, ord='fro')

    d = find_d(U, S, Vt, Imagen, 0.05)
    # Calcular el error de la compresión
    Imagen_reconstruida = U[:, :d] @ np.diag(S[:d]) @ Vt[:d, :]
    error = np.linalg.norm(Imagen - Imagen_reconstruida, ord='fro') / norma_frobenius
    print(f'Error con d = {d}: {error:.3f}')

    #printear la Imagen inicial y la reconstruida en una misma figura con subplots
    # que sea una figura de 4 por 4
    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(Imagen, cmap='gray')
    axs[0,0].set_title('Imagen original')
    axs[0,1].imshow(Imagen_reconstruida, cmap='gray', label='Imagen reconstruida')
    axs[0,1].set_title('D = ' + str(d) +', error = ' + str(error)[:5])
    #sacar los ejes de la imagen
    axs[0,0].axis('off')
    axs[0,1].axis('off')
    fig.tight_layout()
    #guardar la figura como svg
    # plt.savefig('Imagenes_reconstruidas.svg', format="svg")
    # plt.show()

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
    axs[1,0].imshow(Imagen2, cmap='gray')
    axs[1,0].set_title('Imagen original')
    #sacar los ejes de la imagen
    axs[1,0].axis('off')

    axs[1,1].imshow(Imagen_reconstruida, cmap='gray')
    axs[1,1].set_title('D = ' + str(d) +', error = ' + str(error)[:5])
    #sacar los ejes de la imagen
    axs[1,1].axis('off')
    fig.tight_layout()
    plt.savefig('ComparacionEntre2img.svg', format="svg")
    plt.show()

    Imagen = matrices_Imagenes[:,5].reshape(p,p)
    d = [1, 3, 5, 7, 9, 11, 13, 15]
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
        axs[i].set_title('d = ' + str(d[i]))
    plt.savefig('ImagenesReconstruidas.svg', format="svg")
    plt.show()

    #printear el error de la compresión para cada d y para cada imagen motrar la curva
    #Seleccionamos las imagenes a las que le vamos a calcular el error
    img = [2, 3, 5, 6]

    #primero mostramos en una figura las imagenes originales
    fig, axs = plt.subplots(2,2)
    for i in range(len(img)):
        Imagen = matrices_Imagenes[:,img[i]].reshape(p,p)
        axs[i//2,i%2].imshow(Imagen, cmap='gray')
        axs[i//2,i%2].axis('off')
        axs[i//2,i%2].set_title('Imagen ' + str(i+1))
    plt.savefig('ImagenesOriginales.svg', format="svg")
    plt.show()

    #calculamos el error para cada imagen y para cada d
    d = np.arange(1, 28, 1)
    error = np.zeros((d.shape[0], len(img)))
    for i in range(len(d)):
        for j in range(len(img)):
            Imagen = matrices_Imagenes[:,img[j]].reshape(p,p)
            # Calcular la descomposición en valores singulares
            U, S, Vt = np.linalg.svd(Imagen, full_matrices=False)
            # Obtener la matriz de la Imagen reconstruida de 28x28
            Imagen_reconstruida = U[:, :d[i]] @ np.diag(S[:d[i]]) @ Vt[:d[i], :]
            # Calcular la norma de Frobenius de la matriz de la Imagen
            norma_frobenius = np.linalg.norm(Imagen, ord='fro')
            # Calcular el error de la compresión
            error[i,j] = np.linalg.norm(Imagen - Imagen_reconstruida, ord='fro') / norma_frobenius
            #hacer el porcentaje de que tan bien se comprime 100 es la mejor
            error[i,j] = error[i,j] * 100

    #agregar el signo de porcentaje en el eje y
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
    #que el eje x sea de 1 en 1
    plt.xticks(np.arange(1, 28, 2))
    #plotear el error en un mismo grafico indicando la imagen en las leyendas
    plt.title('Error de la compresión de cada imagen para d = 1, 2, ..., 27')
    #agregar malla al grafico
    plt.grid()
    #printear la curva
    plt.plot(d, error)
    plt.xlabel('d')
    plt.ylabel('Error')
    plt.legend(['Imagen 1', 'Imagen 2', 'Imagen 3', 'Imagen 4'])
    plt.savefig('ErrorCompresionPorImagen.svg', format="svg")
    plt.show()
   

def main():
    directorio = 'dataset_imagenes'
    matrices_Imagenes = create_matrix_data(directorio)
    p = np.sqrt(matrices_Imagenes[:,0].shape[0]).astype(int)
    # punto_1(matrices_Imagenes, p)
    
    """
    Dada una Imagen cualquiera del conjunto (por ejemplo la primera) encontrar d, el número mínimo
    de dimensiones a las que se puede reducir la dimensionalidad de su representación mediante valores
    singulares tal que el error entre la Imagen comprimida y la original no exceda el 5% bajo la norma de
    Frobenius. ¿Qué error obtienen si realizan la misma compresión (con el mismo d) para otra Imagen
    cualquiera del conjunto?
    """
    # Obtener la primera Imagen
    NumImagen = 6
    punto_1_2(matrices_Imagenes, p, NumImagen)





if __name__ == '__main__':
    main()




