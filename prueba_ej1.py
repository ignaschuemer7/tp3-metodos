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
directorio = 'dataset_Imagenes'

matrices_Imagenes = []

for nombre_archivo in os.listdir(directorio):
    if nombre_archivo.endswith('.jpeg'):
        ruta_Imagen = os.path.join(directorio, nombre_archivo)
        
        # Abrir la Imagen con PIL
        Imagen = Image.open(ruta_Imagen)
        
        # Convertir la Imagen a una matriz NumPy
        matriz_Imagen = np.array(Imagen)
        
        # Agregar la matriz de la Imagen a la lista
        matrices_Imagenes.append(matriz_Imagen)

# Imprimir el número de imágenes procesadas
print(f'Se importaron {len(matrices_Imagenes)} imágenes.')
# Obtener el tamaño de la matriz de una Imagen
m = matrices_Imagenes[0].shape[0]

# Matriz que almacenará los vectores de cada Imagen
matriz_columnas = np.empty((m * m, len(matrices_Imagenes)))

# Recorrer las matrices de las imágenes
for i, matriz_Imagen in enumerate(matrices_Imagenes):
    vector = matriz_Imagen.reshape(-1)  # Convertir la matriz en un vector
    matriz_columnas[:, i] = vector  # Agregar el vector como columna en la matriz

#printear la matriz de columnas

# Imprimir la matriz de columnas resultante
# print(matriz_columnas)
print(matriz_columnas.shape)

# Calcular la descomposición en valores singulares
U, S, Vt = np.linalg.svd(matriz_columnas, full_matrices=False)

# Imprimir las dimensiones de las matrices resultantes
first_dimension = U[:, 0].reshape(m , m)
last_dimension = U[:, -1].reshape(m , m)

# Visualizar las primeras y últimas dimensiones

# printear la Imagen resultante
# plt.imshow(first_dimension, cmap='gray')
# plt.show()

# plt.imshow(last_dimension, cmap='gray')
# plt.show()


"""
Dada una Imagen cualquiera del conjunto (por ejemplo la primera) encontrar d, el número mínimo
de dimensiones a las que se puede reducir la dimensionalidad de su representación mediante valores
singulares tal que el error entre la Imagen comprimida y la original no exceda el 5% bajo la norma de
Frobenius. ¿Qué error obtienen si realizan la misma compresión (con el mismo d) para otra Imagen
cualquiera del conjunto?
"""
# Obtener la primera Imagen
Imagen = matrices_Imagenes[0]

# Calcular la descomposición en valores singulares
U, S, Vt = np.linalg.svd(Imagen, full_matrices=False)

# Calcular la norma de Frobenius de la matriz de la Imagen
norma_frobenius = np.linalg.norm(Imagen, ord='fro')

# Calcular el error de la compresión para distintos valores de d
for d in range(1, 101):
    # Obtener la matriz de la Imagen reconstruida de 28x28
    Imagen_reconstruida = U[:, :d] @ np.diag(S[:d]) @ Vt[:d, :]
    
    # Calcular el error de la compresión
    error = np.linalg.norm(Imagen - Imagen_reconstruida, ord='fro') / norma_frobenius
    
    # Imprimir el error
    if error <= 0.05:
        print(f'Error con d = {d}: {error:.3f}')
        break

#printear la Imagen inicial y la reconstruida en una misma figura con subplots, agregarle una leyenda
fig, axs = plt.subplots(1, 2)
axs[0].imshow(Imagen, cmap='gray')
axs[0].set_title('Imagen original')
axs[1].imshow(Imagen_reconstruida, cmap='gray', label='Imagen reconstruida')
axs[1].set_title('Imagen reconstruida con d = ' + str(d))
#guardar la figura como svg
plt.savefig('Imagenes_reconstruidas.svg', format="svg")
plt.show()

#con el d hallado, comprimir otra Imagen cualquiera del conjunto y ver el error
Imagen2 = matrices_Imagenes[4]

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




