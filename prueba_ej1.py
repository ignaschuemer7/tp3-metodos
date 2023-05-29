from PIL import Image
import numpy as np
import os
from pylab import *
"""
En el archivo dataset_imagenes.zip se encuentran varias imágenes. Cada imagen es una matriz de p × p que
puede representarse como un vector x ∈ Rp∗p. A su vez, es posible armar un matriz de datos apilando los
vectores de cada imagen. Se desea aprender una representación de baja dimensión de las imágenes mediante
una descomposición en valores singulares.
1. Visualizar en forma matricial p×p las primeras y las últimas dimensiones (autovectores) de la descomposición
obtenida. ¿Qué diferencias existen entre unas y otras? ¿Qué conclusiones pueden sacar?


"""
directorio = 'dataset_imagenes'

matrices_imagenes = []

for nombre_archivo in os.listdir(directorio):
    if nombre_archivo.endswith('.jpeg'):
        ruta_imagen = os.path.join(directorio, nombre_archivo)
        
        # Abrir la imagen con PIL
        imagen = Image.open(ruta_imagen)
        
        # Convertir la imagen a una matriz NumPy
        matriz_imagen = np.array(imagen)
        
        # Agregar la matriz de la imagen a la lista
        matrices_imagenes.append(matriz_imagen)

# Imprimir el número de imágenes procesadas
print(f'Se importaron {len(matrices_imagenes)} imágenes.')
# Obtener el tamaño de la matriz de una imagen
m = matrices_imagenes[0].shape[0]

# Matriz que almacenará los vectores de cada imagen
matriz_columnas = np.empty((m * m, len(matrices_imagenes)))

plt.imshow(matriz_columnas, cmap='gray')
plt.show()

# Recorrer las matrices de las imágenes
for i, matriz_imagen in enumerate(matrices_imagenes):
    vector = matriz_imagen.reshape(-1)  # Convertir la matriz en un vector
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
print("Primeras dimensiones (autovectores):")
print(first_dimension)

print("\nÚltimas dimensiones (autovectores):")
print(last_dimension)

# printear la imagen resultante
plt.imshow(first_dimension, cmap='gray')
plt.show()

plt.imshow(last_dimension, cmap='gray')
plt.show()

