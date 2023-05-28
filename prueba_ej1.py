from PIL import Image
import numpy as np
import os

directorio = '/home/san/Ingeniería UdeSA/Metodos Numericos y Optimizacion/Trabajos practicos/tp3/dataset_imagenes'

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

# Ejemplo de acceso a una matriz de imagen específica
# imagen_1 = matrices_imagenes[0]
# print(f'Dimensiones de la imagen 1: {imagen_1.shape}')

# Acceder a las matrices de todas las imágenes
# for i, matriz_imagen in enumerate(matrices_imagenes):
#     dimensiones = matriz_imagen.shape
#     print(f'Dimensiones de la imagen {i + 1}: {dimensiones}')
#     mat = np.array(matriz_imagen)
#     vector = mat.reshape(-1)

# Obtener el tamaño de la matriz de una imagen
m = matrices_imagenes[0].shape[0]

# Matriz que almacenará los vectores de cada imagen
matriz_columnas = np.empty((m * m, len(matrices_imagenes)))

# Recorrer las matrices de las imágenes
for i, matriz_imagen in enumerate(matrices_imagenes):
    vector = matriz_imagen.reshape(-1)  # Convertir la matriz en un vector
    matriz_columnas[:, i] = vector  # Agregar el vector como columna en la matriz

# Imprimir la matriz de columnas resultante
# print(matriz_columnas)
print(matriz_columnas.shape)