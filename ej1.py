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

matrices_Imagenes = np.array(matrices_Imagenes).T
print(matrices_Imagenes.shape)
#mostrar la matriz A con matplot
plt.imshow(matrices_Imagenes, cmap='gray')
plt.show()
# Imprimir el número de imágenes procesadas
print(f'Se importaron {len(matrices_Imagenes)} imágenes.')
# Obtener el tamaño de la matriz de una Imagen con el primer vector 
p = np.sqrt(matrices_Imagenes[:,0].shape[0])
print(p)





