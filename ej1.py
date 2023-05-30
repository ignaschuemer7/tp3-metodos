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
p = np.sqrt(matrices_Imagenes[:,0].shape[0]).astype(int)

# Calcular la descomposición en valores singulares
U, S, Vt = np.linalg.svd(matrices_Imagenes, full_matrices=True)

# Visualizar en forma matricial p×p las primeras y las últimas dimensiones (autovectores) de la descomposición
# obtenida. ¿Qué diferencias existen entre unas y otras? ¿Qué conclusiones pueden sacar?
#tomamos k autovectores
k = 10
#primeros 10 autovectores
U_10 = U[:,:k]
Vt_10 = Vt[:k,:]
#ultimos 10 autovectores
U_ultimos_10 = U[:,-k:]
Vt_ultimos_10 = Vt[-k:,:]

#reconstruir la matriz de datos con los primeros 10 autovectores
matriz_Imagenes_10 = U_10 @ np.diag(S[:k]) @ Vt_10
#reconstruir la matriz de datos con los ultimos 10 autovectores
matriz_Imagenes_ultimos_10 = U_ultimos_10 @ np.diag(S[-k:]) @ Vt_ultimos_10

#mostrar la matriz A con matplot
fig, axs = plt.subplots(1,2)
axs[0].imshow(matriz_Imagenes_10, cmap='gray')
axs[0].set_title(f'primeros {str(k)} autovectores')
axs[1].imshow(matriz_Imagenes_ultimos_10, cmap='gray')
axs[1].set_title(f'ultimos {str(k)} autovectores')
plt.show()


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





