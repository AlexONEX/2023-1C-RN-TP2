import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cargar los datos del conjunto de datos
data = pd.read_csv('datos.csv', header=None)

# Obtener las frecuencias de palabras como matriz de entrada
X = data.iloc[:, 1:].values

# Obtener las etiquetas de categoría
y = data.iloc[:, 0].values

# Definir el número de dimensiones reducidas
n_components = 9

# Normalizar los datos manualmente
X_norm = X 
# Inicializar los pesos y las respuestas
weights_oja = np.random.randn(X_norm.shape[1], n_components)
responses_oja = np.zeros((X_norm.shape[0], n_components))

# Entrenamiento utilizando la regla de aprendizaje de Oja
for _ in range(100):  # Número de iteraciones de entrenamiento
    for i in range(X_norm.shape[0]):
        x = X_norm[i]
        response_oja = np.dot(x, weights_oja)
        delta_w_oja = 0.0001 * np.outer(x, response_oja)
        weights_oja += delta_w_oja
        responses_oja[i] = response_oja

# Graficar los resultados en R^3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Obtener las etiquetas de categoría
Y1 = responses_oja[:, 0]
Y2 = responses_oja[:, 1]
Y3 = responses_oja[:, 2]

Y4 = responses_oja[:, 3]
Y5 = responses_oja[:, 4]
Y6 = responses_oja[:, 5]

Y7 = responses_oja[:, 6]
Y8 = responses_oja[:, 7]
Y9 = responses_oja[:, 8]


# Graficar los resultados
ax.scatter(Y1, Y2, Y3, c=y, cmap='viridis', s=50)
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
ax.set_zlabel('Componente 3')
plt.title('Primeras 3 componentes principales')
plt.savefig('oja1-3.png')
plt.show()

# Graficar los resultados en R^3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y4, Y5, Y6, c=y, cmap='viridis', s=50)
ax.set_xlabel('Componente 4')
ax.set_ylabel('Componente 5')
ax.set_zlabel('Componente 6')
plt.title('Segundas 3 componentes principales')
plt.savefig('oja4-6.png')
plt.show()

# Graficar los resultados en R^3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y7, Y8, Y9, c=y, cmap='viridis', s=50)
ax.set_xlabel('Componente 7')
ax.set_ylabel('Componente 8')
ax.set_zlabel('Componente 9')
plt.title('Terceras 3 componentes principales')
plt.savefig('oja7-9.png')
plt.show()

# Graficar todas las componentes principales
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
