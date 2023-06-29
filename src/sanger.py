import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Leer los datos del archivo CSV
data = pd.read_csv('data.csv', header=None)

# Normalizar los vectores de palabras
#data_normalized = data
data_normalized = data.div(data.sum(axis=1), axis=0)

# Categorías y vectores de palabras
categories = data_normalized.iloc[:, 0]
word_vectors = data_normalized.iloc[:, 1:]

# Número de características y número de categorías
N, M = word_vectors.shape[1], len(categories.unique())

# Inicialización de los pesos
np.random.seed(0)  # Para la reproducibilidad
W = np.random.normal(0, 0.1, (N, M))

# Parámetros de aprendizaje
epochs = 10
learning_rate = 0.5  # Tasa de aprendizaje inicial
decay_rate = 0.00001  # Tasa de decaimiento de la tasa de aprendizaje
orthogonality_threshold = 0.01  # Umbral de ortogonalidad para el criterio de parada

responses_sanger = np.zeros((word_vectors.shape[0], M))
reconstruction_errors = []
# Aprendizaje con la regla de Sanger
for epoch in range(epochs):
    epoch_error = 0
    for i, x in word_vectors.iterrows():
        x = np.array(x).reshape(-1, 1)
        y = np.dot(W.T, x)
        responses_sanger[i] = y.reshape(-1)
        z = np.zeros(x.shape)
        for j in range(M):
            z += y[j] * W[:, [j]]
        for j in range(M):
            delta_W_j = learning_rate * (y[j] * x - y[j] * z)
            W[:, [j]] += delta_W_j
        epoch_error += np.linalg.norm(x - z)**2
    reconstruction_errors.append(epoch_error / word_vectors.shape[0])
    # Decaimiento de la tasa de aprendizaje
    learning_rate *= (1.0 / (1.0 + decay_rate * epoch))
    
    # Cálculo de la ortogonalidad
    o = np.sum(np.abs(np.dot(W.T, W) - np.identity(M))) / 2
    print(f"Ortogonalidad en la época {epoch+1}: {o}")
    
    # Criterio de parada basado en la ortogonalidad
    if o < orthogonality_threshold:
        print("Criterio de parada alcanzado, terminando el aprendizaje.")
        break

# Visualización de los pesos
plt.figure(figsize=(10, 8))
sns.heatmap(W, annot=True, cmap='coolwarm')
plt.title('Visualización de los pesos')
plt.show()

# Visualización del error de reconstrucción
plt.figure(figsize=(10, 8))
plt.plot(reconstruction_errors)
plt.title('Error de reconstrucción a lo largo del tiempo')
plt.xlabel('Época')
plt.ylabel('Error de reconstrucción')
plt.show()

# Graficar los resultados en R^3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Obtener las etiquetas de categoría
Y1 = responses_sanger[:, 0]
Y2 = responses_sanger[:, 1]
Y3 = responses_sanger[:, 2]

Y4 = responses_sanger[:, 3]
Y5 = responses_sanger[:, 4]
Y6 = responses_sanger[:, 5]

Y7 = responses_sanger[:, 6]
Y8 = responses_sanger[:, 7]
Y9 = responses_sanger[:, 8]

# Graficar los resultados
ax.scatter(Y1, Y2, Y3, c=y, cmap='viridis', s=50)
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
ax.set_zlabel('Componente 3')
plt.title('Primeras 3 componentes principales')
plt.savefig('sanger1-3.png')
plt.show()

# Graficar los resultados en R^3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y4, Y5, Y6, c=y, cmap='viridis', s=50)
ax.set_xlabel('Componente 4')
ax.set_ylabel('Componente 5')
ax.set_zlabel('Componente 6')
plt.title('Segundas 3 componentes principales')
plt.savefig('sanger4-6.png')
plt.show()

# Graficar los resultados en R^3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y7, Y8, Y9, c=y, cmap='viridis', s=50)
ax.set_xlabel('Componente 7')
ax.set_ylabel('Componente 8')
ax.set_zlabel('Componente 9')
plt.title('Terceras 3 componentes principales')
plt.savefig('sanger7-9.png')
plt.show()
