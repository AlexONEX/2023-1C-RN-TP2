import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Leer los datos del archivo CSV
data = pd.read_csv('data.csv', header=None)

# Normalizar los vectores de palabras
data_normalized = data 

# Categorías y vectores de palabras
categories = data_normalized.iloc[:, 0]
word_vectors = data_normalized.iloc[:, 1:]

# Número de características y número de categorías
N, M = word_vectors.shape[1], len(categories.unique())

# Inicialización de los pesos
np.random.seed(0)  # Para la reproducibilidad
W = np.random.normal(0, 0.1, (N, M))

# Parámetros de aprendizaje
epochs = 1000
learning_rate = 0.1  # Tasa de aprendizaje inicial
decay_rate = 0.001  # Tasa de decaimiento de la tasa de aprendizaje
orthogonality_threshold = 0.01  # Umbral de ortogonalidad para el criterio de parada

reconstruction_errors = []
# Aprendizaje con la regla de Sanger
for epoch in range(epochs):
    epoch_error = 0
    for i, x in word_vectors.iterrows():
        x = np.array(x).reshape(-1, 1)
        y = np.dot(W.T, x)
        y_ = np.tril(np.dot(y, y.T))
        delta_W = learning_rate * (np.dot(x, y.T) - np.dot(W, y_))
        W += delta_W
        epoch_error += np.linalg.norm(x - np.dot(W, y))**2
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

# Chequeo de la estabilidad de los resultados
# Podría hacerse reentrenando varias veces y verificando que los pesos convergen a los mismos valores
