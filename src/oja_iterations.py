import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



nombre_archivo = "anotaciones_oja_iterations.txt"
# Definir los parámetros a probar
epochs_list = [10, 20, 30]  # Lista de epochs a probar
learning_rate_list = [0.1, 0.5, 1.0]  # Lista de tasas de aprendizaje a probar
decay_rate_list = [0.00001, 0.0001, 0.001]  # Lista de tasas de decaimiento a probar



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

orthogonality_threshold = 0.01  # Umbral de ortogonalidad para el criterio de parada

best_model_orthogonality = None
best_model_error = None
best_error = float('inf')
best_orthogonality = float('inf')


responses_oja = np.zeros((word_vectors.shape[0], M))
reconstruction_errors = []
for epochs in epochs_list:
    for learning_rate in learning_rate_list:
        for decay_rate in decay_rate_list:
            # Inicialización de los pesos
            np.random.seed(0)  # Para la reproducibilidad
            W = np.random.normal(0, 0.1, (N, M))

            # Aprendizaje con la regla de Oja
            reconstruction_errors = []
            # Aprendizaje con la regla de Oja
            for epoch in range(epochs):
                epoch_error = 0
                for i, x in word_vectors.iterrows():
                    x = np.array(x).reshape(-1, 1)
                    y = np.dot(W.T, x)
                    z = np.dot(W, y)
                    responses_oja[i] = y.reshape(-1)
                    delta_W = learning_rate * (np.dot((x - z), y.T))
                    W += delta_W
                    epoch_error += np.linalg.norm(x - z)**2
                reconstruction_errors.append(epoch_error / word_vectors.shape[0])
                # Decaimiento de la tasa de aprendizaje
                learning_rate *= (1.0 / (1.0 + decay_rate * epoch))
                
                # Cálculo de la ortogonalidad
                o = np.sum(np.abs(np.dot(W.T, W) - np.identity(M))) / 2
                
                # Criterio de parada basado en la ortogonalidad
                if o < orthogonality_threshold:
                    print("Criterio de parada alcanzado, terminando el aprendizaje.")
                    break
            
            with open(nombre_archivo, "w") as archivo:
                archivo.write("El modelo con los parametros: epochs = {}, learning_rate = {}, decay_rate = {} tiene un error de reconstruccion de: {} y ortogonalidad de: {}".format(epochs, learning_rate, decay_rate, reconstruction_errors[-1], o))
            
            print("El modelo con los parámetros: epochs = {}, learning_rate = {}, decay_rate = {} tiene un error de reconstrucción de: {} y ortogonalidad de: {}".format(epochs, learning_rate, decay_rate, reconstruction_errors[-1], o))
            if reconstruction_errors[-1] < best_error :
                best_model_error = (epochs, learning_rate, decay_rate)
                best_error = reconstruction_errors[-1]
            if o < best_orthogonality:
                best_model_orthogonality = (epochs, learning_rate, decay_rate)
                best_error = reconstruction_errors[-1]




print(f"El mejor modelo para la ortagonalidad es: {best_model_orthogonality} con Error: {best_error} y Ortogonalidad: {best_orthogonality}")
print(f"El mejor modelo para el error es: {best_model_error} con Error: {best_error} y Ortogonalidad: {best_orthogonality}")
