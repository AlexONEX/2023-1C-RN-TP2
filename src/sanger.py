import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('datos.csv', header=0, index_col=0)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

n_components = 9

X_norm = X 

# Inicializar los pesos y las respuestas
weights_sanger = np.random.randn(X_norm.shape[1], n_components)
responses_sanger = np.zeros((X_norm.shape[0], n_components))

# Entrenamiento utilizando la regla de aprendizaje de Sanger
for _ in range(100):  # Número de iteraciones de entrenamiento
    for i in range(X_norm.shape[0]):
        x = X_norm[i]
        response_sanger = np.dot(x, weights_sanger)
        for j in range(n_components):
            delta_w_sanger = 0.000001 * response_sanger[j] * (x - np.dot(weights_sanger[:, :j], responses_sanger[i, :j]))
            weights_sanger[:, j] += delta_w_sanger
        responses_sanger[i] = response_sanger

#Print shape y content
print("Shape of responses_sanger: ", responses_sanger.shape)
print("Content of responses_sanger: ", responses_sanger)

# Obtener las componentes principales
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
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y1, Y2, Y3, c=y, cmap='viridis', s=50)
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
ax.set_zlabel('Componente 3')
plt.title('Primeras 3 componentes principales')
plt.savefig('sanger1-3.png')
plt.show()

# Graficar los resultados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y4, Y5, Y6, c=y, cmap='viridis', s=50)
ax.set_xlabel('Componente 4')
ax.set_ylabel('Componente 5')
ax.set_zlabel('Componente 6')
plt.title('Segundas 3 componentes principales')
plt.savefig('sanger4-6.png')
plt.show()

# Graficar los resultados
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y7, Y8, Y9, c=y, cmap='viridis', s=50)
ax.set_xlabel('Componente 7')
ax.set_ylabel('Componente 8')
ax.set_zlabel('Componente 9')
plt.title('Terceras 3 componentes principales')
plt.savefig('sanger7-9.png')
plt.show()
