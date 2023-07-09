import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data.csv', header=None)

categories = data.iloc[:, 0]
word_vectors = data.iloc[:, 1:]

# Normalizar los vectores de palabras
data_normalized = word_vectors.div(word_vectors.sum(axis=1), axis=0)

# Número de características y número de categorías
N, M = data_normalized.shape[1], len(categories.unique())

# Parámetros de aprendizaje
epochs = 100
lr_st = 1.0
lr_dr = 0.1
ir_st = 3
ir_dr = 0.05
M=4

np.random.seed(0)  # Para la reproducibilidad
w = np.random.normal( 0,1, (M,M,N))
y = np.zeros((M, M)) #Para no estar creando la matriz de winner takes all en cada iteración

for epoch in range(epochs):
    for i in range(len(data_normalized)):
        instance = data_normalized.iloc[i, :].values

        # Activación
        ## Para determinar la unidad ganadora se toma la distancia entre los pesos de las unidades de salida y la instancia de entrada.
        e = instance - w 
        n = np.linalg.norm(e, axis=2)
        ## La posicion de la unidad ganadora es la que menor distancia tenga.
        p = np.unravel_index(np.argmin(n), n.shape)
        ## Winner-Takes-All
        y[:]=0
        y[p] = 1
        
        # Aprendizaje
        ## El Coeficiente de Aprendizaje y el Radio de Influencia decaen exponencialmente
        learning_rate = lr_st * np.exp(-epoch * lr_dr)
        influence_radius = ir_st * np.exp(-epoch * ir_dr)
        ## Mapa de coordenadas de las unidades de salida que nos servirá como auxiliar en los cálculos
        il = [[(x, y) for y in range(M)] for x in range(M)]
        im = np.array(il)
        ## Distancia de la unidad ganadora a las otras unidades dentro de la grilla de salida
        d = np.linalg.norm(im - p, axis=2)
        ## Actualización de los pesos
        pf = np.exp(-d / (2 * np.square(influence_radius))).reshape((M, M, 1))
        dw = learning_rate * pf * e
        w += dw

# Visualización de los pesos
labels = sorted(categories.unique())
active_categories = np.zeros((M, M), dtype=int)
## Por cada instancia se fija cual es el elemento de la matriz W que mas se parece y se le asigna la categoría correspondiente
for i in range(len(data_normalized)):
    instance = data_normalized.iloc[i, :].values
    expanded_instance = np.expand_dims(instance, axis=(0, 1))
    e = expanded_instance - w
    n = np.linalg.norm(e, axis=2)
    p = np.unravel_index(np.argmin(n), n.shape)
    active_categories[p] = categories.iloc[i]

## Se grafica el mapa de características
fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.matshow(active_categories, cmap='viridis')

ax.set_xticks(np.arange(M))
ax.set_yticks(np.arange(M))
ax.set_xlabel('Unidad de salida X')
ax.set_ylabel('Unidad de salida Y')

### Leyenda de colores
cmap = plt.cm.get_cmap('viridis', len(labels))
legend_elements = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(labels))]
ax.legend(legend_elements, labels, loc='upper right')

plt.title('Mapa de características')
plt.show()
