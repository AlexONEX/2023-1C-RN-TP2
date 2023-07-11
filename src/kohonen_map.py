import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests

r = requests.get('https://git.exactas.uba.ar/redes-neuronales/clases/-/raw/master/practicas/datos/tp2_training_dataset.csv')

data = pd.read_csv('data.csv', header=None)#np.loadtxt(r.iter_lines(), delimiter=',')

categories = data.iloc[:, 0]
word_vectors = data.iloc[:, 1:]

# Normalizar los vectores de palabras
data_normalized = word_vectors.div(word_vectors.sum(axis=1), axis=0)

# Número de características y número de categorías
N, M = data_normalized.shape[1], len(categories.unique())

# Parámetros de aprendizaje
epochs = 50
lr_st = 1.0
lr_dr = 0.1
ir_st = 3
ir_dr = 0.05
M = 3

np.random.seed(0)  # Para la reproducibilidad
w = np.random.normal(0, 1, (M, M, N))
y = np.zeros((M, M))  # Para no estar creando la matriz de winner takes all en cada iteración


for epoch in range(epochs):
    ## El Coeficiente de Aprendizaje y el Radio de Influencia decaen exponencialmente
    learning_rate = lr_st * np.exp(-epoch * lr_dr)
    influence_radius = ir_st * np.exp(-epoch * ir_dr)
    for i in range(len(data_normalized)):
        instance = data_normalized.iloc[i, :].values

        # Activación
        ## Para determinar la unidad ganadora se toma la distancia entre los pesos de las unidades de salida y la instancia de entrada.
        e = instance - w
        n = np.linalg.norm(e, axis=2)
        ## La posicion de la unidad ganadora es la que menor distancia tenga.
        p = np.unravel_index(np.argmin(n), n.shape)
        ## Winner-Takes-All
        y[:] = 0
        y[p] = 1

        # Aprendizaje
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
active_categories_counter = np.zeros((M, M, 10), dtype=int)

## Por cada instancia se fija cual es el elemento de la matriz W que mas se parece y se le asigna la categoría correspondiente
for i in range(len(data_normalized)):
    instance = data_normalized.iloc[i, :].values
    expanded_instance = np.expand_dims(instance, axis=(0, 1))
    e = expanded_instance - w
    n = np.linalg.norm(e, axis=2)
    p = np.unravel_index(np.argmin(n), n.shape)
    active_categories_counter[p][categories.iloc[i]] += 1

## Se asigna la categoría que mas se repite en cada unidad de salida
for i in range(len(active_categories_counter)):
    for j in range(len(active_categories_counter[i])):
        active_categories[i][j] = np.argmax(active_categories_counter[i][j])

## Se grafica el mapa de características
fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.matshow(active_categories, cmap='viridis')

ax.set_xticks(np.arange(M))
ax.set_yticks(np.arange(M))
ax.set_xlabel('Unidad de salida X')
ax.set_ylabel('Unidad de salida Y')

### Agregar números en cada cuadrado de la matriz
for i in range(M):
    for j in range(M):
        c = active_categories[i, j]
        ax.text(j, i, str(c), va='center', ha='center', color='white')

### Leyenda de colores
cmap = plt.cm.get_cmap('viridis', len(labels))
legend_elements = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(labels))]
ax.legend(legend_elements, labels, loc='upper right')

plt.title('Mapa de características')
plt.show()



##Grafico de puntos
pred_x = np.zeros((data_normalized.shape[0],))
pred_y = np.zeros((data_normalized.shape[0],))

for i in range(len(data_normalized)):
    instance = data_normalized.iloc[i, :].values
    expanded_instance = np.expand_dims(instance, axis=(0, 1))
    e = expanded_instance - w
    n = np.linalg.norm(e, axis=2)
    p = np.unravel_index(np.argmin(n), n.shape)
    pred_x[i] = p[0]
    pred_y[i] = M - p[1] - 1

jitter_var = 0.15


jitter_x = pred_x + np.random.normal(0, jitter_var, len(pred_x)) #ruido agregado para que se vean mejor los clusters
jitter_y = pred_y + np.random.normal(0, jitter_var, len(pred_y)) #ruido agregado para que se vean mejor los clusters

fig, ax = plt.subplots()
scatter = ax.scatter(jitter_x, jitter_y, c=categories, cmap="Set1")
ax.set_xlabel('Unidades de Salida')
ax.set_ylabel('Unidades de Salida')
ax.set_title('Clusters con ruido')

# Add colorbar
cbar = fig.colorbar(scatter)
cbar.set_label('Categories')

plt.show()