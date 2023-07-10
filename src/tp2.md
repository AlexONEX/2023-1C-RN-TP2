---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# TP 1 - Redes Neuronales
Intregrantes: 
- Ivan Charabora LU: 234/20
- Alejandro Schwartzmann LU: 390/20
- Paula Pérez Bianchi LU: 7/20

+++

## Introducción + código de los métodos

```{code-cell} ipython3
#Si los imports de abajo no funciona descomentar la línea correspondiente
#!pip install numpy
#!pip install matplotlib
#!pip install requests
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as mpl, cm
from mpl_toolkits import mplot3d


import requests

import random # Para poder setear una semilla global y obtener resultados reproducibles
random.seed(1221)
```

```{code-cell} ipython3
def oja_generalizada(X, M, lr, epochs=100, decay_rate= 1e-7):
    
    N = X.shape[1]
    W = np.random.normal(size = (N, M))
    Y = np.dot( X, W)
    orthogonality_values = np.zeros(epochs)
    for epoch in range(epochs):  
        for x in X: 
            deltaW = np.zeros(W.shape)
            x = x.reshape(1, N)
            Y = x@W
            Z = np.dot( Y, W.T)
            dif= x-Z
            dW = np.outer(dif, Y)
            W += lr*dW

        lr *= (1.0 / (1.0 + decay_rate * epoch))
        orthogonality_values[epoch] = np.sum(np.abs(np.dot(W.T, W) - np.identity(M))) / 2

    return W, orthogonality_values

def sanger(X, M, lr, epochs=100, decay_rate= 1e-7):
    
    N = X.shape[1]
    W = np.random.normal(size = (N, M))
    Y = np.dot( X, W)
    orthogonality_values = np.zeros(epochs)
    for epoch in range(epochs):  
        for x in X: 
            deltaW = np.zeros(W.shape)
            x = x.reshape(1, N)
            Y = x@W
            D = np.triu( np.ones((M,M)))
            Z = np.dot( W, Y.T*D)
            
            dW = (x.T - Z) * Y
            W += lr*dW

        lr *= (1.0 / (1.0 + decay_rate * epoch))
        orthogonality_values[epoch] = np.sum(np.abs(np.dot(W.T, W) - np.identity(M))) / 2

    return W, orthogonality_values
```



```{code-cell} ipython3
def kohonen_map(X, M, lr,ir,  epochs=100, decay_rate= 1e-7):
    N = X.shape[1]
    w = np.random.normal( 0,1, (M,M,N))
    y = np.zeros((M, M))
    for epoch in range(epochs):
        for x_i in X:
            # Activación
            ## Para determinar la unidad ganadora se toma la distancia entre los pesos de las unidades de salida y la instancia de entrada.
            e = x_i - w 
            n = np.linalg.norm(e, axis=2)
            ## La posicion de la unidad ganadora es la que menor distancia tenga.
            p = np.unravel_index(np.argmin(n), n.shape)
            ## Winner-Takes-All
            y[:]=0
            y[p] = 1
            
            # Aprendizaje
            ## El Coeficiente de Aprendizaje y el Radio de Influencia decaen exponencialmente
            learning_rate = lr* np.exp(-epoch * lr)
            influence_radius = ir * np.exp(-epoch * ir)
            ## Mapa de coordenadas de las unidades de salida que nos servirá como auxiliar en los cálculos
            il = [[(x, y) for y in range(M)] for x in range(M)]
            im = np.array(il)
            ## Distancia de la unidad ganadora a las otras unidades dentro de la grilla de salida
            d = np.linalg.norm(im - p, axis=2)
            ## Actualización de los pesos
            pf = np.exp(-d / (2 * np.square(influence_radius))).reshape((M, M, 1))
            dw = learning_rate * pf * e
            w += dw
    return w
    
```

### Visualización de las componentes de a 3

```{code-cell} ipython3
def plot_components_in_3d(Y, training_idx, data_with_labels, method_name):

    fig, ax = plt.subplots(1, 3, figsize = (15, 15), subplot_kw={'projection': '3d'})
    c = np.round(data_with_labels[training_idx,0])
    ax[0].scatter(Y[:, 0], Y[:, 1], Y[:, 2], c = c, cmap='Dark2', s=50)
    ax[0].set_xlabel('Componente 1')
    ax[0].set_ylabel('Componente 2')
    ax[0].set_zlabel('Componente 3')
    ax[0].set_title('Primeras 3 componentes principales')

    #ax2 = fig.add_subplot(321, projection='3d')
    c = np.round(data_with_labels[training_idx,0])
    ax[1].scatter(Y[:, 3], Y[:, 4], Y[:, 5], c = c, cmap='Dark2', s=50)
    ax[1].set_xlabel('Componente 4')
    ax[1].set_ylabel('Componente 5')
    ax[1].set_zlabel('Componente 6')
    ax[1].set_title('Segundas 3 componentes principales')

    #ax3 = fig.add_subplot(331, projection='3d')
    c = np.round(data_with_labels[training_idx,0])
    ax[2].scatter(Y[:, 6], Y[:, 7], Y[:, 8], c = c, cmap='Dark2', s=50)
    ax[2].set_xlabel('Componente 7')
    ax[2].set_ylabel('Componente 8')
    ax[2].set_zlabel('Componente 9')
    ax[2].set_title('Terceras 3 componentes principales')

    plt.show()

    #plt.savefig("") ---> Para guardar las imagenes
```

### Cross Validation
Función para separación de folds mezclando los datos  

```{code-cell} ipython3
def split_folds(number_of_folds, x_size):
    x_idx = np.arange(x_size)
    np.random.shuffle(x_idx)
    idx_splits = []
    fold_size = (x_size//number_of_folds)
    
    for idx_fold in range(number_of_folds):
       if idx_fold == number_of_folds-1:
          test = x_idx[idx_fold*fold_size:]
       else:
          test = x_idx[idx_fold*fold_size: idx_fold*fold_size + fold_size]
       
       training = x_idx[~np.isin(x_idx, test)]
       idx_splits.append((training,  test))
    
    return idx_splits
```

# Experimentación

+++

### Problema - Descripciones en texto de compañias brasileñas 

El objetivo es poder usar los textos representados en el formato Bag of Words para poder clasificar las empresas segun su actividad principal. Por conocimiento previo sabemos que las empresas pueden clasificarse en 9 categorias por su actividad principal. Nos interesa ver si podemos a través de los distintos métodos de aprendizaje no supervisado separar los datos de forma que se diferencien estas 9 clases. 

```{code-cell} ipython3
r = requests.get("https://git.exactas.uba.ar/redes-neuronales/clases/-/raw/master/practicas/datos/tp2_training_dataset.csv")
data_with_labels = np.loadtxt(r.iter_lines(), delimiter=",")

data = data_with_labels[::,1:]
```

#### Separamos el training set 

```{code-cell} ipython3
training_data_len = 800
training_idx = np.random.choice(899, training_data_len, replace =False)

training = data[training_idx]
evaluation = data[~np.isin(np.arange(900),training_idx)] #hand-out set 

print("training data shape ", training.shape)
print("eval data shape ", evaluation.shape)

#Normalizamos 
training = (training- np.mean(training, axis = 0))/ np.std(training)
```

### 2.1 Reducción de dimensiones 
Utilizamos los métodos de Oja y Sager para reducir la dimensionalidad de los datos manteniendo la varianza de los mismos. Por la representación que usamos de las descripciones nuestros datos viven en dimensión 856 pero como se puede ver las instancias se anulan en muchas de estas dimensiones por lo que es interesante encontrar una representación de los datos en la dimensión real de los mismos. En este caso como ya sabemos que hay 9 clases queremos una representación con esa cantidad de variables. 

```{code-cell} ipython3
W_oja, o_oja  = oja_generalizada(training, 9, 0.000001, 2000)
Y_oja = training@W_oja
plot_components_in_3d(Y_oja, training_idx, data_with_labels, "Regla de Oja Generalizada")
```

```{code-cell} ipython3
W_sanger, o_sanger = sanger(training, 9, 0.000001, 1000)
Y_sanger = training@W_sanger
plot_components_in_3d(Y_sanger, training_idx, data_with_labels, "Regla de Sanger")
```

### 2.2 Mapeo de características

```{code-cell} ipython3
W_kohonen = kohonen_map(training, 9, 1e-5, 1e-4, 1000)
```

```{code-cell} ipython3
labels = np.arange(1, 10)
M = 9
active_categories = np.zeros((M, M), dtype=int)

## Por cada instancia se fija cual es el elemento de la matriz W que mas se parece y se le asigna la categoría correspondiente
for i in range(len(training)):
    instance = training[i]
    expanded_instance = np.expand_dims(instance, axis=(0, 1))
    e = expanded_instance - W_kohonen
    n = np.linalg.norm(e, axis=2)
    p = np.unravel_index(np.argmin(n), n.shape)
    active_categories[p] = data_with_labels[:,0][i]

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
```
