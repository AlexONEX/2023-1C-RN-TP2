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

## Introducción
En este trabajo vamos a utilizar métodos de reduccion de la dimensión y clustering basados en redes neuronales. Primero presentamos nuestras implementaciones de cada uno de los métodos y luego nuestra experimentación y resultados con los datos propuestos. 

## Implementaciones de los métodos

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

### Métodos de entranamiento no supervisado con el método de Oja y Sanger

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

#### Método para calcular el mapa de kohonen

```{code-cell} ipython3
def kohonen_map(X, M, lr_st = 1.0, lr_dr =0.1 , ir_st =3, ir_dr = 0.05,  epochs=1000):
    N = X.shape[1]
    w = np.random.normal( 0,1, (M,M,N))
    y = np.zeros((M, M))
    for epoch in range(epochs):
         ## El Coeficiente de Aprendizaje y el Radio de Influencia decaen exponencialmente
        learning_rate = lr_st * np.exp(-epoch * lr_dr)
        influence_radius = ir_st * np.exp(-epoch * ir_dr)
        
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
            ## Mapa de coordenadas de las unidades de salida que nos servirá como auxiliar en los cálculos
            il = [[(x, y) for y in range(M)] for x in range(M)]
            im = np.array(il)
            ## Distancia de la unidad ganadora a las otras unidades dentro de la grilla de salida
            d = np.linalg.norm(im - p, axis=2)
            ## Actualización de los pesos
            pf = np.exp(-d / (2 * np.square(influence_radius))).reshape((M, M, 1))
            dw = learning_rate * pf * e
            w += dw
    return w, y
    
```

### Visualización de las componentes de a 3

```{code-cell} ipython3
def plot_components_in_3d(Y, training_idx, data_with_labels, method_name):

    fig, ax = plt.subplots(1, 3, figsize = (15, 15), subplot_kw={'projection': '3d'})
    c = np.round(data_with_labels[training_idx,0])
    ax[0].scatter(Y[:, 0], Y[:, 1], Y[:, 2], c = c, cmap='tab10', s=50)
    ax[0].set_xlabel('Componente 1')
    ax[0].set_ylabel('Componente 2')
    ax[0].set_zlabel('Componente 3')
    ax[0].set_title('Primeras 3 componentes principales')

    ax[1].scatter(Y[:, 3], Y[:, 4], Y[:, 5], c = c, cmap='tab10', s=50)
    ax[1].set_xlabel('Componente 4')
    ax[1].set_ylabel('Componente 5')
    ax[1].set_zlabel('Componente 6')
    ax[1].set_title('Segundas 3 componentes principales')

    ax[2].scatter(Y[:, 6], Y[:, 7], Y[:, 8], c = c, cmap='tab10', s=50)
    ax[2].set_xlabel('Componente 7')
    ax[2].set_ylabel('Componente 8')
    ax[2].set_zlabel('Componente 9')
    ax[2].set_title('Terceras 3 componentes principales')

    plt.show()

    #plt.savefig("") ---> Para guardar las imagenes
```

#### Plot del mapa de categorias

```{code-cell} ipython3
def plot_category_maps(data, W, data_labels):

    labels = np.arange(0, 9)
    M = 9
    active_categories = np.zeros((M, M), dtype=int)
    active_categories_counter = np.zeros((M, M, 10), dtype=int)

    ## Por cada instancia se fija cual es el elemento de la matriz W que mas se parece y se le asigna la categoría correspondiente
    for i in range(len(data)):
        instance = data[i]
        instance = np.expand_dims(instance, axis=(0, 1))
        e = instance - W
        n = np.linalg.norm(e, axis=2)
        p = np.unravel_index(np.argmin(n), n.shape)
        active_categories_counter[p][int(data_labels[i])] += 1
    
    ## Se asigna la categoría que mas se repite en cada unidad de salida
    for i in range(len(active_categories_counter)):
        for j in range(len(active_categories_counter[i])):
            active_categories[i][j] = np.argmax(active_categories_counter[i][j])


    ## Se grafica el mapa de características
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(active_categories, cmap='tab10')

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
    cmap = plt.cm.get_cmap('tab10', len(labels))
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(labels))]
    ax.legend(legend_elements, labels+1, loc='upper right')

    plt.title('Mapa de características')
    plt.show()
```

```{code-cell} ipython3
def plot_category_points(data, W, data_labels):
    categories = np.arange(1, 10)
    M = 9
    ##Grafico de puntos
    pred_x = np.zeros((data.shape[0],))
    pred_y = np.zeros((data.shape[0],))

    for i in range(len(data)):
        instance = data[i]
        expanded_instance = np.expand_dims(instance, axis=(0, 1))
        e = expanded_instance - W
        n = np.linalg.norm(e, axis=2)
        p = np.unravel_index(np.argmin(n), n.shape)
        pred_x[i] = p[0]
        pred_y[i] = M - p[1] - 1

    jitter_var = 0.15
    jitter_x = pred_x + np.random.normal(0, jitter_var, len(pred_x)) #ruido agregado para que se vean mejor los clusters
    jitter_y = pred_y + np.random.normal(0, jitter_var, len(pred_y)) #ruido agregado para que se vean mejor los clusters

    fig, ax = plt.subplots()
    scatter = ax.scatter(jitter_x, jitter_y, c=data_labels, cmap="Set1")
    ax.set_xlabel('Unidades de Salida')
    ax.set_ylabel('Unidades de Salida')
    ax.set_title('Clusters con ruido')

    # Add colorbar
    cbar = fig.colorbar(scatter)
    cbar.set_label('Categories')

    plt.show()
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
data_norm = (data- np.mean(data, axis = 0))/ np.std(data)
training = (training- np.mean(training, axis = 0))/ np.std(training)
evaluation = (evaluation- np.mean(training, axis = 0))/ np.std(training)
#training = np.divide(training.T,training.sum(axis = 1)).T
```

### 2.1 Reducción de dimensiones 
Utilizamos los métodos de Oja y Sager para reducir la dimensionalidad de los datos manteniendo la varianza de los mismos. Por la representación que usamos de las descripciones nuestros datos viven en dimensión 856 pero como se puede ver las instancias se anulan en muchas de estas dimensiones por lo que es interesante encontrar una representación de los datos en la dimensión real de los mismos. En este caso como ya sabemos que hay 9 clases queremos una representación con esa cantidad de variables. 

```{code-cell} ipython3
W_oja, o_oja = oja_generalizada(data_norm, 9, 1e-05, 5000, 1e-06) #Mejor modelo: tarda 4 minutos aproximadamente
#W_oja, o_oja = oja_generalizada(data_norm, 9, 1e-05, 1000, 1e-05) #Buen modelo: tarda poco más de 1 minuto
#W_oja, o_oja = oja_generalizada(data_norm, 9, 1e-05, 500, 1e-04) #Buen modelo: tarda aproximadamente 20 segundos
Y_oja = data_norm@W_oja
plot_components_in_3d(Y_oja, np.arange(len(data_norm)), data_with_labels, "Regla de Oja Generalizada")
```

```{code-cell} ipython3
W_sanger, o_sanger = sanger(data_norm, 9, 1e-05, 5000, 1e-06) #Mejor modelo: tarda 5 minutos aproximadamente
# W_sanger, o_sanger = sanger(data_norm, 9, 1e-04, 1000, 1e-05) #Buen modelo: tarda 2 minutos aproximadamente
# W_sanger, o_sanger = sanger(data_norm, 9, 1e-04, 500, 1e-04) #Buen modelo: tarda 40 segundos aproximadamente

Y_sanger = data_norm@W_sanger
plot_components_in_3d(Y_sanger, np.arange(len(data_norm)), data_with_labels, "Regla de Sanger")
```

Las siguientes secciones de código se usaron para encontrar el mejor modelo con la regla de Oja y Sanger. 

La idea fue probar con distintos valores de:
* Épocas: se incrementaron gradualmente las iteraciones para evaluar cómo afectaba al rendimiento del modelo. Se observó si había mejoras significativas o si se alcanzaba un punto de convergencia.
* Learning rates iniciales: con cuanto empieza aprendiendo
* Decay rate: que tan rápido bajamos el learning rate

El objetivo principal es encontrar la combinación óptima para obtener la menor ortogonalidad posible.

```{code-cell} ipython3
epochs_list = [10, 50, 100, 500, 1000, 2500, 5000] 
lr_list = [0.00001, 0.0001] 
decay_rate_list = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]  
```

```{code-cell} ipython3
for epochs in epochs_list:
    for lr in lr_list:
        for decay_rate in decay_rate_list:
            print(decay_rate, lr, epochs, oja_generalizada(training, 9, lr, epochs, decay_rate)[1][-1])
```

Los 10 modelos con la menor ortogonalidad con la regla de Oja generalizada son los siguientes:

| Epocas | Learning Rate | Decay Rate | Ortogonalidad    |
| ------ | ------------- | ---------- | ---------------- |
| 5000   | 1e-05         | 1e-06      | 0.000223         |
| 5000   | 1e-04         | 1e-06      | 0.000278         |
| 5000   | 1e-05         | 1e-05      | 0.000627         |
| 2500   | 1e-05         | 1e-05      | 0.000634         |
| 1000   | 1e-05         | 1e-05      | 0.000765         |
| 2500   | 1e-04         | 1e-05      | 0.000833         |
| 5000   | 1e-04         | 1e-05      | 0.000843         |
| 2500   | 1e-05         | 1e-06      | 0.001111         |
| 2500   | 1e-05         | 1e-04      | 0.001560         |
| 1000   | 1e-05         | 1e-04      | 0.001720         |
| 500    | 1e-05         | 1e-04      | 0.001753         |


Al analizar los resultados, podemos observar que los modelos con un mayor número de épocas de entrenamiento, como 5000, tienden a tener menor ortogonalidad. Esto indica que a medida que aumentamos la cantidad de épocas, el modelo se adapta mejor a los datos. Lo que si se puede tener en cuenta es tener un balance respecto tiempo y cuanto más mejora aumentar las épocas. Podemos ver que con 1000 épocas se puede tener un buen módelo bajando mucho el tiempo. Para tener un ejemplo a consideración, el primer módelo nos tardó 3 minutos con 45 segundos mientras el primero de 1000 épocas nos tardó 1 minuto con 15 segundos. Incluso, el de 500 épocas consideramos que sigue siendo un buen modelo y no tarda más de medio minuto.

En cuanto a los valores de learning rate y decay rate, encontramos una variedad de combinaciones que han obtenido buenos resultados en términos de ortogonalidad así que no vemos correlación tan directa con la ortogonalidad. Igualmente, podemos notar, que no aparece ninguno de los 2 valores mayores a 0.0001 testeados en el decay_rate ni el menor valor (1e-07).

```{code-cell} ipython3
for epochs in epochs_list:
    for lr in lr_list:
        for decay_rate in decay_rate_list:
            print(decay_rate, lr, epochs, sanger(training, 9, lr, epochs, decay_rate)[1][-1])
```

Los 10 modelos con la menor ortogonalidad con la regla de Sanger son los siguientes:

| Epocas | Learning Rate | Decay Rate | Ortogonalidad |
| ------ | ------------- | ---------- | ------------- |
| 5000   | 1e-05         | 1e-06      | 0.001065      |
| 5000   | 1e-04         | 1e-06      | 0.001331      |
| 2500   | 1e-04         | 1e-05      | 0.003786      |
| 5000   | 1e-04         | 1e-05      | 0.003786      |
| 2500   | 1e-05         | 1e-05      | 0.004773      |
| 5000   | 1e-05         | 1e-05      | 0.004886      |
| 2500   | 1e-05         | 1e-06      | 0.005027      |
| 1000   | 1e-04         | 1e-05      | 0.009478      |
| 5000   | 1e-04         | 1e-04      | 0.010467      |
| 2500   | 1e-04         | 1e-04      | 0.010467      |
| 500    | 1e-04         | 1e-04      | 0.010468      |

Podemos ver que es un caso parecido al de la Oja generalizada. En primer lugar, nuevamente, podemos notar que el número de épocas de entrenamiento parece tener una influencia significativa en la ortogonalidad alcanzada por los modelos. A medida que aumentamos el número de épocas, observamos una disminución en la ortogonalidad. Esto indica que a medida que el modelo se entrena durante más tiempo, logra una mejor adaptación a los datos. En este caso el primer modelo tomó 5 minutos en ejecutarse mientras que el de 1000 épocas tomó poco menos de 2 minutos

Luego, se ve el mismo resultado respecto learning rate y el decay rate.

+++

Al comparar los resultados de los dos conjuntos de parámetros de prueba, podemos concluir que, en esta configuración específica, la regla de la Oja produjo mejores resultados en términos de ortogonalidad. Además, también se observó que la implementación de la regla de la Oja tuvo un tiempo de ejecución más rápido en cada una de las instancias. Estas observaciones sugieren que la regla de la Oja es una opción más eficiente para este escenario de entrenamiento. 

+++

### 2.2 Mapeo de características

En este punto utilizamos la técnica de mapas de Kohonen para encontrar clusters y poder clasificar datos nuevos de empresas por su actividad principal. Esta técnica también es conocida como SOM. 

Primero entrenamos la red con los siguientes parametros: 

```{code-cell} ipython3
W_kohonen,y_kohonen = kohonen_map(training, 9, lr_st= 1, lr_dr=0.1, ir_st =3, ir_dr=0.05, epochs= 1000)
```

Ahora para poder ver cuales son las neuronas que se activan más con los datos de cada clase podemos ver el mapa de características. Primero veamos esto para los datos del training set

```{code-cell} ipython3
plot_category_maps(training, W_kohonen, data_with_labels[training_idx , 0])
```

De forma equivalente podemos ver para cada instancia que neurona es la que más se activa y a que clase pertenece. Notar que los colores no quedaron iguales con respecto al gráfico anterior. Este gráfico nos sirve para ver que los clusters más marcados en el mapa de categorias son tambien los que sus instancias se encuentran menos mezcladas. Probablemente estas sean las clases que son más fáciles de separar o las que resultaron mejores para estos parametros. Otro punto interesante que podemos ver en este gráfico es que hay algunas regiones que tienen muchas intancias de distintos colores y probablemente corresponden con las fronteras entre los clusters. 

```{code-cell} ipython3
plot_category_points(training,  W_kohonen, data_with_labels[training_idx , 0])
```

Finalmente veamos si esto nos sirve para clasificar datos nuevos no vistos por el modelo. 

```{code-cell} ipython3
plot_category_maps(evaluation, W_kohonen, data_with_labels[~np.isin(np.arange(900),training_idx), 0])
```

```{code-cell} ipython3
plot_category_points(evaluation, W_kohonen, data_with_labels[~np.isin(np.arange(900),training_idx), 0])
```

Como se puede ver la arquitectura planteada no generaliza bien para instancias no vistas durante el entrenamiento. Probablemente esto corresponde con que ajustamos tanto los parametros del modelo que terminamos overfitteando demasiado al training set. Para solucionar esto se podrían utilizar medidas para evaluar los clusters y así poder usar CV y evaluar muchos más modelos sin tener que ver uno por uno los mapas de características de cada arquitectura.
