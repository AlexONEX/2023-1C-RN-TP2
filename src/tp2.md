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
import requests

import random # Para poder setear una semilla global y obtener resultados reproducibles
random.seed(1221)
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

### Problema 1 - Descripciones en texto de compañias brasileñas 

El objetivo es poder usar los textos representados en el formato Bag of Words para poder clasificarlos como descripciones de una de las 9 compañias presentes en los datos.

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
training = (training- np.mean(training, axis = 0))/ np.var(training)
```
