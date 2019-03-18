#!/bin/python3

#implementer un descente de gradient ordinaire avec early stopping
# qui dit early stopping: data validation + data test
# pour la regression logistique avec softmax (multinomial)

import numpy as np
from sklearn import datasets


iris=datasets.load_iris()
X=iris["data"][:,2:4] # petal lenght & width
y=iris["target"]     # prendre toutes les valeurs multinomial

# Add ones to X
X_with_bias=np.c_[np.ones((len(X),1)),X]

#Added:
np.random.seed(2042)

test_ratio=0.2
validation_ratio=0.2
total_size=len(X_with_bias)

# Size of each chunck: 
test_size= int(test_ratio * total_size)
validation_size= int(validation_ratio * total_size)
train_size=total_size - test_size - validation_size

# choix d'indice:

rnd_indices= np.random.permutation(total_size)

X_train=X_with_bias[rnd_indices[:train_size]]
