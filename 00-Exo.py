#!/bin/python

import numpy as np
X=np.c_[np.ones([150,2])]
print(X)
total_size=150
train_size = total_size * 0.2

x=np.random.permutation(total_size)

y=X[x[:train_size]]
print(x)
print(y)


