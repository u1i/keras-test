# https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-2

import numpy as np
np.random.seed(123)  # for reproducibility

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

from keras.datasets import mnist
 
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print( X_train.shape )

from matplotlib import pyplot as plt
plt.imshow(X_train[0])


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)


X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print( X_train.shape )

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print( y_train.shape )

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

print( Y_train.shape )



model = load_model('bla.model')

score = model.evaluate(X_test, Y_test, verbose=0)

model.save("bla.model")
print(score)
