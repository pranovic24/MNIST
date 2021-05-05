# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:20:27 2019

@author: janik
"""

from mlxtend.data import loadlocal_mnist
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, AveragePooling2D, Flatten
from keras.optimizers import Adam
from keras.utils import plot_model

GPU = True

# =============================================================================
# restrict keras to use define CPU/GPU usage
import tensorflow as tf
from keras import backend as K

num_cores = 8

if GPU:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU}
                       )

session = tf.Session(config=config)
K.set_session(session)
# =============================================================================


# fix random seed for reproducibility
np.random.seed(7)


def create_model(activation='tanh', loss='sparse_categorical_crossentropy', learn_rate=0.01):
    
    # create model
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), activation=activation, input_shape=(28,28,1)))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), activation=activation))
    model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=120, activation=activation))
    model.add(Dense(units=84, activation=activation))
    model.add(Dense(units=10, activation = 'softmax'))
    
    # Compile model
    adam = Adam(lr=learn_rate, decay=0.001)
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    
    # plot out model for the report
    plot_model(model, to_file='cnn_model.png', show_shapes=True)
    
    return model

# =============================================================================
# Load data

X, y = loadlocal_mnist(images_path='mnist/train-images.idx3-ubyte', 
                       labels_path='mnist/train-labels.idx1-ubyte')

X_test, y_test = loadlocal_mnist(images_path='mnist/t10k-images.idx3-ubyte', 
                                 labels_path='mnist/t10k-labels.idx1-ubyte')

# =============================================================================

# normalize data
X = preprocessing.scale(X, axis = 0)
X_test = preprocessing.scale(X_test, axis = 0)

X = X.reshape(X.shape[0], 28,28,1)
X_test = X_test.reshape(X_test.shape[0], 28,28,1) 

model = create_model()
hist = model.fit(X, y, epochs=20, batch_size=64, shuffle=False)
score = model.evaluate(X_test, y_test)
model.summary()
print("Accuracy on test set = ", score[1] * 100 , "% \n")




