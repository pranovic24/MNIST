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
from keras.layers import Dense, Activation, Conv2D, AveragePooling2D, Flatten, LeakyReLU, GlobalAveragePooling2D
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


def create_model(feature_channels=32, loss='sparse_categorical_crossentropy', learn_rate=0.01):
    
    # create model
    model = Sequential()
    
    model.add( Conv2D(filters=feature_channels, dilation_rate=1, kernel_size=(3,3), padding='same', activation='linear', input_shape=(28,28,1)) )
    model.add( LeakyReLU(alpha=0.3) )
    
    model.add( Conv2D(filters=feature_channels, dilation_rate=2, kernel_size=(3,3), padding='same', activation='linear') )
    model.add( LeakyReLU(alpha=0.3) )
    
    model.add( Conv2D(filters=feature_channels, dilation_rate=4, kernel_size=(3,3), padding='same', activation='linear') )
    model.add( LeakyReLU(alpha=0.3) )
    
    model.add( Conv2D(filters=feature_channels, dilation_rate=8, kernel_size=(3,3), padding='same', activation='linear') )
    model.add( LeakyReLU(alpha=0.3) )
    
    model.add( Conv2D(filters=10, dilation_rate=1, kernel_size=(3,3), padding='same', activation='linear') )
    model.add( LeakyReLU(alpha=0.3) )
    
    model.add( GlobalAveragePooling2D(data_format='channels_last') )
    
    model.add(Dense(units=10, activation = 'softmax'))
    
    
    # Compile model
    adam = Adam(lr=learn_rate, decay=0.001)
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    
    # plot out model for the report
    plot_model(model, to_file='can_model.png', show_shapes=True)
    
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

num_cha = [16, 32, 64]
accuracies = []

for fc in num_cha:
    model = create_model(feature_channels=fc)
    hist = model.fit(X, y, epochs=20, batch_size=64, shuffle=False)
    score = model.evaluate(X_test, y_test)
    model.summary()
    print("Accuracy on test set = ", score[1] * 100 , "% \n")        
    accuracies.append(score[1])
    
# plot  
fig, ax = plt.subplots(1, 1, figsize=(12,8), dpi = 300)
plt.bar(np.arange(len(num_cha)), accuracies, align='center', alpha=0.5)
plt.xticks(np.arange(len(num_cha)), ('16', '32', '64'))
plt.yticks(np.linspace(0.95, 1.0, num=9))
plt.ylim((0.95, 1.0))
ax.set_xlabel('#Feature channels')
ax.set_ylabel('accuracy')
plt.savefig('can_accuracies.png', dpi = 300)
np.savetxt('can_accuracies.txt', accuracies, fmt='%.5f')

