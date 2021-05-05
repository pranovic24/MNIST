# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:21:31 2019

@author: janik    
"""

from mlxtend.data import loadlocal_mnist
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.utils import plot_model

# fix random seed for reproducibility
np.random.seed(7)

def create_model(nodes=32, activation='relu', loss='sparse_categorical_crossentropy', learn_rate=0.01):
    # create model
    model = Sequential()
    model.add(Dense(nodes, activation='linear', input_dim=784, kernel_initializer='uniform'))
    model.add(Activation(activation))
    model.add(Dense(nodes, activation='linear', kernel_initializer='uniform'))
    model.add(Activation(activation))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    adam = Adam(lr=learn_rate, decay=0.001)
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    
    # plot out model for the report
    plot_model(model, to_file='mlp_model.png', show_shapes=True)
    
    return model


# =============================================================================
# Load data

X, y = loadlocal_mnist(images_path='mnist/train-images.idx3-ubyte', 
                       labels_path='mnist/train-labels.idx1-ubyte')

X_test, y_test = loadlocal_mnist(images_path='mnist/t10k-images.idx3-ubyte', 
                                 labels_path='mnist/t10k-labels.idx1-ubyte')

# =============================================================================
# train model % report accuracies
accuracies = []
num_nodes = [4, 8, 16, 32, 64, 128, 256]

# normalize data
X = preprocessing.scale(X, axis = 0)
X_test = preprocessing.scale(X_test, axis = 0)

for nodes in num_nodes:
    print("===================== Number of nodes = " , nodes , " ===================== ")
    model = create_model(nodes=nodes)
    hist = model.fit(X, y, epochs=20, batch_size=64, shuffle=False, verbose=0)
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(score[1])
    model.summary()
    print("Accuracy on test set = ", score[1]* 100 , "% \n")


# plot  
fig, ax = plt.subplots(1, 1, figsize=(12,8), dpi = 300)
plt.bar(np.arange(len(num_nodes)), accuracies, align='center', alpha=0.5)
plt.xticks(np.arange(len(num_nodes)), ('4', '8', '16', '32', '64', '128', '256'))
plt.yticks(np.linspace(0.8, 1.0, num=9))
plt.ylim((0.8, 1.0))
ax.set_xlabel('#nodes in hidden layers')
ax.set_ylabel('accuracy')
plt.savefig('mlp_accuracies.png', dpi = 300)
np.savetxt('mlp_accuracies.txt', accuracies, fmt='%.5f')






