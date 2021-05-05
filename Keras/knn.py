# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:32:49 2019

@author: janik

Sources:
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
"""

from mlxtend.data import loadlocal_mnist
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Load data

X, y = loadlocal_mnist(images_path='mnist/train-images.idx3-ubyte', 
                       labels_path='mnist/train-labels.idx1-ubyte')

X_test, y_test = loadlocal_mnist(images_path='mnist/t10k-images.idx3-ubyte', 
                                 labels_path='mnist/t10k-labels.idx1-ubyte')

# =============================================================================
# train a KNN using k from 1 to 10 & plot their accuracies

accuracies = []
for k in np.arange(1,11):
    print("--- Training for k = ", k, " ---")
    neigh = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=1, n_jobs=-1 )   # minkowski with p=1 corresponds to manhattan distance, i.e. SAD in our case
    neigh.fit(X, y)
    # uses scipy's stats.mode to find the majority label of the k nearest neighbors
    acc = neigh.score(X_test, y_test)
    accuracies.append(acc)
    
    

# plot  
fig, ax = plt.subplots(1, 1, figsize=(12,8), dpi = 300)
x_coords = np.arange(1,11)
plt.plot(x_coords,accuracies)
ax.grid(zorder=0)
ax.set_xticks(np.arange(1,11))
ax.set_xlabel('k')
ax.set_ylabel('accuracy')
plt.savefig('knn_accuracies.png', dpi = 300)
np.savetxt('knn_accuracies.txt', accuracies, fmt='%.5f')


# =============================================================================
# # prediction example
# import cv2
# image = X_test[5]
# prediction = neigh.predict(image.reshape(1, -1))[0]
# print("Prediction = ", prediction)
# image = image.reshape((28, 28)).astype("uint8")
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# =============================================================================
