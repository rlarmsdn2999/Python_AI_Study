# The CIFAR-10 is a collection of images that are commonly used to train machine learning. 
# It is one of the most widely used datasets for machine learning research.
# The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.
import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
(train_images, train_labels), (test_images, test_labels)=datasets.cifar10.load_data()
# train_images=train_images.reshape((60000, 32, 32, 3))
# test_images=test_images.reshape((10000, 32, 32, 3))
train_images, test_images=train_images/255, test_images/255
################ Feature Extraction <Convolution Block> #################
model=models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))  
# Better check the more parameters such as 'padding', 'stride' and so on. 
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))     # disconnecting within 25%
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Dropout(0.25))     # disconnecting within 25%  
################## Fully Connected NN <Neual Net Block> ###################
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
##########################  <Optimization Block>  ##########################
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=3) 
##########################  Batch Image Test #####################
test_batch=test_images[:31]   # Batch image test from the 0th to the 31th
print(test_batch.shape)       # check the shape: (31, 32, 32, 3)
##########################  Prediction with Probability ######################
pred=model.predict(test_batch)
print(pred.shape)    # 32 test images and 10 labels(classes)
numbers=np.argmax(pred, -1)  # -1: for all of prediction (32 in this case) 
print("Prediction of each Pannel's No: {}".format(numbers))
# 0:Plane   1:Car    2:Bird     3:Cat     4:Deer
# 5:Dog     6:Frog   7:Horse    8:Boat    9:Truck