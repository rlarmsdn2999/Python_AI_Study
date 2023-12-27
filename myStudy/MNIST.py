# The MNIST is a large database of handwritten digits that is commonly used 
# for training various image processing systems.
# The MNIST dataset contains 60,000 28x28 gray-scaled images in 10 different classes.
# the source code is very simple and short. You should memorize all of codes.
import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
(train_images, train_labels), (test_images, test_labels)=datasets.mnist.load_data()
train_images=train_images.reshape((60000, 28, 28, 1))
test_images=test_images.reshape((10000, 28, 28, 1))
train_images, test_images=train_images/255, test_images/255
################ Feature Extraction <Convolution Block> #################
model=models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))  
model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Dropout(0.25))     # disconnecting within 25%
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.Dropout(0.25))     # disconnecting within 25%  
################## Fully Connected NN <Neual Net Block> ###################
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
##########################  <Optimization Block>  ##########################
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=1)   # No dropout, only 1 epoch
##########################  Image Test #####################
test_image=test_images[463, :, :, 0]
plt.title("Number of the Image: {}".format(test_labels[463]))
plt.imshow(test_image)   
plt.show()   # Check the image of the 464th No and the correct answer(label).
##########################  Prediction with Probability ######################
pred=model.predict(test_image.reshape(1, 28, 28, 1))
pred.shape
print(pred)
# check the probabilty list by prediction. 
# The value of 'label 6' (digit no. 6) will be the highest in the list
num=np.argmax(pred)
print("Predicted No: {}".format(num))