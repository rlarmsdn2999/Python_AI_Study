# Learning by not CNN but ANN. Fast in learninf speed, but worse than CNN in accuracy. 
# from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
 
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# plt.figure()              # You can check the image[0] with deactivating of the below comment line
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(True)
# plt.show()
train_images = train_images / 255.0   # train_images, test_images = train_images/255, test_images/255
test_images = test_images / 255.0
# plt.figure(figsize=(10,10))         # You can check the image from 0 to 25 with deactivating of comment line
# for i in range(25):        
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = keras.Sequential([                 # Compare the source code with the one of MINIST
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)  # check the test accuracy
# print('\nTest Accuracy:', test_acc)

predictions = model.predict(test_images)
test_no=3
print(predictions[test_no])                                                   # predicting with probability
print("Prediction of Image No.: {}".format(np.argmax(predictions[test_no])))   # predicting with image No.
print("Correct No. of Image: {}".format(test_labels[test_no]))

# Visualization with graph
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
# Prediction Reliabilty of the 0th Image. 'Blue': Correct Label, 'Red': Wrong Label
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
# Prediction Reliabilty of the 12th Image
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# Showing 15 test images, predicted labels and correct labels
# 'Blue': Correct, 'Red': Wrong
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# Image prediction with a trained model (select 1 image from testsets)
# img = test_images[0]      # the 0th image
# img = (np.expand_dims(img,0))  # (28,28) -> (1, 28, 28) :adding a dimention for batch-size
# predictions_single = model.predict(img)
# print(predictions_single)
# plot_value_array(0, predictions_single, test_labels)
# _ = plt.xticks(range(10), class_names, rotation=45)
# np.argmax(predictions_single[0])