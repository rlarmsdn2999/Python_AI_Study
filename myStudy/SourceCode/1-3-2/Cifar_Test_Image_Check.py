# The program for checking each images
# Use after run of 'CIFAR10.py' for checking the each image. 

import matplotlib.pyplot as plt
from keras import datasets
cf=datasets.cifar10
(x_train, y_train), (x_test, y_test)=cf.load_data()

start_no_pannel=0   # Predicting with CIFAR10.py, Checking the image with no_pannel
finish_no_pannel=5
for i in range(start_no_pannel, finish_no_pannel):
    plt.title("<Number:{}>".format(y_test[i]))
    plt.imshow(x_test[i])
    plt.show()

# 0:Plane   1:Car    2:Bird     3:Cat     4:Deer
# 5:Dog     6:Frog   7:Horse    8:Boat    9:Truck