# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 22:49:44 2020
@author: user
"""
from keras.models import load_model
import tensorflow as tf
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

data_path=glob('dataset/test/*.jpg') # Path for test images

########################  load trained model ####################
path_model = 'weights/best.h5'       # Path for the trained model 
model = load_model(path_model)       # Loading the trained model

########################  Prediction with Probability ####################
np.random.shuffle(data_path)        # shuffle the path of test images
input_shape=(50, 50, 3)

def read_image(path):               # Function for read images
    gfile=tf.io.read_file(path)     # Reading a image in the Path and input to "gfile"
    image=tf.io.decode_image(gfile, dtype=tf.float32)  # Decoding the image to an image array
    return image

for test_no in range(20):       # Image No. for test
    path=data_path[test_no]     # the Path of the "test_no+1"th file
    
    img=read_image(path)
    img=tf.image.resize(img, input_shape[:2])  # Resizing the image size to the same size as input-image 

    image=np.array(img)   # making the image to an image array in order to check the image through "imshow"
    # print(image.shape)  # Check the shape of an image ; shape (50,50,3)
    plt.imshow(image)
    plt.title('Check the Image and Predict Together!')
    plt.show()
       
    test_image=image[tf.newaxis,...]   # Make the shape of test-images;  (50, 50) --> (1, 50, 50, 3)
    pred=model.predict(test_image)         
    print(pred)                        # Predicting the "test_no"th image with probability (One-Hot Encoding)
    num=np.argmax(pred)                # Checking where the maximum value is in probability array 
    if num==0:
        print("Maybe Daisy") 
    elif num==1:
        print("Maybe Dandelion")
    elif num==2:
        print("Maybe Rose") 
    elif num==3:
        print("Maby Sunflower")
    elif num==4:
        print("Maybe Tulip")
    # else:
    #     print("I don't know!") 