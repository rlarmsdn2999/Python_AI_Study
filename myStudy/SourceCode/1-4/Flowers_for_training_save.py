# Daisy, Dandelion, Rose, Sunflower, Tulip
# Any size, any types of images are available as datasets. 
import os
from glob import glob
import numpy as np
import tensorflow as tf
from keras import datasets, layers, models
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

train_dir='D:/AI Study/Datasets/flower_photos/flower_photos'    # Change the path on your PC.
test_dir='D:/AI Study/Datasets/flower_photos/My Flowers'

###################### Hyperparameter Tuning ####################
num_epoch=30                           # iteration of training
batch_size=16                          # better higher for high resolution images
learning_rate=0.001                    # just remember 3M(Mauch, Many, Meticulous)
dropout_rate=0.3                       # cutting connection within 30% in range to prevent overfitting
input_shape=(50, 50, 3)                # resize all of images to the same size
num_class=5                            # number of output(class, label)

########################## Preprocess ############################
train_datagen=ImageDataGenerator(         # Image reconfiguration with "Datagenerator"
    rescale=1./255.,                      # Normalization of image datasets
    width_shift_range=0.3,                # Changing the width of images within 30% in range
    zoom_range=0.2,                       # Resizing within 20% in range
    horizontal_flip=True,                 # Fliping images upside-down
    validation_split=0.2       # Splitting datasets to 80% for train and 20% for validation 
)
test_datagen=ImageDataGenerator(          # For test, No need to reconfigure but rescale
    rescale=1./255.,
)

train_generator=train_datagen.flow_from_directory(   # be also able to load with "DataGenerator"
    train_dir,
    target_size=input_shape[:2],    # (50, 50, 3)  --> (50, 50)
    batch_size=batch_size, 
    color_mode='rgb',
    class_mode='categorical',        # for classification; have to set "Binary" in the case of binary output
    subset='training',               # for train 
)                                    # check the message "Found 60000 images belonging to 10 classes" in terminal

validation_generator=train_datagen.flow_from_directory(       # be also able to load with "DataGenerator"
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size, 
    color_mode='rgb',
    class_mode='categorical',
    subset= 'validation'              # for validation
)                                 

############### Feature Extraction <Convolution Block> ##############
model=models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))  
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(dropout_rate)) 
model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(dropout_rate)) 

################ Fully Connected NN <Neual Net Block> ################
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(dropout_rate)) 
model.add(layers.Dense(num_class, activation='softmax'))

########################  Optimization Block  ########################
model.compile(optimizer=tf.optimizers.Adam(learning_rate), 
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

##############################  Callback  #############################
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

checkpoint = ModelCheckpoint('weights/best.h5', monitor='val_accuracy', save_best_only=True) 
# Saving a model with training if the "acc" would be improved
earlystopping = EarlyStopping( monitor='val_accuracy', patience=20) 
# If the "val_acc" no more improved, training would be stopped.
logger = CSVLogger('weights/history.csv') 
# model(weight..), loss, acc.. would be saved
os.makedirs('weights/', exist_ok=True)
callbacks = [checkpoint, earlystopping, logger]

###########################  Training Block  ##########################
hist = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=num_epoch,
        callbacks=callbacks,
        validation_data=validation_generator,      # No need this part if there are no labeled testset
        validation_steps=len(validation_generator) # No need this part if there are no labeled testset
        )

model.save('weights/last.h5')                      # Saving trained model 
