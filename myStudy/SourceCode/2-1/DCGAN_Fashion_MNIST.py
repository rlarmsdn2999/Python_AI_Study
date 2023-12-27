# A Very famous Unsupervised Learning method 
# Generative Adversarial Network(GAN)
# AI Composer, Virtual Human, DeepFake, U-GAT-IT, Datasets Generator....

from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

########### Loading Images and Setting Parameters ############
# mnist data set load
(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# Normalize Images to [-1, 1]
train_images = (train_images - 127.5) / 127.5 

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Make Data Batch and Suffle
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

########### Building a Generator Network #############
# Using tf.keras.layers.Conv2DTranspose (Up-Sampling) Layer,  
# Generator starts making Images from the random noise with Seed 
# The first Dense layer is using this seed as an input 
# In the next step, making up-sampling many times to make images with the size of 28x28x1 
# Activation Function: tanh in output layer
# Activation Function: tf.keras.layers.LeakyReLU in the other layers
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, use_bias=False, input_shape=(100,)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(np.prod(28*28), activation='tanh'))
    model.add(layers.Reshape((28,28,1)))
    
    return model

# Try to make images using the untrained Generator
generator = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()

########## Building a Discriminator Network ##############
# Discriminator is a kind of Image-Classfier based on CNN
def make_discriminator_model():
    model = tf.keras.Sequential()
        
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # model.summary()
    return model

# Using the untrained Discriminator, discriminate the generated images are real or fake. 
# This Model will be trained to make an output with positive values for real images, 
# to make an output with negative values for fake images.
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
# print (decision)

############# Define Loss Function and Optimizer ############
# In this Method, Returning helper function to calculating "cross entropy loss"
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

########### Define Loss Function for Discriminator ###########
# In this Method, Scoring the ability of Discriminator how well distinguish fake from real
# Comparing Discriminator's decision for real images to the matrix with elements of 1, 
# Comparing Discriminator's decision for fake images to the matrix with elements of 0. 
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

############## Define Loss Function for Generator ##############
# Scoring the ability of Generator how well cheat the discriminator. 
# If the generator makes good performance, the discriminatr could not distinguish fake(0) from real(1). 
# Comparing Discriminator's dicision to the matrix with elements of 1.
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

######## Generator and Discriminator are to be trained one another, 
######## It is why we have to use diffrent Optimizer each other.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

############### Save Check points ##################
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

################ Define Traning Loop ################
EPOCHS = 1000
noise_dim = 100
num_examples_to_generate = 16

# Use this "Seed" to the end because it is helpful to make GIF Animation.
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Training Loop will be started in the same time to input a randon seed to the generator. 
# The generator will make images with the "seed". 
# The discriminator will distinguish between the real images from datasets 
# and the fake images the generator generate. After calculating "Loss of Model" and "Gradient", 
# Updating the Generator and the Discriminator to make them better.

# Check `tf.function` 
# @ decorator: "compile" function
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Image Generating for GIF Animation
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Saving the model every 15 epoch
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generating after the last epoch is finished
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

################ Generate and Save Images ###################
def generate_and_save_images(model, epoch, test_input):
  # Check that `training = False'.
  # With this, Every layers including batch normailization are acting in inference-mode. 
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  # plt.show()

##################### Model Training #########################
# Call the "train() Method to make the Generator and the Discriminatr train at the same time. 
# Sometimes It is very difficult to make training a "GAN Model"  
# The balance between the generator and the discriminator is very important. 
# You have to check the problem of "Mode Collapse"

train(train_dataset, EPOCHS)

# Restore the last check point
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

################# Generating GAN Images ####################
# Show the generated image in each epoch
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)
