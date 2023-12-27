# 2016년에 가장 관심을 많이 받았던 비감독(Unsupervised) 학습 방법인
# Generative Adversarial Network(GAN)을 구현해 보자.

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

########### 이미지 로드 및 파라메터 설정 ############
# mnist data set load
(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# 이미지를 [-1, 1]로 정규화합니다.
train_images = (train_images - 127.5) / 127.5 

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# 데이터 배치를 만들고 섞습니다.
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

########### Generator Network 구성 #############
# 생성자는 시드값 (seed; 랜덤한 잡음)으로부터 이미지를 생성하기 위해, 
# tf.keras.layers.Conv2DTranspose (업샘플링) 층을 이용함. 
# 처음 Dense층은 이 시드값을 인풋으로 받는다. 
# 그 다음 원하는 사이즈 28x28x1의 이미지가 나오도록 업샘플링을 여러번 한다. 
# tanh를 사용하는 마지막 층을 제외한 나머지 각 층마다 활성함수로 
# tf.keras.layers.LeakyReLU을 사용하고 있음을 주목.
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

# (아직 훈련이 되지않은) 생성자를 이용해 이미지를 생성해 본다.
generator = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()

########## Discriminator Network 구성 ##############
# 감별자(Discriminator)는 CNN 기반의 이미지 분류기이다.
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

# (아직까지 훈련이 되지 않은) 감별자를 사용하여, 생성된 이미지가 
# 진짜인지 가짜인지 판별한다. 모델은 진짜 이미지에는 양수의 값 (positive values)을
# 가짜 이미지에는 음수의 값 (negative values)을 출력하도록 훈련되어진다.
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
# print (decision)

####### 손실함수 및 옵티마이저 정의 #########
# 이 메서드는 cross entropy loss를 계산하기 위해 helper함수를 반환.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

####### 감별자 손실함수 #########
# 이 메서드는 감별자가 가짜 이미지에서 얼마나 진짜 이미지를 잘 판별하는지 수치화한다. 
# 진짜 이미지에 대한 감별자의 예측과 1로 이루어진 행렬을 비교하고, 
# 가짜 (생성된) 이미지에 대한 감별자의 예측과 0으로 이루어진 행렬을 비교한다.
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

######## 생성자 손실함수 #########
# 생성자의 손실함수는 감별자를 얼마나 잘 속였는지에 대해 수치화한다. 
# 직관적으로 생성자가 원활히 수행되고 있다면, 감별자는 가짜 이미지를 
# 진짜 (또는 1)로 분류를 할 것이다. 여기서 우리는 생성된 이미지에 대한 
# 감별자의 결정을 1로 이루어진 행렬과 비교할 것이다.
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# 감별자와 생성자는 따로 훈련되기 때문에, 감별자와 생성자의 옵티마이저는 다르다.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

###### 체크포인트 저장 #######
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

######### 훈련 루프 정의하기 ##########
EPOCHS = 1000
noise_dim = 100
num_examples_to_generate = 16

# 이 시드를 계속 재활용함. 
# (GIF 애니메이션에서 진전 내용을 시각화하는데 쉽기 때문이다.) 
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# 훈련 루프는 생성자가 입력으로 랜덤시드를 받는 것으로부터 시작된다. 
# 그 시드값을 사용하여 이미지를 생성한다. 
# 감별자를 사용하여 (훈련 세트에서 갖고온) 진짜 이미지와 
# (생성자가 생성해낸) 가짜이미지를 분류한다. 각 모델의 손실을 계산하고, 
# 그래디언트 (gradients)를 사용해 생성자와 감별자를 업데이트한다.

# `tf.function`이 어떻게 사용되는지 주목.
# 이 데코레이터는 함수를 "컴파일"한다.
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

    # GIF를 위한 이미지를 바로 생성한다.
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # 15 에포크가 지날 때마다 모델을 저장한다.
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    
    # print (' 에포크 {} 에서 걸린 시간은 {} 초 이다'.format(epoch +1, time.time()-start))
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # 마지막 에포크가 끝난 후 생성한다.
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

######## 이미지 생성 및 저장 ##########
def generate_and_save_images(model, epoch, test_input):
  # `training`이 False로 맞춰진 것을 주목.
  # 이렇게 하면 (배치정규화를 포함하여) 모든 층들이 추론 모드로 실행된다. 
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  # plt.show()

########## 모델 훈련 #############
# 위에 정의된 train() 메서드를 생성자와 감별자를 동시에 훈련하기 위해 호출한다. 
# 생성적 적대 신경망을 학습하는 것은 매우 까다로울 수 있다. 
# 생성자와 감별자가 서로를 제압하지 않는 것이 중요하다. 
# (예를 들어 학습률이 비슷하면 한쪽이 우세해진다.) 
# 훈련 초반부에는 생성된 이미지는 랜덤한 노이즈처럼 보인다. 
# 훈련이 진행될수록, 생성된 숫자는 점차 진짜처럼 보일 것이다. 
# 약 50 에포크가 지난 후, MNIST 숫자와 닮은 이미지가 생성된다. 

train(train_dataset, EPOCHS)

# 마지막 체크포인트를 복구.
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

####### GIF 생성 #########
# 에포크 숫자를 사용하여 하나의 이미지를 보여준다.
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

# imageio로 훈련 중에 저장된 이미지를 사용해 GIF 애니메이션을 만든다.
# anim_file = 'dcgan.gif'

# with imageio.get_writer(anim_file, mode='I') as writer:
#   filenames = glob.glob('image*.png')
#   filenames = sorted(filenames)
#   last = -1
#   for i,filename in enumerate(filenames):
#     frame = 2*(i**0.5)
#     if round(frame) > round(last):
#       last = frame
#     else:
#       continue
#     image = imageio.imread(filename)
#     writer.append_data(image)
#   image = imageio.imread(filename)
#   writer.append_data(image)

# import IPython
# if IPython.version_info > (6,2,0,''):
#   display.Image(filename=anim_file)